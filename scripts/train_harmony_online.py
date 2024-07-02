import argparse
import logging
import math
import os
import json
from accelerate import DistributedDataParallelKwargs, DistributedType

import copy
import torch
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers import __version__

print(f"you are using diffusers {__version__}")

# import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import UNet2DConditionModel
from diffusers import (
    EulerAncestralDiscreteScheduler,
)
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from utils import EMAModel


from tqdm.auto import tqdm

# from src.dataset.image_folder_ds import ImageFolderDataset as Dataset
# from src.dataset.ihd_dataset import IhdWithRandomMask as Dataset
from src.dataset.ihd_dataset import IhdWithRandomMaskComp as Dataset
from src.diffusers_overwrite import UNet2DCustom
from pipeline_stable_diffusion_harmony import StableDiffusionHarmonyPipeline

# from utils import comp_method_mapping
from utils import select_cand, make_comp
from consistencydecoder import ConsistencyDecoder


model_cls = UNet2DConditionModel
logger = get_logger(__name__)


def get_trainable_parameters(model):
    trainable_parameters = [p for p in model.parameters() if p.requires_grad == True]
    return trainable_parameters


def print_trainable_parameters(model):
    trainable_parameters = get_trainable_parameters(model)
    size = total_params = sum(p.numel() for p in trainable_parameters)
    units = ["B", "K", "M", "G"]
    unit_index = 0
    while size > 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    print(f"total trainable params : {size:.3f}{units[unit_index]}")


from typing import List

def mix_comp_data(
    online_comp: torch.Tensor,
    local_comp: torch.Tensor,
    online_data_prob: float,
    select_mask: torch.Tensor,
):
    """
    online_comp : (b,c,h,w)
    local_comp : (b,c,h,w)
    """
    bs = len(online_comp)
    # select_mask = torch.tensor(
    #     [(subset_name in subset_to_mix) for subset_name in subset_names],
    #     dtype=torch.bool,
    # )
    select_mask = (torch.rand(size=(bs,)).to(select_mask.device) < online_data_prob) & select_mask # (b,)
    select_mask = select_mask[..., None, None, None].to(online_comp)

    comp = select_mask * online_comp + (1 - select_mask) * local_comp.to(online_comp)
    return comp


from PIL import Image
import numpy as np


def tensor_to_pil(tensor: torch.Tensor, mode="RGB"):
    if tensor.dim() == 3:
        tensor = tensor[None, ...]
    # make sure input is (b,c,h,w)
    if mode == "RGB":
        image_np = (
            tensor.permute(2, 3, 1).add(1).multiply(127.5).numpy().astype(np.uint8)
        )
        image = [Image.fromarray(image_np_i, mode=mode) for image_np_i in image_np]
    elif mode == "1":
        image_np = tensor.squeeze().multiply(255).numpy().astype(np.uint8)
        if tensor.dim() == 2:
            tensor = tensor[None, ...]
        image = [Image.fromarray(image_np_i).convert("1") for image_np_i in image_np]
    else:
        raise ValueError(f"not supported mode {mode}")
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="")
    if True:  # * args to be used ; just to wrap them
        parser.add_argument(
            "--pipeline_path",
            type=str,
            default=None,
            required=True,
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=None,
            required=True,
            help="如果指向一个 json 文件，则为配置文件，从 config 初始化模型；如果指向一个文件夹，则 from pretrained , 加载一个已有的模型，确认文件夹中有 config 和 pytorch_model.bin",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="sd-model-finetuned",
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument(
            "--seed", type=int, default=None, help="A seed for reproducible training."
        )
        parser.add_argument(
            "--dataloader_num_workers",
            type=int,
            default=8,
            help=(
                "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
            ),
        )
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=16,
            help="Batch size (per device) for the training dataloader.",
        )
        parser.add_argument(
            "--max_train_steps",
            type=int,
            default=None,
            help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--gradient_checkpointing",
            action="store_true",
            help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Initial learning rate (after the potential warmup period) to use.",
        )
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="constant",
            help=(
                'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                ' "constant", "constant_with_warmup"]'
            ),
        )
        parser.add_argument(
            "--lr_warmup_steps",
            type=int,
            default=500,
            help="Number of steps for the warmup in the lr scheduler.",
        )
        parser.add_argument(
            "--lr_warmup_ratio", type=float, default=None, help="Ratio of steps for the warmup in the lr scheduler."
        )
        parser.add_argument(
            "--use_8bit_adam",
            action="store_true",
            help="Whether or not to use 8-bit Adam from bitsandbytes.",
        )
        parser.add_argument(
            "--use_ema", action="store_true", help="Whether to use EMA model."
        )
        parser.add_argument(
            "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
        )
        parser.add_argument(
            "--mixed_precision",
            type=str,
            default=None,
            choices=["no", "fp16", "bf16"],
            help=(
                "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
            ),
        )
        parser.add_argument(
            "--checkpointing_steps",
            type=int,
            default=500,
            help=(
                "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
                " training using `--resume_from_checkpoint`."
            ),
        )
        parser.add_argument(
            "--checkpointing_epochs",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--checkpoints_total_limit",
            type=int,
            default=3,
            help=("Max number of checkpoints to store."),
        )
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help=(
                "Whether training should be resumed from a previous checkpoint. Use a path saved by"
                ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
            ),
        )
        parser.add_argument(
            "--enable_xformers_memory_efficient_attention",
            action="store_true",
            help="Whether or not to use xformers.",
        )
        parser.add_argument(
            "--dataset_root",
            type=str,
            nargs="+",
            default=None,
        )
        parser.add_argument(
            "--resolution",
            type=int,
            default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--random_flip",
            action="store_true",
            default=False,
            help="whether to randomly flip images horizontally",
        )
        parser.add_argument(
            "--random_crop",
            action="store_true",
            default=False,
            help="random_crop",
        )
        parser.add_argument(
            "--center_crop",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--nofg",
            action="store_true",
            default=False,
            help="do not use foreground mask when computing loss",
        )
        parser.add_argument(
            "--ema_decay",
            type=float,
            default=0.9999,
        )
        parser.add_argument(
            "--mask_gen_config",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--image_log_interval",
            type=int,
            default=100,
        )
        parser.add_argument(
            "--image_mask_mapping",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--stage2_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--mask_comp_mapping",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--online_mix_rate",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--comp_method",
            type=str,
            choices=["v1", "v2-fgfg", "v2-fgbg"],
            default="v1",
        )
        parser.add_argument(
            "--refer_method",
            type=str,
            choices=["batch"],
            default="batch",
        )
        parser.add_argument(
            "--train_file",
            type=str,
            default=None,
        )
        parser.add_argument("--use_consistency_decoder", action="store_true")
        parser.add_argument(
            "--mix_area_thres",
            type=float,
            default=0.,
        )

    if True:  # * args set to default ; just to wrap them
        # NOTE : not to use , set as default
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            required=False,
            help="Revision of pretrained model identifier from huggingface.co/models.",
        )
        parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help=(
                "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
                " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
                " or to a folder containing files that 🤗 Datasets can understand."
            ),
        )
        parser.add_argument(
            "--max_train_samples",
            type=int,
            default=None,
            help=(
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            ),
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="The directory where the downloaded models and datasets will be stored.",
        )
        parser.add_argument("--num_train_epochs", type=int, default=100)
        parser.add_argument(
            "--scale_lr",
            action="store_true",
            default=False,
            help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
        )
        parser.add_argument(
            "--allow_tf32",
            action="store_true",
            help=(
                "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
            ),
        )
        parser.add_argument(
            "--non_ema_revision",
            type=str,
            default=None,
            required=False,
            help=(
                "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
                " remote repository specified with --pretrained_model_name_or_path."
            ),
        )
        parser.add_argument(
            "--adam_beta1",
            type=float,
            default=0.9,
            help="The beta1 parameter for the Adam optimizer.",
        )
        parser.add_argument(
            "--adam_beta2",
            type=float,
            default=0.999,
            help="The beta2 parameter for the Adam optimizer.",
        )
        parser.add_argument(
            "--adam_epsilon",
            type=float,
            default=1e-08,
            help="Epsilon value for the Adam optimizer",
        )
        parser.add_argument(
            "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
        )
        parser.add_argument(
            "--push_to_hub",
            action="store_true",
            help="Whether or not to push the model to the Hub.",
        )
        parser.add_argument(
            "--hub_token",
            type=str,
            default=None,
            help="The token to use to push to the Model Hub.",
        )
        parser.add_argument(
            "--hub_model_id",
            type=str,
            default=None,
            help="The name of the repository to keep in sync with the local `output_dir`.",
        )
        parser.add_argument(
            "--project_dir",
            type=str,
            default="logs",
            help=(
                "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
            ),
        )
        parser.add_argument(
            "--report_to",
            type=str,
            default="tensorboard",
            help=(
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
            ),
        )
        parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="For distributed training: local_rank",
        )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    assert not (args.random_crop and args.center_crop)

    return args


def convert_args(args):
    args.dataset_root = "_|_".join(args.dataset_root)
    return args


class DynamicEMAModel(EMAModel):

    """重写了 EMA model , 使其绑定一个模型的参数，能够实时更新；重写 to 函数，禁用；重写 load , 加载 shadow_params 的时候覆写绑定的 base 模型的参数"""

    def __init__(
        self,
        base_model: torch.nn.Module,
        decay: float = 0.9999,
        min_decay: float = 0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float | int = 1,
        power: float | int = 2 / 3,
        **kwargs,
    ):
        self.base_model = base_model
        self.temp_stored_params = None
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`

    @property
    def shadow_params(self):
        return list(self.base_model.parameters())

    def to(self, device=None, dtype=None) -> None:
        raise NotImplementedError("please move base model in advance")

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get(
            "optimization_step", self.optimization_step
        )
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get(
            "update_after_step", self.update_after_step
        )
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")
            for s_param, s_param_loaded in zip(self.shadow_params, shadow_params):
                s_param.data.copy_(s_param_loaded.to(s_param.device).data)


def main():
    args = parse_args()
    project_dir = os.path.join(args.output_dir, args.project_dir)

    ddp_kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=project_dir,
        kwargs_handlers=[ddp_kwargs],
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # make_comp = comp_method_mapping[args.comp_method]

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # TODO : 加载 pipeline
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    unet = UNet2DConditionModel.from_pretrained(
        args.model_path, torch_dtype=weight_dtype,
    )
    if args.use_consistency_decoder:
        consistency_decoder = ConsistencyDecoder(
            device=accelerator.device
        )  # Model size: 2.49 GB
    else:
        consistency_decoder = None
    harmony_pipeline = StableDiffusionHarmonyPipeline.from_pretrained(
        args.pipeline_path,
        unet=unet,
        consistency_decoder=consistency_decoder,
        torch_dtype=weight_dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    harmony_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        harmony_pipeline.scheduler.config
    )
    harmony_pipeline.set_progress_bar_config(disable=True)
    harmony_pipeline.to(accelerator.device)

    noise_scheduler = DDPMScheduler.from_config(harmony_pipeline.scheduler.config)
    tokenizer = harmony_pipeline.tokenizer
    text_encoder = harmony_pipeline.text_encoder
    vae = harmony_pipeline.vae

    if args.stage2_path:
        stage2_model = UNet2DCustom.from_pretrained(
            args.stage2_path,
            torch_dtype=weight_dtype,
        )
        stage2_model.to(accelerator.device)
        stage2_model.eval()
        stage2_model.requires_grad_(False)
    else:
        stage2_model = None

    model = UNet2DConditionModel.from_pretrained(
        args.model_path,
    )
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
            harmony_pipeline.enable_xformers_memory_efficient_attention()
            print("use xformers")
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("use gradient checkpointing")

    
    print_trainable_parameters(model)

    # NOTE : create ema for all trainable params
    if args.use_ema:
        ema_model = DynamicEMAModel(harmony_pipeline.unet, decay=args.ema_decay)
        accelerator.register_for_checkpointing(ema_model)
        # ema_model.to(accelerator.device)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        get_trainable_parameters(model),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    with accelerator.main_process_first():
        train_dataset = Dataset(
            tokenizer,
            args,
        )

    # DataLoaders creation:
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    overide_lr_warmup_steps = False
    if args.lr_warmup_ratio is not None:
        overide_lr_warmup_steps=True
        args.lr_warmup_steps = math.ceil(args.lr_warmup_ratio * (args.max_train_steps // accelerator.num_processes))
        
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.checkpointing_epochs is not None:
        args.checkpointing_steps = args.checkpointing_epochs * num_update_steps_per_epoch
        
    args_to_save = convert_args(args)
    if accelerator.is_main_process:
        accelerator.init_trackers("diffharmony-dev", config=vars(args_to_save))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            # if args.use_ema:
            #     ema_model.to(accelerator.device)

            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    start_step = first_epoch * num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(start_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    json.dump(
        vars(args_to_save),
        open(os.path.join(args.output_dir, "args.json"), "w"),
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    real = batch["image"]  # (b,c,h,w) [-1,1] tensor
                    caption_ids = batch["caption_ids"]
                    mask = batch["mask"]

                    real = real.to(weight_dtype)
                    mask = mask.to(weight_dtype)

                    if args.refer_method == "batch":
                        cand_indices = select_cand(real, mask, args.comp_method)
                        cand = real[cand_indices]
                        cand_mask = mask[cand_indices]
                    else:
                        cand = batch["refer_image"]
                        cand_mask = batch["refer_mask"]
                    cand = cand.to(real)
                    cand_mask = cand_mask.to(mask)

                    online_comp = make_comp(
                        real,
                        cand,
                        mask,
                        args.comp_method,
                        harmony_pipeline,
                        stage2_model=stage2_model,
                        cand_mask=cand_mask,
                    )
                    local_comp = batch["comp"]
                    comp = mix_comp_data(
                        online_comp, local_comp, args.online_mix_rate, batch["select_mask"]
                    )

                    # Convert images to latent space
                    # =================================================================
                    latents = vae.encode(
                        real.to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    masked_image_latents = vae.encode(
                        comp.to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    masked_image_latents = (
                        masked_image_latents * vae.config.scaling_factor
                    )

                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                    downsampled_mask = torch.nn.functional.interpolate(
                        mask,
                        size=(
                            comp.shape[2] // vae_scale_factor,
                            comp.shape[3] // vae_scale_factor,
                        ),
                    )

                    # =================================================================
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    latent_model_input = torch.cat(
                        [noisy_latents, downsampled_mask, masked_image_latents], dim=1
                    )

                    encoder_hidden_states = text_encoder(caption_ids)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )
                    # Predict the noise residual and compute loss
                    model_pred = model(
                        latent_model_input, timesteps, encoder_hidden_states
                    ).sample

                    # Compute loss
                    # =================================================================================================
                    if args.nofg:
                        # Compute the loss without foreground mask
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                    else:
                        loss = F.mse_loss(
                            model_pred.float() * downsampled_mask,
                            target.float() * downsampled_mask,
                            reduction="none",
                        ).mean(dim=(2, 3), keepdim=True)
                        loss_scale = (
                            (downsampled_mask.shape[2] * downsampled_mask.shape[3])
                            / downsampled_mask.sum(dim=(2, 3), keepdim=True)
                        ).clamp(1, 5)
                        loss = loss * loss_scale
                        loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        get_trainable_parameters(model), args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(get_trainable_parameters(model))
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if (global_step % args.checkpointing_steps == 0) or (
                    global_step == args.max_train_steps
                ):
                    # 让主进程执行一些清理工作
                    if (
                        accelerator.is_main_process
                        and args.checkpoints_total_limit is not None
                    ):
                        import shutil

                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (
                                len(checkpoints) - args.checkpoints_total_limit + 1
                            )
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint
                                )
                                shutil.rmtree(removing_checkpoint)

                    if (
                        accelerator.distributed_type != DistributedType.DEEPSPEED
                        and accelerator.is_main_process
                    ):
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    elif accelerator.distributed_type == DistributedType.DEEPSPEED:
                        # NOTE : deepspeed 需要在每个进程调用保存函数；check it
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)

                    # NOTE : 保存所有 learnable weights
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        if args.use_ema:
                            ema_model.copy_to(get_trainable_parameters(unwrapped_model))

                        model_to_save = unwrapped_model
                        model_save_dir = os.path.join(
                            args.output_dir, f"weights-{global_step}"
                        )
                        os.makedirs(model_save_dir, exist_ok=True)
                        model_to_save.save_pretrained(
                            model_save_dir, safe_serialization=False
                        )

                if (
                    accelerator.is_main_process
                    and global_step % args.image_log_interval == 0
                ):
                    #  保存本轮次的 real , mask , comp
                    image_logging_dir = os.path.join(args.output_dir, "images")
                    if not os.path.exists(image_logging_dir):
                        # 如果文件夹不存在，则创建它
                        os.makedirs(image_logging_dir)
                    bs = len(real)
                    nrow = bs // int(math.sqrt(bs))
                    real_to_save = make_grid(
                        real, nrow=nrow, normalize=True, value_range=(-1, 1)
                    )
                    comp_to_save = make_grid(
                        comp, nrow=nrow, normalize=True, value_range=(-1, 1)
                    )
                    mask_to_save = make_grid(
                        mask,
                        nrow=nrow,
                    )
                    save_image(
                        real_to_save,
                        os.path.join(image_logging_dir, f"s{global_step}_real.jpg"),
                    )
                    save_image(
                        comp_to_save,
                        os.path.join(image_logging_dir, f"s{global_step}_comp.jpg"),
                    )
                    save_image(
                        mask_to_save,
                        os.path.join(image_logging_dir, f"s{global_step}_mask.jpg"),
                    )

                    # 保存 harm image
                    h, w = comp.shape[-2:]
                    with torch.no_grad():
                        harm_image = harmony_pipeline(
                            prompt=[""] * bs,
                            image=comp,
                            mask_image=mask,
                            height=h,
                            width=w,
                            guidance_scale=1.0,
                            num_inference_steps=5,
                            output_type="numpy",
                        ).images
                    # output (b,h,w,c) 的 numpy ndarray , [0,1] 区间 value
                    harm_image = (
                        torch.tensor(harm_image)
                        .permute(0, 3, 1, 2)
                        .sub(0.5)
                        .multiply(2)
                    )  # convert to [-1,1] (b,c,h,w,) pytorch tensor
                    harm_to_save = make_grid(
                        harm_image, nrow=nrow, normalize=True, value_range=(-1, 1)
                    )
                    save_image(
                        harm_to_save,
                        os.path.join(image_logging_dir, f"s{global_step}_harm.jpg"),
                    )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # NOTE : 保存所有 learnable weights
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            if args.use_ema:
                ema_model.copy_to(get_trainable_parameters(unwrapped_model))

            model_to_save = unwrapped_model
            model_save_dir = os.path.join(args.output_dir, f"weights-{global_step}")
            os.makedirs(model_save_dir, exist_ok=True)
            model_to_save.save_pretrained(model_save_dir, safe_serialization=False)

    accelerator.end_training()


if __name__ == "__main__":
    main()
