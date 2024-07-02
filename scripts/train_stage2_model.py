# 需要准备的流程

# harmony pipeline ; 提供 pipeline_path

# 需要训练的模型 stage2 model ; 选择 autoencoder kl ; 两种加载方式：可以从 config 初始化，可以指定 pretrained_model_path 加载一个 vae ；定为 256px 版本的模型

# 数据集 ;
# - real 和 comp 经过 random crop 之后，保留 512px 和 256px 两个版本；512px 的 comp 输入 harmony pipeline , 得到 harm image 512px ; harm image 512px resize 到 256px , 和 mask 256px , comp 256px 输入 stage2 model ; 训练 target 为 real 256px
# - (backup) real 和 comp 只准备 256px 的版本 ；comp 256px 需要 resize 到 512px , 输入 harmony pipeline , 之后的流程相同
# - - 注：这个方案是为了保证公平性设置的，但是目前版本的模型在训练时候已经使用了 original -> 512px 的图片直接作为输入 , 所以不公平性已经存在 ；这个方案只有在重新训练模型之后才会采用

import argparse
import logging
import math
import os
import json
from accelerate import DistributedDataParallelKwargs, DistributedType

import torch
import torchvision.transforms.functional as TF
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
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from tqdm.auto import tqdm
from src.dataset.ihd_dataset import IhdDatasetMultiRes as Dataset
from src.utils import EMAModel
from src.pipelines.pipeline_stable_diffusion_harmony import StableDiffusionHarmonyPipeline
from src.models.unet_2d import UNet2DCustom
from src.models.condition_vae import ConditionVAE

model_cls = UNet2DCustom
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


def harm_image_postprocess(harm_image, comp_image, mask_image):
    assert isinstance(harm_image, torch.Tensor)
    assert isinstance(comp_image, torch.Tensor)
    assert isinstance(mask_image, torch.Tensor)

    assert harm_image.shape == comp_image.shape
    assert mask_image.shape[0] == harm_image.shape[0]
    assert mask_image.shape[1] == 1
    assert mask_image.shape[2:] == harm_image.shape[2:]

    bin_mask = (mask_image >= 0.5).float()
    fused_image = harm_image * bin_mask + comp_image * (1 - bin_mask)
    fused_image = fused_image.clamp(-1, 1)

    return fused_image


from src.utils import make_stage2_input, tensor_to_pil


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model
        if isinstance(model, AutoencoderKL):
            self.forward = self.vae_forward
        elif isinstance(model, UNet2DCustom):
            self.forward = self.unet2dcustom_forward
        else:
            raise ValueError("Unsupported model type")

    # for autoencoder kl
    def vae_forward(self, input):
        self.model: AutoencoderKL
        posterior = self.model.encode(input).latent_dist
        z = posterior.sample()
        kl_div = posterior.kl()
        model_pred = self.model.decode(z).sample
        return model_pred, kl_div

    # for Unet2dCustom
    def unet2dcustom_forward(
        self,
        input,
    ):
        self.model: UNet2DCustom
        model_pred = self.model(input).sample
        return model_pred


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
            "--pretrained_vae_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--pretrained_unet_path",
            type=str,
            default=None,
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
        parser.add_argument("--num_train_epochs", type=int, default=100)
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
            "--lr_warmup_ratio",
            type=float,
            default=None,
            help="Ratio of steps for the warmup in the lr scheduler.",
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
            "--infer_resolution",
            type=int,
            default=256,
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
            "--mask_dilate",
            type=int,
            default=0,
            help="mask_dilate",
        )
        parser.add_argument(
            "--train_file",
            type=str,
            default="constant",
            help=("train dataset file"),
        )
        parser.add_argument(
            "--test_file",
            type=str,
            default="constant",
            help=("train dataset file"),
        )
        parser.add_argument(
            "--in_channels",
            type=int,
            choices=[3, 4, 7],
            default=3,
        )
        parser.add_argument(
            "--kl_div_weight",
            type=float,
            default=1e-5,
        )
        parser.add_argument(
            "--ema_decay",
            type=float,
            default=0.9999,
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

    return args


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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # TODO : 确认 pipeline 正确加载
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.pretrained_vae_path is None:
        args.pretrained_vae_path = os.path.join(args.pipeline_path, "vae")
    accelerator.print(f"use vae from <<{args.pretrained_vae_path}>>")
    if ("cvae" in args.pretrained_vae_path) or ("condition_vae" in args.pretrained_vae_path):
        vae_cls = ConditionVAE
    else:
        vae_cls = AutoencoderKL
    vae = vae_cls.from_pretrained(
        args.pretrained_vae_path,
        torch_dtype=weight_dtype,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_unet_path,
        torch_dtype=weight_dtype,
    )

    harmony_pipeline = StableDiffusionHarmonyPipeline.from_pretrained(
        args.pipeline_path,
        vae=vae,
        unet=unet,
        torch_dtype=weight_dtype,
        # device_map={0: accelerator.device},
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    harmony_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        harmony_pipeline.scheduler.config
    )
    harmony_pipeline.set_progress_bar_config(disable=True)
    harmony_pipeline.to(accelerator.device)

    if args.model_path.endswith(".json"):
        # 从 .json 的 config 初始化 model
        model_config = model_cls.load_config(args.model_path)
        model_config["in_channels"] = args.in_channels
        model = model_cls.from_config(model_config)

    elif os.path.isdir(args.model_path):
        # folder , 加载一个已经存在的 model
        def check_files_existence(folder_path):
            config_json_path = os.path.join(folder_path, "config.json")
            pytorch_model_bin_path = os.path.join(
                folder_path, "diffusion_pytorch_model.bin"
            )

            if not (
                os.path.exists(config_json_path)
                and os.path.exists(pytorch_model_bin_path)
            ):
                raise FileNotFoundError(
                    "Either 'config.json' or 'diffusion_pytorch_model.bin' is missing in the folder."
                )

        check_files_existence(args.model_path)
        model = model_cls.from_pretrained(
            args.model_path,
        )

    in_channels = model.config.in_channels
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
            harmony_pipeline.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    model = ModelWrapper(model)
    print_trainable_parameters(model)

    # NOTE : create ema for all trainable params
    if args.use_ema:
        ema_model = EMAModel(get_trainable_parameters(model), decay=args.ema_decay)
        accelerator.register_for_checkpointing(ema_model)

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

    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    #     num_training_steps=args.max_train_steps * accelerator.num_processes,
    # )

    with accelerator.main_process_first():
        train_dataset = Dataset(
            "train", None, [args.resolution, args.infer_resolution], args
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

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    overide_lr_warmup_steps = False
    if args.lr_warmup_ratio is not None:
        overide_lr_warmup_steps = True
        args.lr_warmup_steps = math.ceil(
            args.lr_warmup_ratio * (args.max_train_steps // accelerator.num_processes)
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.checkpointing_epochs is not None:
        args.checkpointing_steps = (
            args.checkpointing_epochs * num_update_steps_per_epoch
        )

    args_to_save = args
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
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )
    start_step = first_epoch * num_update_steps_per_epoch
    if args.use_ema:
        ema_model.to(accelerator.device)

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
                    # NOTE : batch 的构造逻辑：key 为 res (256,512) , 对应一个 dict 包含 (comp, mask, real, caption_ids)

                    # NOTE : 当输入 pipeline torch tensor 的时候， pipeline 可以正确处理
                    # stage1 : inference stage

                    infer_batch = batch[args.infer_resolution]
                    with torch.no_grad():
                        harm_image = harmony_pipeline(
                            prompt=[""] * args.train_batch_size,
                            image=infer_batch["comp"],
                            mask_image=infer_batch["mask"],
                            # generator=generator,
                            height=args.infer_resolution,
                            width=args.infer_resolution,
                            guidance_scale=1.0,
                            num_inference_steps=5,
                            output_type="numpy",
                        ).images

                        # NOTE : 输出是 (b,h,w,c) 的 numpy ndarray , [0,1] 区间 value
                        harm_image = (
                            torch.tensor(harm_image)
                            .to(accelerator.device)
                            .permute(0, 3, 1, 2)
                            .sub(0.5)
                            .multiply(2)
                        )  # convert to [-1,1] (b,c,h,w,) pytorch tensor

                        if harm_image.shape[-2:] != torch.Size(
                            (args.resolution, args.resolution)
                        ):
                            harm_image = TF.resize(
                                harm_image,
                                size=[args.resolution, args.resolution],
                                antialias=True,
                            ).clamp(-1,1) # [-1,1] (b,c,h,w,) pytorch tensor

                    train_batch = batch[args.resolution]
                    # harm_image = harm_image_postprocess(
                    #     harm_image, train_batch["comp"], train_batch["mask"]
                    # )  # [-1,1] (b,c,h,w,) pytorch tensor

                    # stage 2 : restoration stage
                    stage2_input = make_stage2_input(
                        harm_image,
                        train_batch["comp"],
                        train_batch["mask"],
                        in_channels,
                    )
                    stage2_input = stage2_input.detach()
                    # model_pred, kl_div = model(stage2_input)
                    model_pred = model(stage2_input)
                    target = batch[args.resolution]["real"]

                    # loss = (
                    #     F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    #     + args.kl_div_weight * kl_div.mean()
                    # )
                    # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss = F.mse_loss(
                        model_pred.float(),
                        target.float(),
                        reduction="mean",
                    )

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

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(get_trainable_parameters(model))
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {"train_loss": train_loss}, step=global_step
                )
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

                        model_to_save = unwrapped_model.model
                        model_save_dir = os.path.join(
                            args.output_dir, f"weights-{global_step}"
                        )
                        os.makedirs(model_save_dir, exist_ok=True)
                        model_to_save.save_pretrained(model_save_dir)

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

            model_to_save = unwrapped_model.model
            model_save_dir = os.path.join(args.output_dir, f"weights-{global_step}")
            os.makedirs(model_save_dir, exist_ok=True)
            model_to_save.save_pretrained(model_save_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
