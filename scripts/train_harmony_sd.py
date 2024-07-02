import argparse
import logging
import math
import os
import json
from typing import Optional
from accelerate import DistributedDataParallelKwargs, DistributedType

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from diffusers import __version__

print(f"you are using diffusers {__version__}")

# import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, whoami

from src.utils import EMAModel

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from src.dataset.ihd_dataset import IhdDatasetSingleRes as Dataset
from src.pipelines.pipeline_stable_diffusion_harmony import (
    StableDiffusionHarmonyPipeline,
)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
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
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
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
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the eval dataloader.",
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
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
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
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
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
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
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
        "--mask_dilate",
        type=int,
        default=0,
        help="mask_dilate",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--nofg",
        action="store_true",
        default=False,
        help="do not use foreground mask when computing loss",
    )
    parser.add_argument(
        "--force_original",
        action="store_true",
        default=False,
        help="always load from original image",
    )

    parser.add_argument(
        "--vae_path", type=str, default=None, help="might use new sd-vae-ft-mse"
    )

    parser.add_argument(
        "--load_pretrained_weights",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--infer_resolution",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--image_log_interval",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def convert_args(args):
    return args


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


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
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    vae_path = args.vae_path
    if vae_path == None:
        vae_path = os.path.join(args.pretrained_model_name_or_path, "vae")
    accelerator.print(f"using vae from <<{vae_path}>>")
    vae = AutoencoderKL.from_pretrained(vae_path, revision=args.revision)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.load_pretrained_weights != None:
        unet_path = args.load_pretrained_weights
    else:
        unet_path = os.path.join(args.pretrained_model_name_or_path, "unet")
    accelerator.print(f"using unet from <<{unet_path}>>")
    unet = UNet2DConditionModel.from_pretrained(
        unet_path,
    )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    model = unet
    print_trainable_parameters(model)

    # NOTE : create ema for all trainable params
    if args.use_ema:
        ema_model = EMAModel(get_trainable_parameters(model), decay=args.ema_decay)
        accelerator.register_for_checkpointing(ema_model)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
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

    train_dataset = Dataset("train", tokenizer, args.resolution, args)
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    if accelerator.is_main_process:
        eval_dataset = Dataset("test", tokenizer, args.resolution, args)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=True,
            batch_size=args.eval_batch_size,
            num_workers=args.dataloader_num_workers,
        )
        eval_batch = next(iter(eval_dataloader))

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

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    # if args.use_ema:
    #     accelerator.register_for_checkpointing(ema_unet)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
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
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )
    start_step = first_epoch * num_update_steps_per_epoch

    if args.use_ema:
        ema_model.to(accelerator.device)

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
                    # comp, mask, real, caption_ids=batch[]
                    real = batch["real"]
                    mask = batch["mask"]
                    comp = batch["comp"]
                    caption_ids = batch["caption_ids"]

                    if args.resolution != args.infer_resolution:
                        # * 默认：加载出来的图片分辨率小，缩放到大分辨率进行训练
                        tgt_size = [args.infer_resolution, args.infer_resolution]
                        real = TF.resize(
                            real,
                            size=tgt_size,
                            interpolation=InterpolationMode.BILINEAR,
                            antialias=True,
                        ).clamp(-1,1)
                        comp = TF.resize(
                            comp,
                            size=tgt_size,
                            interpolation=InterpolationMode.BILINEAR,
                            antialias=True,
                        ).clamp(-1,1)
                        mask = TF.resize(
                            mask,
                            size=tgt_size,
                            interpolation=InterpolationMode.BILINEAR,
                            antialias=True,
                        ).clamp(0,1)
                        mask = (mask > 0.5).float()

                    # Convert images to latent space
                    # =================================================================
                    latents = vae.encode(real.to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    masked_image_latents = vae.encode(
                        comp.to(weight_dtype)
                    ).latent_dist.sample()
                    masked_image_latents = (
                        masked_image_latents * vae.config.scaling_factor
                    )

                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                    mask = torch.nn.functional.interpolate(
                        mask,
                        size=(
                            comp.shape[2] // vae_scale_factor,
                            comp.shape[3] // vae_scale_factor,
                        ),
                    )

                    # =================================================================
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    latent_model_input = torch.cat(
                        [noisy_latents, mask, masked_image_latents], dim=1
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
                            model_pred.float() * mask,
                            target.float() * mask,
                            reduction="none",
                        ).mean(dim=(2, 3), keepdim=True)
                        loss_scale = (
                            (mask.shape[2] * mask.shape[3])
                            / mask.sum(dim=(2, 3), keepdim=True)
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
                logs = {
                    "step_loss": train_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "internal_step": step,
                }
                progress_bar.set_postfix(**logs)
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
                        model_to_save.save_pretrained(model_save_dir)
                if (
                    accelerator.is_main_process
                    and global_step % args.image_log_interval == 0
                ):

                    # unet = accelerator.unwrap_model(model)
                    # if args.use_ema:
                    #     # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    #     ema_model.store(get_trainable_parameters(unet))
                    #     ema_model.copy_to(get_trainable_parameters(unet))
                        
                    eval_unet=UNet2DConditionModel.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="unet",
                        torch_dtype=weight_dtype,
                    )
                    if args.use_ema:
                        ema_model.copy_to(get_trainable_parameters(eval_unet))

                    pipeline = StableDiffusionHarmonyPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=eval_unet,
                        torch_dtype=weight_dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=False)
                    pipeline.enable_xformers_memory_efficient_attention()

                    generator = (
                        torch.Generator(device=accelerator.device).manual_seed(
                            args.seed
                        )
                        if args.seed
                        else None
                    )

                    eval_composite_images = eval_batch["comp"]
                    eval_real_images = eval_batch["real"]
                    eval_mask_images = eval_batch["mask"]

                    if args.resolution != args.infer_resolution:
                        tgt_size = [args.infer_resolution, args.infer_resolution]
                        eval_composite_images = TF.resize(
                            eval_composite_images,
                            size=tgt_size,
                            interpolation=InterpolationMode.BILINEAR,
                            antialias=True,
                        ).clamp(-1,1)
                        eval_real_images = TF.resize(
                            eval_real_images,
                            size=tgt_size,
                            interpolation=InterpolationMode.BILINEAR,
                            antialias=True,
                        ).clamp(-1,1)
                        eval_mask_images = TF.resize(
                            eval_mask_images,
                            size=tgt_size,
                            interpolation=InterpolationMode.BILINEAR,
                            antialias=True,
                        ).clamp(0,1)
                        eval_mask_images = (eval_mask_images > 0.5).float()

                    with torch.inference_mode():
                        samples = pipeline(
                            prompt=eval_batch["caption"],
                            image=eval_composite_images,
                            mask_image=eval_mask_images,
                            height=args.infer_resolution,
                            width=args.infer_resolution,
                            generator=generator,
                            num_inference_steps=5,
                            guidance_scale=1.0,
                            output_type="pt",
                        ).images

                    del pipeline
                    torch.cuda.empty_cache()

                    # if args.use_ema:
                    #     ema_model.restore(get_trainable_parameters(unet))

                    #  保存本轮次的 real , mask , comp
                    image_logging_dir = os.path.join(args.output_dir, "images")
                    if not os.path.exists(image_logging_dir):
                        # 如果文件夹不存在，则创建它
                        os.makedirs(image_logging_dir)
                    bs = len(real)
                    nrow = bs // int(math.sqrt(bs))
                    real_to_save = make_grid(
                        eval_real_images, nrow=nrow, normalize=True, value_range=(-1, 1)
                    )
                    comp_to_save = make_grid(
                        eval_composite_images,
                        nrow=nrow,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    mask_to_save = make_grid(
                        eval_mask_images,
                        nrow=nrow,
                    )
                    harm_to_save = make_grid(samples, nrow=nrow)

                    save_image(
                        real_to_save,
                        os.path.join(image_logging_dir, f"s{global_step:08d}_real.jpg"),
                    )
                    save_image(
                        comp_to_save,
                        os.path.join(image_logging_dir, f"s{global_step:08d}_comp.jpg"),
                    )
                    save_image(
                        mask_to_save,
                        os.path.join(image_logging_dir, f"s{global_step:08d}_mask.jpg"),
                    )
                    save_image(
                        harm_to_save,
                        os.path.join(image_logging_dir, f"s{global_step:08d}_harm.jpg"),
                    )
            if global_step >= args.max_train_steps:
                break
    progress_bar.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
