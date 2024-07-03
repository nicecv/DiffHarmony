import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import DistributedType
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from src.dataset.ihd_dataset import IhdDatasetWithSDXLMetadata as Dataset
from src.models.condition_vae import ConditionVAE

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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
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
        "--output_dir",
        type=str,
        default="logs/test",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crop_resolution",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
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
        help="The number of images to generate for evaluation.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
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
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--image_logging_epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--image_logging_steps",
        type=int,
        default=100,
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
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
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
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
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
        "--logging_dir",
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
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--additional_in_channels",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--freeze_decoder",
        action="store_true",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=["normal", "inverse"],
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    if "condition_vae" in args.pretrained_vae_model_name_or_path:
        condition_vae = ConditionVAE.from_pretrained(
            args.pretrained_vae_model_name_or_path, additional_in_channels=args.additional_in_channels
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_vae_model_name_or_path, 
        )
        condition_vae = ConditionVAE.from_vae(vae, load_weights=True, additional_in_channels=args.additional_in_channels)
    condition_vae.config.force_upcast=False
    condition_vae.train()
    condition_vae.requires_grad_(True, freeze_decoder=args.freeze_decoder)

    model = condition_vae
    print_trainable_parameters(model)
    
    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            model_cls=ConditionVAE,
            model_config=model.config,
            decay=args.ema_decay,
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "condition_vae_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "condition_vae"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "condition_vae_ema"), ConditionVAE
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ConditionVAE.from_pretrained(
                    input_dir, subfolder="condition_vae"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

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

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        get_trainable_parameters(model),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = Dataset(split="train", resolution=args.resolution, opt=args)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    if accelerator.is_main_process:
        eval_dataset = Dataset(split="test", resolution=args.resolution, opt=args)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
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
            args.lr_warmup_ratio * (args.max_train_steps)
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

    if args.use_ema:
        ema_model.to(accelerator.device)

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
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
        args.checkpointing_steps = (
            args.checkpointing_epochs * num_update_steps_per_epoch
        )
    if args.image_logging_epochs is not None:
        args.image_logging_steps = args.image_logging_epochs * num_update_steps_per_epoch

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config=vars(args))
        import json
        json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=4, ensure_ascii=False)

    # Function for unwrapping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
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
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        total_l1_loss = 0.0
        total_mse_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                
                real = batch["real"]
                comp = batch["comp"]
                mask = batch["mask"]

                if args.mode=='normal':
                    model_input = real
                    input_cond = torch.cat([comp, mask], dim=1)
                else:
                    model_input = comp
                    input_cond = torch.cat([real, mask], dim=1)
                    
                model_input = model_input.contiguous()
                input_cond = input_cond.contiguous()
                
                model_output = model(model_input, sample_posterior=True, cond=input_cond).sample
                target = model_input
                
                l1_loss = F.l1_loss(model_output.float(), target.float(), reduction="mean") 
                mse_loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
                loss=l1_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                avg_l1_loss = accelerator.gather(l1_loss.repeat(args.train_batch_size)).mean()
                total_l1_loss += avg_l1_loss.item() / args.gradient_accumulation_steps
                avg_mse_loss = accelerator.gather(mse_loss.repeat(args.train_batch_size)).mean()
                total_mse_loss += avg_mse_loss.item() / args.gradient_accumulation_steps

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
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                logs = {
                    "step_loss": train_loss,
                    "l1_loss": total_l1_loss,
                    "mse_loss": total_mse_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "internal_step": step,
                }
                if args.use_ema:
                    logs["ema_decay"] = ema_model.cur_decay_value
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)
                global_step += 1
                train_loss = 0.0
                total_l1_loss = 0.0
                total_mse_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if (
                        accelerator.is_main_process
                        and args.checkpoints_total_limit is not None
                    ):
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
                        accelerator.distributed_type == DistributedType.DEEPSPEED
                        or accelerator.is_main_process
                    ):
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                    if accelerator.is_main_process:
                        unwrapped_model = unwrap_model(model)
                        if args.use_ema:
                            ema_model.copy_to(unwrapped_model.parameters())

                        model_to_save = unwrapped_model
                        model_save_dir = os.path.join(
                            args.output_dir, f"weights-{global_step}"
                        )
                        os.makedirs(model_save_dir, exist_ok=True)
                        model_to_save.save_pretrained(model_save_dir)

                # Generate sample images for visual inspection
                if (global_step % args.image_logging_steps == 0) and (
                    accelerator.is_main_process
                ):
                    condition_vae = unwrap_model(model)
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_model.store(condition_vae.parameters())
                        ema_model.copy_to(condition_vae.parameters())

                    generator = (
                        torch.Generator(device=accelerator.device).manual_seed(
                            args.seed
                        )
                        if args.seed
                        else None
                    )

                    eval_model_input = eval_batch["real"]
                    eval_input_cond = torch.cat([eval_batch["comp"], eval_batch["mask"]], dim=1)
                    eval_model_input=eval_model_input.to(accelerator.device, dtype=weight_dtype)
                    eval_input_cond=eval_input_cond.to(accelerator.device, dtype=weight_dtype)

                    with torch.inference_mode():
                        eval_conditioned_rec = condition_vae(eval_model_input, sample_posterior=True, cond=eval_input_cond, generator=generator).sample
                        eval_rec = condition_vae(eval_model_input, sample_posterior=True, generator=generator).sample

                    if args.use_ema:
                        ema_model.restore(condition_vae.parameters())

                    del condition_vae
                    torch.cuda.empty_cache()

                    image_logging_dir = os.path.join(args.output_dir, "images")
                    if not os.path.exists(image_logging_dir):
                        os.makedirs(image_logging_dir)

                    bs = len(eval_model_input)
                    nrow = bs // int(math.sqrt(bs))

                    input_to_save = make_grid(
                        eval_model_input, nrow=nrow, normalize=True, value_range=(-1, 1)
                    )
                    rec_to_save = make_grid(
                        eval_rec, nrow=nrow, normalize=True, value_range=(-1, 1)
                    )
                    conditioned_rec_to_save = make_grid(
                        eval_conditioned_rec, nrow=nrow, normalize=True, value_range=(-1, 1)
                    )

                    save_image(
                        input_to_save,
                        os.path.join(image_logging_dir, f"s{global_step:08d}_input.jpg"),
                    )
                    save_image(
                        rec_to_save,
                        os.path.join(image_logging_dir, f"s{global_step:08d}_rec.jpg"),
                    )
                    save_image(
                        conditioned_rec_to_save,
                        os.path.join(image_logging_dir, f"s{global_step:08d}_conditioned_rec.jpg"),
                    )
            if global_step >= args.max_train_steps:
                break
    progress_bar.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
