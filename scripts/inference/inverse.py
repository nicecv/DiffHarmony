import argparse
import os
from pathlib import Path

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.transforms.functional import to_pil_image, resize, to_tensor 
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler,
)
from src.pipelines.pipeline_stable_diffusion_harmony import (
    StableDiffusionHarmonyPipeline,
)
from src.dataset.ihd_dataset import IhdDatasetMultiRes as Dataset
from src.models.condition_vae import ConditionVAE

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a inference script."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--pretrained_unet_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--stage2_model_name_or_path",
        type=str,
        default=None,
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
        required=True,
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--output_resolution",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--mask_dilate",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="The number of images to generate for evaluation.",
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
        "--rounds",
        type=int,
        default=1,
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def list_in_str(input_list, target_str):
    for item in input_list:
        if item in target_str:
            return True
    return False

def replace_background(fake_image:torch.Tensor, real_image:torch.Tensor, mask:torch.Tensor):
    real_image = real_image.to(fake_image)
    mask = mask.to(fake_image)
    
    fake_image = fake_image * mask + real_image * (1 - mask)
    fake_image = to_pil_image(fake_image)
    return fake_image


def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if list_in_str(["condition_vae", "cvae"], args.pretrained_vae_model_name_or_path):
        vae_cls = ConditionVAE
    else:
        vae_cls = AutoencoderKL
        
    vae = vae_cls.from_pretrained(
        args.pretrained_vae_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_unet_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    pipeline = StableDiffusionHarmonyPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline.to(accelerator.device)
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.set_progress_bar_config(disable=True)

    dataset = Dataset(
        split="test",
        tokenizer=None,
        resolutions=[args.resolution, args.output_resolution],
        opt=args,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    dataloader = accelerator.prepare(dataloader)
    
    for num_round in range(1, args.rounds+1):
        progress_bar = tqdm(
            range(0, len(dataloader)),
            initial=0,
            desc="Batches",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
        for step, batch in enumerate(dataloader):
            eval_mask_images = batch[args.resolution]["mask"]
            eval_gt_images = batch[args.resolution]["real"]
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            with torch.inference_mode():
                samples = pipeline(
                    prompt=batch[args.resolution]["caption"],
                    image=eval_gt_images,
                    mask_image=eval_mask_images,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=5,
                    guidance_scale=1.0,
                    generator=generator,
                    output_type="pt",
                ).images  # [0,1] torch tensor

            samples = samples.clamp(0, 1)

            for i, sample in enumerate(samples):
                sample = to_pil_image(sample)
                output_shape = (args.output_resolution, args.output_resolution)
                if sample.size != output_shape:
                    sample = resize(sample, output_shape, antialias=True)
                    
                sample = replace_background(to_tensor(sample), batch[args.output_resolution]["real"][i].add(1).div(2), batch[args.output_resolution]["mask"][i])
                
                save_name = (
                    batch[args.output_resolution]["mask_path"][i]
                    .split("/")[-1]
                    .split(".")[0]
                    + f"_{num_round}.jpg"
                )
                sample.save(
                    os.path.join(args.output_dir, save_name), quality=100
                )
            progress_bar.update(1)
        progress_bar.close()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    args = parse_args()
    main(args)
