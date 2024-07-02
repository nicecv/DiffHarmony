from PIL import Image
import json
import torch
import torchvision.transforms.functional as TF
from os.path import join as pjoin
import os, sys
import numpy as np
import torch.multiprocessing as mp
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
)
from diffusers import UNet2DConditionModel, AutoencoderKL
from src.diffusers_overwrite import UNet2DCustom
from pipeline_stable_diffusion_harmony import StableDiffusionHarmonyPipeline
import cv2
from loguru import logger
from utils import get_paths, comp_to_harm_path, make_stage2_input
from inspect_samples import bmse
from consistencydecoder import ConsistencyDecoder
from copy import deepcopy
from diffusers.utils.import_utils import is_xformers_available

import argparse

model_cls = UNet2DCustom


def parse_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--stage1_path", type=str, default="./logs/notext-150000-1e6-50000/unet"
    )
    parser.add_argument("--stage2_path", type=str, default=None)
    parser.add_argument(
        "--data_file",
        type=str,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="数据集目录",
        default="./data/iHarmony4",
    )
    parser.add_argument(
        "--nocopy",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--refresh", action="store_true", default=False, help="清除 output_dir 已有的内容"
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        default=False,
    )
    parser.add_argument("--use_consistency_decoder", action="store_true")
    parser.add_argument(
        "--use_special_encoder",
        action="store_true",
    )
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--seed", type=int, default=100, help="随机种子")

    # 解析命令行参数
    args = parser.parse_args()
    return args


args = parse_args()

sampler_mapping = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm++2m": DPMSolverMultistepScheduler,
    "ddim": DDIMScheduler,
}


def merge_images(
    harm_image: torch.Tensor, comp_image: torch.Tensor, mask_image: torch.Tensor
):
    assert isinstance(harm_image, torch.Tensor)
    assert isinstance(comp_image, torch.Tensor)
    assert isinstance(mask_image, torch.Tensor)

    bin_mask = (mask_image >= 0.5).float()
    fused_image = harm_image * bin_mask + comp_image * (1 - bin_mask)

    return fused_image


# 定义一个函数，用于将一个列表分割成多个子列表
def split_array(arr, n):
    avg = len(arr) / n
    result = []
    last = 0.0

    while last < len(arr):
        if len(result) == n - 1:
            result.append(arr[int(last) :])
        else:
            idx = int(round(last + avg))
            result.append(arr[int(last) : idx])
            last = idx

    return result


def assign_sampler(output_dir, mapping_dict):
    for key, value in mapping_dict.items():
        if key in output_dir:
            return value
    logger.warning("no specification of sampler; use <<ddim>> by default")
    return DDIMScheduler


def replace_extension(filename):
    # 使用os.path.splitext获取文件名和扩展名
    name, ext = os.path.splitext(filename)

    # 如果扩展名是.jpg，则替换为.png
    if ext.lower() == ".jpg":
        new_filename = name + ".png"
        return new_filename
    else:
        # 如果扩展名不是.jpg，则返回原始文件名
        return filename


from utils import tensor_to_pil


def dilate_mask_image(mask: torch.Tensor, mask_dilate: int) -> torch.Tensor:
    if mask_dilate > 0:
        mask_np = (mask * 255).numpy().astype(np.uint8)
        mask_np = cv2.dilate(mask_np, np.ones((mask_dilate, mask_dilate), np.uint8))
        mask = torch.tensor(mask_np.astype(np.float32) / 255.0)
    return mask


from typing import Union


def format_time_interval(num: Union[float, int]):
    # 确保输入是正数
    num = abs(num)

    # 计算小时、分钟和秒
    hours = int(num // 3600)
    minutes = int((num % 3600) // 60)
    seconds = int(num % 60)

    # 格式化为 hh:mm:ss 字符串
    time_interval = "{:02d}h {:02d}m {:02d}s".format(hours, minutes, seconds)

    return time_interval


# 定义一个函数，用于处理单个图像的合成和保存
@torch.no_grad()
def inference(rank, data):
    print(f"rank {rank} : {len(data)} samples")
    # 获取当前进程使用的GPU设备
    device = torch.device(f"cuda:{rank}")
    weight_dtype = torch.float16

    # 加载 stage 1 model
    unet = UNet2DConditionModel.from_pretrained(
        args.stage1_path,
        torch_dtype=weight_dtype,
    )
    if args.use_consistency_decoder:
        consistency_decoder = ConsistencyDecoder(device=device)  # Model size: 2.49 GB
    else:
        consistency_decoder = None
    if args.use_special_encoder:
        special_encoder_path = os.path.join(
            os.path.dirname(args.stage1_path), "special_encoder", "model.pt"
        )
        special_encoder = torch.load(special_encoder_path, map_location=device)
        special_encoder.requires_grad_(False)
        special_encoder.eval()
        special_encoder.to(dtype=weight_dtype)
    else:
        special_encoder = None
    # 创建Stable Diffusion Harmony Pipeline
    pipe = StableDiffusionHarmonyPipeline.from_pretrained(
        "checkpoints/stable-diffusion-inpainting",
        unet=unet,
        consistency_decoder=consistency_decoder,
        special_encoder=special_encoder,
        torch_dtype=weight_dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    sampler_cls = assign_sampler(args.output_dir, sampler_mapping)
    pipe.scheduler = sampler_cls.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()

    output_dir_name = args.output_dir.split("/")[-1]
    guidance_scale = 1.0
    inference_steps = 5
    hw = 512
    dilate = 0
    output_res = 256
    use_stage2 = False
    for s in output_dir_name.split("-"):
        if "cfg" in s:
            guidance_scale = float(s.strip("cfg"))
            print("guidance_scale:", guidance_scale)
        elif "outpx" in s:
            output_res = int(s.strip("outpx"))
            print("output resolution:", output_res)
        elif "is" in s:
            inference_steps = int(s.strip("is"))
            print("inference_steps:", inference_steps)
        elif "px" in s:
            hw = int(s.strip("px"))
            print("inference resolution:", hw)
        elif "dilate" in s:
            dilate = int(s.strip("dilate"))
            print("mask dilate:", dilate)
        elif "restore" in s:
            use_stage2 = True
            print("use stage 2 model to enhance harmonized image")

    if use_stage2:
        # 加载 stage 2 model
        stage2_model = model_cls.from_pretrained(
            args.stage2_path,
            torch_dtype=weight_dtype,
        )
        stage2_model.to(device)
        stage2_model.eval()
        stage2_model.requires_grad_(False)
        in_channels = stage2_model.config.in_channels
        if is_xformers_available():
            stage2_model.enable_xformers_memory_efficient_attention()

    import time

    st = time.time()
    for idx, (img_paths, text) in enumerate(data, start=1):
        comp_path = img_paths["comp"]
        if comp_path is None:
            break

        out_path = comp_to_harm_path(args.output_dir, comp_path, suffix="png")
        # print(out_path)
        if os.path.exists(out_path):
            continue

        if not args.nocopy:
            real_path = img_paths["real"]
            real = Image.open(real_path).convert("RGB")
            if real.size != (output_res, output_res):
                real = TF.resize(real, [output_res, output_res])
            real.save(os.path.join(args.output_dir, os.path.basename(real_path)))

        ## * prepare comp
        comp = Image.open(comp_path).convert("RGB")
        stage1_comp = deepcopy(comp)
        if stage1_comp.size != (hw, hw):
            stage1_comp = TF.resize(stage1_comp, [hw, hw])
        stage1_comp = TF.normalize(TF.to_tensor(stage1_comp), [0.5], [0.5])

        stage2_comp = deepcopy(comp)
        if stage2_comp.size != (output_res, output_res):
            stage2_comp = TF.resize(stage2_comp, [output_res, output_res])
        if not args.nocopy:
            stage2_comp.save(os.path.join(args.output_dir, os.path.basename(comp_path)))
        stage2_comp = TF.normalize(TF.to_tensor(stage2_comp), [0.5], [0.5])

        ## * prepare mask
        mask_path = img_paths["mask"]
        mask = Image.open(mask_path).convert("1")
        stage1_mask = deepcopy(mask)
        if stage1_mask.size != (hw, hw):
            stage1_mask = TF.resize(stage1_mask, [hw, hw])
        stage1_mask = dilate_mask_image(TF.to_tensor(stage1_mask), dilate)

        stage2_mask = deepcopy(mask)
        if stage2_mask.size != (output_res, output_res):
            stage2_mask = TF.resize(stage2_mask, [output_res, output_res])
        if not args.nocopy:
            stage2_mask.save(os.path.join(args.output_dir, os.path.basename(mask_path)))
        stage2_mask = dilate_mask_image(TF.to_tensor(stage2_mask), dilate)

        generator = torch.Generator(device=device).manual_seed(args.seed)
        # stage 1 : 和谐化
        harm_image = pipe(
            prompt=text,
            image=stage1_comp,
            mask_image=stage1_mask,
            generator=generator,
            height=hw,
            width=hw,
            guidance_scale=guidance_scale,
            num_inference_steps=inference_steps,
            output_type="numpy",
        ).images[0]
        # NOTE : 输出是 (h,w,c) 的 numpy ndarray , [0,1] 区间 value
        harm_image = (
            torch.tensor(harm_image).permute(2, 0, 1).sub(0.5).multiply(2)
        )  # [-1,1] (c,h,w,) pytorch tensor
        harm_image = harm_image.float().cpu()
        if tuple(harm_image.shape[-2:]) != (output_res, output_res):
            harm_image = TF.resize(
                harm_image, size=[output_res, output_res], antialias=True,
            ).clamp(-1,1) # [-1,1] (c,h,w,) pytorch tensor

        if use_stage2:
            # stage 2 : restoration
            stage2_input = make_stage2_input(
                harm_image,
                stage2_comp,
                stage2_mask,
                in_channels,
            )
            stage2_input = stage2_input.to(
                device=device, dtype=stage2_model.dtype
            ).unsqueeze(0)
            restored_image = stage2_model(stage2_input).sample.squeeze()
            output = tensor_to_pil(restored_image.float().clamp(-1, 1).cpu())
        elif "nobg" not in output_dir_name:
            harm_image = merge_images(harm_image, stage2_comp, stage2_mask)
            output = tensor_to_pil(harm_image.cpu())
        
        output.save(
            out_path,
            compression=None,
        )
        print(
            f"{device} || {comp_path} || {format_time_interval(time.time()-st)} || [{idx}/{len(data)}]"
        )


if __name__ == "__main__":
    # 如果输出目录不存在，则创建输出目录
    if not os.path.exists(args.output_dir):
        from pathlib import Path

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 设置多进程启动方式为'spawn'
    mp.set_start_method("spawn")

    # 获取可用的GPU设备数量
    devices_ct = torch.cuda.device_count()

    # 从json文件中读取文件名和文本信息，并加入数据列表
    qs = []
    # data_file = "ihd_all_notext_test_sample.jsonl"
    data_file = args.data_file
    if args.refresh:
        os.system(f"rm {args.output_dir}/*")

    for i, line in enumerate(open(pjoin(args.dataset_dir, data_file), "r").readlines()):
        data_item = json.loads(line.strip("\n"))
        img_paths = get_paths(os.path.join(args.dataset_dir, data_item["file_name"]))

        comp_path = img_paths["comp"]
        out_path = comp_to_harm_path(args.output_dir, comp_path, suffix="png")
        if os.path.exists(out_path):
            continue

        qs.append((img_paths, data_item["text"]))

    import random

    random.shuffle(qs)

    if not args.skip_inference:
        # 根据GPU数量将数据列表划分成多个子列表
        qs = split_array(qs, devices_ct)

        process_list = []
        for i in range(devices_ct):
            # 创建多个进程，每个进程执行inference函数处理一部分数据
            p = mp.Process(target=inference, args=(i, qs[i]))
            p.start()
            process_list.append(p)

        # 等待所有进程完成处理
        for p in process_list:
            p.join()
