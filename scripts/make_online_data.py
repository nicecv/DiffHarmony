import os
import torch
import torch.nn.functional as F

from diffusers import UNet2DConditionModel
from diffusers import (
    EulerAncestralDiscreteScheduler,
)
from tqdm import tqdm
from src.dataset.ihd_dataset import IhdDatasetMultiRes as Dataset
from torch.utils.data import DataLoader
from src.diffusers_overwrite import UNet2DCustom
from pipeline_stable_diffusion_harmony import StableDiffusionHarmonyPipeline
import random
from accelerate.utils import set_seed
from torchvision.utils import save_image, make_grid
from utils import select_cand, make_comp

from consistencydecoder import ConsistencyDecoder

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--stage2_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/iHarmony4Online/test",
    )
    # parser.add_argument(
    #     "--device",
    #     type=int,
    #     default=0,
    # )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--rerun",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--comp_method",
        type=str,
        default="v1",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./data/iHarmony4",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="DataSpecs/ihd_all_notext_test.jsonl",
    )
    parser.add_argument("--use_consistency_decoder", action="store_true")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    return args


def denormalize(image_tensor):
    return (image_tensor + 1) / 2


# * 写入JSONL文件
import json


def write_jsonl_file(filename, data):
    with open(filename, "w", encoding="utf-8") as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    from types import SimpleNamespace

    dataset_cfg = SimpleNamespace(
        random_flip=False,
        random_crop=False,
        mask_dilate=0,
        test_file=args.test_file,
        dataset_root=args.dataset_root,
        split="test",
        resolution=512,
        resolutions=[512],
    )

    weight_dtype = torch.float16
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path, torch_dtype=weight_dtype, device_map="auto"
    )
    if args.use_consistency_decoder:
        consistency_decoder = ConsistencyDecoder(device="cuda")  # Model size: 2.49 GB
    else:
        consistency_decoder = None
    harmony_pipeline = StableDiffusionHarmonyPipeline.from_pretrained(
        args.pipeline_path,
        unet=unet,
        consistency_decoder=consistency_decoder,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        torch_dtype=weight_dtype,
        device_map="auto",
    )
    harmony_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        harmony_pipeline.scheduler.config
    )
    harmony_pipeline.set_progress_bar_config(disable=True)
    harmony_pipeline.enable_xformers_memory_efficient_attention()

    if args.stage2_path:
        stage2_model = UNet2DCustom.from_pretrained(
            args.stage2_path,
            torch_dtype=weight_dtype,
            device_map="auto",
        )
        stage2_model.eval()
        stage2_model.requires_grad_(False)
    else:
        stage2_model = None

    ds = Dataset(
        split=dataset_cfg.split,
        tokenizer=None,
        resolutions=dataset_cfg.resolutions,
        opt=dataset_cfg,
    )

    import shutil

    if os.path.exists(args.save_path) and args.rerun:
        shutil.rmtree(args.save_path)
    from pathlib import Path

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "real_images"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "masks"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "refer_composite_images"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "composite_images"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "infer_inputs"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "infer_outputs"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "all"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "DataSpecs"), exist_ok=True)

    json.dump(
        vars(args),
        open(os.path.join(args.save_path, "args.json"), "w"),
        indent=4,
        ensure_ascii=False,
    )

    weight_dtype = harmony_pipeline.unet.dtype
    device = harmony_pipeline.unet.device

    cur_idx = 0
    for run_count in range(1, args.num_runs + 1):
        print(f"Run {run_count}/{args.num_runs}")
        meta_info = []
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        for batch in tqdm(
            dl, total=((len(ds) // args.batch_size) + (len(ds) % args.batch_size != 0))
        ):
            real = batch[dataset_cfg.resolution]["real"].to(device, dtype=weight_dtype)
            mask = batch[dataset_cfg.resolution]["mask"].to(device, dtype=weight_dtype)
            refer_comp = batch[dataset_cfg.resolution]["comp"].to(
                device, dtype=weight_dtype
            )
            cand_indices = select_cand(real, mask, args.comp_method)
            cand = real[cand_indices]
            cand_mask = mask[cand_indices]
            comp_output = make_comp(
                real,
                cand,
                mask,
                args.comp_method,
                harmony_pipeline,
                stage2_model,
                cand_mask,
                return_dict=True,
            )

            comp, infer_input, infer_output = (
                comp_output["comp"],
                comp_output["infer_input"],
                comp_output["infer_output"],
            )

            inner_bs = len(real)
            for i in range(inner_bs):
                image_name = os.path.basename(
                    batch[dataset_cfg.resolution]["real_path"][i]
                )
                mask_name = os.path.basename(
                    batch[dataset_cfg.resolution]["mask_path"][i]
                )
                comp_image_name = mask_name.split(".")[0] + f"_{run_count}.jpg"

                if not os.path.exists(
                    os.path.join(args.save_path, "real_images", image_name)
                ):
                    save_image(
                        denormalize(real[i]),
                        os.path.join(args.save_path, "real_images", image_name),
                    )
                if not os.path.exists(os.path.join(args.save_path, "masks", mask_name)):
                    save_image(
                        mask[i], os.path.join(args.save_path, "masks", mask_name)
                    )
                save_image(
                    denormalize(refer_comp[i]),
                    os.path.join(
                        args.save_path, "refer_composite_images", comp_image_name
                    ),
                )
                save_image(
                    denormalize(refer_comp[i]),
                    os.path.join(
                        args.save_path, "refer_composite_images", comp_image_name
                    ),
                )
                save_image(
                    denormalize(comp[i]),
                    os.path.join(args.save_path, "composite_images", comp_image_name),
                )
                save_image(
                    denormalize(infer_input[i]),
                    os.path.join(args.save_path, "infer_inputs", comp_image_name),
                )
                save_image(
                    denormalize(infer_output[i]),
                    os.path.join(args.save_path, "infer_outputs", comp_image_name),
                )

                all_outputs = torch.stack(
                    [real[i], refer_comp[i], comp[i], infer_input[i], infer_output[i]],
                    dim=0,
                )
                save_image(
                    make_grid(
                        all_outputs,
                        nrow=len(all_outputs),
                        normalize=True,
                        value_range=(-1, 1),
                    ),
                    os.path.join(args.save_path, "all", comp_image_name),
                )

                meta_info.append(
                    {
                        "index": cur_idx,
                        "source_real": batch[dataset_cfg.resolution]["real_path"][i],
                        "source_mask": batch[dataset_cfg.resolution]["mask_path"][i],
                        "source_refer_comp": batch[dataset_cfg.resolution]["comp_path"][
                            i
                        ],
                    }
                )

                cur_idx += 1

            write_jsonl_file(os.path.join(args.save_path, "meta_info.jsonl"), meta_info)

            if args.debug:
                break
    import glob

    write_jsonl_file(
        os.path.join(args.save_path, "DataSpecs", "refer_comps.jsonl"),
        [
            {"file_name": p, "text": ""}
            for p in sorted(
                glob.glob(
                    os.path.join("refer_composite_images", "*.jpg"),
                    root_dir=args.save_path,
                ),
                key=lambda x: x.split("/")[-1].split("_")[0],
            )
        ],
    )
    write_jsonl_file(
        os.path.join(args.save_path, "DataSpecs", "comps.jsonl"),
        [
            {"file_name": p, "text": ""}
            for p in sorted(
                glob.glob(
                    os.path.join("composite_images", "*.jpg"), root_dir=args.save_path
                ),
                key=lambda x: x.split("/")[-1].split("_")[0],
            )
        ],
    )