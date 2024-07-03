import torch
import numpy as np
from PIL import Image
def tensor_to_pil(tensor: torch.Tensor, mode="RGB"):
    if mode == "RGB":
        image_np = (
            tensor.permute(1, 2, 0).add(1).multiply(127.5).numpy().astype(np.uint8)
        )
        image = Image.fromarray(image_np, mode=mode)
    elif mode == "1":
        image_np = tensor.squeeze().multiply(255).numpy().astype(np.uint8)
        image = Image.fromarray(image_np).convert("1")
    else:
        raise ValueError(f"not supported mode {mode}")
    return image

import os
def get_paths(comp_path):
    # .../ds_name/composite_images/xxx_x_x.jpg
    parts = comp_path.split("/")
    img_name_parts = parts[-1].split("_")
    real_path = os.path.join(*parts[:-2], "real_images", f"{img_name_parts[0]}.jpg")
    mask_path = os.path.join(
        *parts[:-2], "masks", f"{img_name_parts[0]}_{img_name_parts[1]}.png"
    )

    return {
        "real": real_path,
        "mask": mask_path,
        "comp": comp_path,
    }

comp_to_paths = get_paths


def harm_to_names(harm_name):
    # xxx_x_x_harmonized.jpg
    img_name_parts = harm_name.split("_")
    real_name = f"{img_name_parts[0]}.jpg"
    mask_name = f"{img_name_parts[0]}_{img_name_parts[1]}.png"
    comp_name = f"{img_name_parts[0]}_{img_name_parts[1]}_{img_name_parts[2]}.jpg"

    return {
        "real": real_name,
        "mask": mask_name,
        "comp": comp_name,
        "harm": harm_name,
    }


def comp_to_harm_path(output_dir, comp_path, suffix="jpg"):
    img_name = comp_path.split("/")[-1]
    prefix, _ = img_name.split(".")
    out_path = os.path.join(output_dir, f"{prefix}_harmonized.{suffix}")
    return out_path


import torch
import torch.nn.functional as F


def get_mask_shift_with_corresponding_center(
    masks: torch.Tensor, cand_masks: torch.Tensor
):
    # masks : (b,c,h,w)
    h, w = masks.shape[-2:]

    # mask boundaries
    true_rows = torch.any(masks, dim=3)
    true_cols = torch.any(masks, dim=2)
    true_cand_rows = torch.any(cand_masks, dim=3)
    true_cand_cols = torch.any(cand_masks, dim=2)
    bs = len(masks)
    dy, dx = [], []

    for i in range(bs):
        try:
            row_min, row_max = torch.nonzero(true_rows[i], as_tuple=True)[-1][
                [0, -1]
            ].tolist()
            col_min, col_max = torch.nonzero(true_cols[i], as_tuple=True)[-1][
                [0, -1]
            ].tolist()
            cand_row_min, cand_row_max = torch.nonzero(
                true_cand_rows[i], as_tuple=True
            )[-1][[0, -1]].tolist()
            cand_col_min, cand_col_max = torch.nonzero(
                true_cand_cols[i], as_tuple=True
            )[-1][[0, -1]].tolist()
            center_y, center_x = (row_min + row_max) / 2, (col_min + col_max) / 2
            cand_center_y, cand_center_x = (cand_row_min + cand_row_max) / 2, (
                cand_col_min + cand_col_max
            ) / 2
            dy_i = (
                torch.tensor([cand_center_y - center_y])
                .float()
                .clamp(-row_min, (h - 1) - row_max)
            )
            dx_i = (
                torch.tensor([cand_center_x - center_x])
                .float()
                .clamp(-col_min, (w - 1) - col_max)
            )
            dy.append(dy_i)
            dx.append(dx_i)
        except:
            dy.append(torch.tensor([0], dtype=torch.float32))
            dx.append(torch.tensor([0], dtype=torch.float32))
    dy = torch.cat(dy, dim=0)[..., None, None]
    dx = torch.cat(dx, dim=0)[..., None, None]

    shift = torch.stack([dy, dx], dim=0).to(masks.device)
    # (2,b,1,1)
    return shift


def get_random_mask_shift(
    masks,
):
    """ """
    # masks : (b,c,h,w)
    h, w = masks.shape[-2:]

    # mask boundaries
    true_rows = torch.any(masks, dim=3)
    true_cols = torch.any(masks, dim=2)
    bs = len(masks)
    dy, dx = [], []
    for i in range(bs):
        try:
            row_min, row_max = torch.nonzero(true_rows[i], as_tuple=True)[-1][
                [0, -1]
            ].tolist()
            col_min, col_max = torch.nonzero(true_cols[i], as_tuple=True)[-1][
                [0, -1]
            ].tolist()
            dy.append(torch.randint(-row_min, h - row_max, (1,)).float())
            dx.append(torch.randint(-col_min, w - col_max, (1,)).float())
        except:
            dy.append(torch.tensor([0], dtype=torch.float32))
            dx.append(torch.tensor([0], dtype=torch.float32))
    dy = torch.cat(dy, dim=0)[..., None, None]
    dx = torch.cat(dx, dim=0)[..., None, None]

    shift = torch.stack([dy, dx], dim=0).to(masks.device)
    # (2,b,1,1)
    return shift


def shift_grid(shape, shift):
    h, w = shape
    dy, dx = shift

    # make grid for space transformation
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    y, x = y.to(shift.device)[None, ...], x.to(shift.device)[None, ...]  # (1,h,w)

    # xy shift
    y = y + 2 * (dy.float() / (h - 1))
    x = x + 2 * (dx.float() / (w - 1))

    grid = torch.stack((x, y), dim=-1)
    return grid


def make_stage2_input(
    harm: torch.Tensor, comp: torch.Tensor, mask: torch.Tensor, in_channels
):
    if harm.dim()==4:
        cat_dim=1
    elif harm.dim()==3:
        cat_dim=0
    else:
        raise ValueError(f"image dims should be 3 or 4 but got {harm.dim()}")
    if in_channels == 3:
        stage2_input = harm
    elif in_channels == 4:
        stage2_input = torch.cat([harm, mask.to(harm)], dim=cat_dim)
    elif in_channels == 7:
        stage2_input = torch.cat([harm, mask.to(harm), comp.to(harm)], dim=cat_dim)
    else:
        raise ValueError(
            f"unsupported stage2 input type : got in channels {in_channels}"
        )
    return stage2_input


import random
import torch.nn.functional as F
from typing import Union

METHOD_VERSION=("v1", "v2-fgfg", "v2-fgbg")

def select_cand(real:torch.Tensor, mask:torch.Tensor, method_version:str):
    assert method_version in METHOD_VERSION , f"method_version {METHOD_VERSION} not implemented"
    if method_version=='v1':    # * random selection
        cand_indices = []
        bs = len(real)
        range_indices = list(range(bs))
        while True:
            shuffled_indices = list(range(bs))
            random.shuffle(shuffled_indices)
            if any([i == j for i, j in zip(range_indices, shuffled_indices)]):
                continue
            break
        cand_indices = torch.tensor(shuffled_indices, dtype=torch.long)
    elif method_version.startswith("v2"):   # * select based on lightness difference
        fg_images = real * mask
        h, w = real.shape[-2:]
        EPS = 1e-6
        fg_lightness = (
            ((fg_images.max(dim=1).values + fg_images.min(dim=1).values) / 2).mean(
                dim=[1, 2]
            )
            * (h * w)
            / (mask.sum(dim=[1, 2, 3]) + EPS)
        )  # (b,c,h,w) -> (b,h,w) -> (b,) ; average lightness of foreground region
        if method_version == "v2-fgfg":
            lightness_diff = fg_lightness.view(-1, 1) - fg_lightness.view(1, -1)  # (b,b)
        elif method_version == "v2-fgbg":
            bg_images = real * (1 - mask)
            bg_lightness = (
                ((bg_images.max(dim=1).values + bg_images.min(dim=1).values) / 2).mean(
                    dim=[1, 2]
                )
                * (h * w)
                / ((1 - mask).sum(dim=[1, 2, 3]) + EPS)
            )  # (b,c,h,w) -> (b,h,w) -> (b,) ; average lightness of background region
            
            lightness_diff = fg_lightness.view(-1, 1) - bg_lightness.view(1, -1)  # (b,b)
        cand_indices = lightness_diff.abs().max(dim=1).indices
    return cand_indices

@torch.inference_mode()
def make_comp(
    real: torch.Tensor,
    cand: torch.Tensor,
    mask: torch.Tensor,
    method_version:str, 
    pipeline,
    stage2_model = None,
    cand_mask: torch.Tensor = None,
    return_dict=False,
):
    assert method_version in METHOD_VERSION , f"method_version {method_version} not implemented"
    if method_version=='v1':
        shift = get_random_mask_shift(mask)
    elif method_version.startswith('v2'):
        shift = get_mask_shift_with_corresponding_center(mask, cand_mask)
        
    grid = shift_grid(mask.shape[-2:], -shift)
    mask_cand = F.grid_sample(
        input=mask.float(), grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
    ).to(
        mask
    )  # (b,c,h,w)
    infer_input = (
        mask_cand
        * F.grid_sample(
            input=real.float(), grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
        ).to(real)
        + (1 - mask_cand) * cand
    )
    
    if return_dict:
        dict_output={
            "infer_input":infer_input,
        }

    bs=len(real)
    h, w = infer_input.shape[-2:]
    infer_output = pipeline(
        prompt=[""] * bs,
        image=infer_input,
        mask_image=mask_cand,
        # generator=generator,
        height=h,
        width=w,
        guidance_scale=1.0,
        num_inference_steps=5,
        output_type="numpy",
    ).images
    # output (b,h,w,c) numpy ndarray , [0,1] range value
    infer_output = (
        torch.tensor(infer_output).to(real).permute(0, 3, 1, 2).sub(0.5).multiply(2)
    )  # convert to [-1,1] (b,c,h,w,) pytorch tensor
    
    if stage2_model is not None:
        stage2_input = make_stage2_input(
            infer_output,
            infer_input,
            mask,
            stage2_model.config.in_channels,
        )
        infer_output = stage2_model(stage2_input).sample.clamp(-1, 1)
        
    if return_dict:
        dict_output['infer_output']=infer_output

    inverse_grid = shift_grid(mask.shape[-2:], shift)
    comp = (1 - mask) * real + mask * F.grid_sample(
        input=infer_output.float(),
        grid=inverse_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ).to(infer_output)
    
    if return_dict:
        dict_output['comp']=comp
        return dict_output

    return comp