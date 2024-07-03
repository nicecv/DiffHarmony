import os.path
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from argparse import Namespace

from PIL import Image
import json
import cv2
import os
from typing import List, Dict


def get_paths(path) -> Dict[str,str]:
    parts = path.split("/")
    img_name_parts = parts[-1].split(".")[0].split("_")
    if "masks" in path:
        return {
            "gt_path": os.path.join(
                *parts[:-2], "real_images", f"{img_name_parts[0]}.jpg"
            ),
            "mask_path": path,
            "image_path": os.path.join(
                *parts[:-2], "real_images", f"{img_name_parts[0]}.jpg"
            ),
        }
    elif "composite" in path:
        return {
            "gt_path": os.path.join(
                *parts[:-2], "real_images", f"{img_name_parts[0]}.jpg"
            ),
            "mask_path": os.path.join(
                *parts[:-2], "masks", f"{img_name_parts[0]}_{img_name_parts[1]}.png"
            ),
            "image_path": path,
        }
    else:
        raise ValueError(f"Unknown path type: {path}")


class IhdDatasetMultiRes(Dataset):
    def __init__(self, split, tokenizer, resolutions: List[int], opt):

        self.image_paths = []
        self.captions = []
        self.split = split
        self.tokenizer = tokenizer
        self.resolutions = list(set(resolutions))
        self.random_flip = opt.random_flip
        self.random_crop = opt.random_crop
        self.mask_dilate = opt.mask_dilate

        data_file = opt.train_file if split == "train" else opt.test_file
        if split == "test":
            self.random_flip = False
            self.random_crop = False

        with open(os.path.join(opt.dataset_root, data_file), "r") as f:
            for line in f:
                cont = json.loads(line.strip())
                image_path = os.path.join(
                    opt.dataset_root,
                    cont["file_name"],
                )
                self.image_paths.append(image_path)
                self.captions.append(cont.get("text", ""))

        self.create_image_transforms()

    def __len__(self):
        return len(self.image_paths)

    def create_image_transforms(self):
        self.rgb_normalizer = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __getitem__(self, index):
        paths = get_paths(self.image_paths[index])

        comp = Image.open(paths["image_path"]).convert("RGB")  # RGB , [0,255]
        mask = Image.open(paths["mask_path"]).convert("1")
        real = Image.open(paths["gt_path"]).convert("RGB")  # RGB , [0,255]

        caption = self.captions[index]
        if self.tokenizer is not None:
            caption_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
        else:
            caption_ids = torch.empty(size=(1,), dtype=torch.long)

        if self.random_flip and np.random.rand() > 0.5 and self.split == "train":
            comp, mask, real = TF.hflip(comp), TF.hflip(mask), TF.hflip(real)
        if self.random_crop:
            for _ in range(5):
                mask_tensor = TF.to_tensor(mask)
                crop_box = T.RandomResizedCrop.get_params(
                    mask_tensor, scale=[0.5, 1.0], ratio=[3 / 4, 4 / 3]
                )
                cropped_mask_tensor = TF.crop(mask_tensor, *crop_box)
                h, w = cropped_mask_tensor.shape[-2:]
                if cropped_mask_tensor.sum() < 0.01 * h * w:
                    continue
                break

        example = {}
        for resolution in self.resolutions:
            if self.random_crop:
                this_res_comp = TF.resize(
                    TF.crop(comp, *crop_box),
                    size=(resolution, resolution),
                )  # default : bilinear resample ; antialias for PIL Image
                this_res_real = TF.resize(
                    TF.crop(real, *crop_box),
                    size=(resolution, resolution),
                )  # default : bilinear resample ; antialias for PIL Image
                this_res_mask = TF.resize(
                    TF.crop(mask, *crop_box),
                    size=(resolution, resolution),
                )  # default : bilinear resample ; antialias for PIL Image
            else:
                if comp.size == (resolution, resolution):
                    this_res_comp = comp
                    this_res_mask = mask
                    this_res_real = real
                else:
                    this_res_comp = TF.resize(
                        comp, [resolution, resolution]
                    )  # default : bilinear resample ; antialias for PIL Image
                    this_res_mask = TF.resize(
                        mask, [resolution, resolution]
                    )  # default : bilinear resample ; antialias for PIL Image
                    this_res_real = TF.resize(
                        real, [resolution, resolution]
                    )  # default : bilinear resample ; antialias for PIL Image

            this_res_comp = self.rgb_normalizer(this_res_comp)  # tensor , [-1,1]
            this_res_real = self.rgb_normalizer(this_res_real)  # tensor , [-1,1]
            this_res_mask = TF.to_tensor(this_res_mask)
            this_res_mask = (this_res_mask >= 0.5).float()  # mask : tensor , 0/1
            this_res_mask = self.dilate_mask_image(this_res_mask)
            example[resolution] = {
                "real": this_res_real,
                "mask": this_res_mask,
                "comp": this_res_comp,
                "real_path": paths["gt_path"],
                "mask_path": paths["mask_path"],
                "comp_path": paths["image_path"],
                "caption": caption,
                "caption_ids": caption_ids,
            }

        return example

    def dilate_mask_image(self, mask: torch.Tensor) -> torch.Tensor:
        if self.mask_dilate > 0:
            mask_np = (mask * 255).numpy().astype(np.uint8)
            mask_np = cv2.dilate(
                mask_np, np.ones((self.mask_dilate, self.mask_dilate), np.uint8)
            )
            mask = torch.tensor(mask_np.astype(np.float32) / 255.0)
        return mask


class IhdDatasetSingleRes(Dataset):
    def __init__(self, split, tokenizer, resolution, opt):
        self.resolution = resolution
        self.multires_ds = IhdDatasetMultiRes(split, tokenizer, [resolution], opt)

    def __len__(self):
        return len(self.multires_ds)

    def __getitem__(self, index):
        return self.multires_ds[index][self.resolution]



import json
import numpy as np

subset_names = [
    "HAdobe5k",
    "HCOCO",
    "Hday2night",
    "HFlickr",
]

def extract_ds_name(path):
    for subset_name in subset_names:
        if subset_name in path:
            return subset_name
    return None

def read_jsonl_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


SUBSET_TO_MIX = [
    "HCOCO",
    "HFlickr",
    "HAdobe5k",
    "Hday2night",
]

class IhdWithRandomMaskComp(Dataset):
    def __init__(
        self,
        tokenizer,
        opt,
    ) -> None:
        super().__init__()
        self.resolution = opt.resolution
        self.random_flip = opt.random_flip
        self.random_crop = opt.random_crop
        self.center_crop = opt.center_crop
        self.dataset_root = opt.dataset_root
        self.mix_area_thres = opt.mix_area_thres
        
        if isinstance(self.dataset_root, list):
            self.dataset_root = self.dataset_root[0]
        self.tokenizer = tokenizer
        self.refer_method = opt.refer_method
        self.real_mask_mapping = json.load(open(opt.image_mask_mapping, "r"))
        self.mask_comp_mapping = json.load(open(opt.mask_comp_mapping, "r"))
        if opt.train_file is None:
            self.image_rel_paths = list(self.real_mask_mapping.keys())
        else:
            train_file_content = [line['file_name'] for line in read_jsonl_file(os.path.join(self.dataset_root, opt.train_file))]
            self.image_rel_paths = list(set(self._convert_rel_comp_to_real(train_file_content)))
        self.image_rel_paths.sort()
        self.image_processor = self.create_image_transforms()
        self.image_normalizer = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        
    def _convert_rel_comp_to_real(self, rel_comp_paths:List[str]):
        def _comp_to_real(comp_path):
            parts = comp_path.split("/")
            img_name_parts = parts[-1].split("_")
            real_path = os.path.join(*parts[:-2], "real_images", f"{img_name_parts[0]}.jpg")
            return real_path
        rel_real_paths = []
        for rel_comp_path in rel_comp_paths:
            # rel_real_path = _comp_to_real(insert_ds_suffix(rel_comp_path, self.imgdir_suffix))
            rel_real_path = _comp_to_real(rel_comp_path)
            rel_real_paths.append(rel_real_path)
        return rel_real_paths

    def create_image_transforms(self):
        transforms = []
        if self.random_flip:
            transforms.append(T.RandomHorizontalFlip())
        if self.random_crop:
            transforms.append(
                T.RandomResizedCrop(
                    size=[self.resolution, self.resolution],
                    scale=(0.5, 1),
                    antialias=True,
                ),
            )
        elif self.center_crop:
            transforms.extend(
                [
                    T.Resize(size=self.resolution, antialias=True),
                    T.CenterCrop(size=[self.resolution, self.resolution]),
                ]
            )
        else:
            transforms.append(
                T.Resize(size=[self.resolution, self.resolution], antialias=True)
            )

        transforms = T.Compose(transforms)
        return transforms

    def __getitem__(self, i):
        image_rel_path = self.image_rel_paths[i]
        image_path = os.path.join(self.dataset_root, image_rel_path)

        image = Image.open(image_path).convert("RGB")

        mask_rel_path = np.random.choice(self.real_mask_mapping[image_rel_path])
        comp_rel_path = np.random.choice(self.mask_comp_mapping[mask_rel_path])

        mask_path = os.path.join(self.dataset_root, mask_rel_path)
        comp_path = os.path.join(self.dataset_root, comp_rel_path)
        mask = Image.open(mask_path).convert("1")
        comp = Image.open(comp_path).convert("RGB")

        image = self.image_normalizer(image)
        comp = self.image_normalizer(comp)
        mask = TF.to_tensor(mask).to(dtype=torch.float32)

        for _ in range(5):
            merged = torch.cat([image, mask, comp], dim=0)
            if tuple(merged.shape[-2:]) == (self.resolution, self.resolution):
                break
            else:
                merged_processed = self.image_processor(merged)
            image, mask, comp = torch.split(merged_processed, [3, 1, 3], dim=0)
            h, w = mask.shape[-2:]
            if self.random_crop and mask.sum() < (0.01 * h * w):
                continue
            break

        mask = (mask >= 0.5).float()
        image = image.clamp(-1, 1)
        comp = comp.clamp(-1, 1)

        # caption = self.captions[index]
        caption = ""
        if self.tokenizer is not None:
            caption_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
        else:
            caption_ids = torch.empty(size=(1,))

        example = {
            "image": image,
            "image_path": image_path,
            "caption_ids": caption_ids,
            "subset": extract_ds_name(image_path),
        }

        example["mask"] = mask
        example["mask_path"] = mask_path

        example["comp"] = comp
        example["comp_path"] = comp_path
        
        select_mask = torch.tensor(1)
        if extract_ds_name(image_path) not in SUBSET_TO_MIX:
            select_mask = 0
        h,w=mask.shape[-2:]
        if mask.sum() < self.mix_area_thres:
            select_mask=0
        example["select_mask"] = select_mask
        
        if self.refer_method=='batch':
            pass

        return example

    def __len__(self) -> int:
        return len(self.image_rel_paths)


class IhdDatasetWithSDXLMetadata(Dataset):
    def __init__(self, split, resolution: int, opt):
        self.image_paths = []
        self.captions = []
        self.split = split
        self.resolution = resolution
        self.random_flip = opt.random_flip
        self.random_crop = opt.random_crop
        if hasattr(opt, "crop_resolution") and opt.crop_resolution is not None:
            self.crop_resolution = opt.crop_resolution
        else:
            self.crop_resolution = resolution

        data_file = opt.train_file if split == "train" else opt.test_file

        with open(os.path.join(opt.dataset_root, data_file), "r") as f:
            for line in f:
                cont = json.loads(line.strip())
                image_path = os.path.join(
                    opt.dataset_root,
                    cont["file_name"],
                )
                self.image_paths.append(image_path)
                self.captions.append(cont.get("text", ""))

        self.transforms = Namespace(
            resize=T.Resize(
                self.resolution,
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            crop=(
                T.CenterCrop(self.crop_resolution)
                if not self.random_crop
                else T.RandomCrop(self.crop_resolution)
            ),
            flip = T.RandomHorizontalFlip(p=0.5),
            normalize = T.Normalize([0.5], [0.5])
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        paths = get_paths(self.image_paths[index])

        comp = Image.open(paths["image_path"]).convert("RGB")  # RGB , [0,255]
        mask = Image.open(paths["mask_path"]).convert("1")
        real = Image.open(paths["gt_path"]).convert("RGB")  # RGB , [0,255]

        original_size = torch.tensor([comp.height, comp.width])
        comp = TF.to_tensor(comp)
        real = TF.to_tensor(real)
        mask = TF.to_tensor(mask)
        all_img = torch.cat([comp, real, mask], dim=0)
        all_img = self.transforms.resize(all_img)
        
        if self.random_flip:
            # flip
            all_img = self.transforms.flip(all_img)
                
        if not self.random_crop:
                y1 = max(0, int(round((all_img.shape[0] - self.crop_resolution) / 2.0)))
                x1 = max(0, int(round((all_img.shape[1] - self.crop_resolution) / 2.0)))
                all_img = self.transforms.crop(all_img)
        else:
            y1, x1, h, w = self.transforms.crop.get_params(
                all_img, (self.crop_resolution, self.crop_resolution)
            )
            all_img = TF.crop(all_img, y1, x1, h, w)
        
        crop_top_left = torch.tensor([y1, x1])
        comp, real, mask = torch.split(all_img, [3, 3, 1], dim=0)
        comp = self.transforms.normalize(comp)  # tensor , [-1,1]
        mask = torch.ge(mask, 0.5).float()  # >= 0.5 is True
        # mask : tensor , 0/1
        real = self.transforms.normalize(real)  # tensor , [-1,1]
        
        return {
            "real": real,
            "mask": mask,
            "comp": comp,
            "real_path": paths["gt_path"],
            "mask_path": paths["mask_path"],
            "comp_path": paths["image_path"],
            "caption": self.captions[index],
            "original_size": original_size,
            "crop_top_left": crop_top_left,
        }
