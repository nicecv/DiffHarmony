import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from PIL import Image
import glob
from collections import defaultdict
from accelerate import Accelerator
from einops import rearrange
from accelerate.utils import set_seed


class CustomDataset(Dataset):
    def __init__(self, dataset_root):
        # 加载 cand_composite_images 下所有的 jpg 图片路径；使用 glob
        cand_composite_paths = sorted(glob.glob(
            os.path.join(dataset_root, "cand_composite_images", "*.jpg")
        ))
        data = defaultdict(list)
        for cand_composite_path in tqdm(cand_composite_paths):
            image_name = os.path.basename(cand_composite_path).split("_")[0]
            data[image_name].append(cand_composite_path)
        self.data = [
            {
                "real_path": os.path.join(dataset_root, "real_images", k + ".jpg"),
                "cand_composite_paths": v,
            }
            for k, v in data.items()
        ]

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        cand_composite_images = [
            self.transform(Image.open(cand_composite_path).convert("RGB"))
            for cand_composite_path in item["cand_composite_paths"]
        ]
        cand_composite_images = torch.stack(
            cand_composite_images, dim=0
        )  # (t, c, h, w)
        return {
            "cand_composite_images": cand_composite_images,
            "cand_composite_paths": item["cand_composite_paths"],
        }
        
def collate_fn(batch):
    cand_composite_images = [item["cand_composite_images"] for item in batch]
    cand_composite_paths = [item["cand_composite_paths"] for item in batch]
    return {
        "cand_composite_images": torch.stack(cand_composite_images, dim=0), # (b, t, c, h, w)
        "cand_composite_paths": cand_composite_paths,
    }


if __name__ == "__main__":
    seed = 0
    dataset_root = "data/iHarmonyGen/imaterial"
    model_path = "checkpoints/harmony_classifier/model_state_dict.pth"
    batch_size = 16
    
    set_seed(seed)
    accelerator = Accelerator()
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(accelerator.device)

    dataset = CustomDataset(dataset_root)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    dataloader = accelerator.prepare(dataloader)

    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=0,
        desc="Batches",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # import pdb ; pdb.set_trace()
    for step, batch in enumerate(dataloader):
        imgs = batch["cand_composite_images"]
        paths = batch["cand_composite_paths"]
        bs = imgs.size(0)
        imgs = (
            rearrange(imgs, "b t c h w -> (b t) c h w")
            .contiguous()
            .to(accelerator.device)
        )
        with torch.inference_mode():
            logits = model(imgs)  # (b*t, 2)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        unharmony_probs = rearrange(
            probabilities[:, 1], "(b t) -> b t", b=bs
        ).contiguous()
        _, max_index = torch.max(unharmony_probs, dim=1)  # (b,)
        # 复制 max index 指定的图片
        for i, (p, idx) in enumerate(zip(paths, max_index)):
            os.system(
                f"cp {p[idx.item()]} {os.path.join(dataset_root, 'composite_images')}"
            )
    progress_bar.close()
    accelerator.wait_for_everyone()
