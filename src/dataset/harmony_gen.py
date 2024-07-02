import os.path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from argparse import Namespace
from PIL import Image
import json
import os


class GenHarmonyDataset(Dataset):
    def __init__(self, dataset_root, resolution):
        self.metadata = [
            json.loads(line)
            for line in open(os.path.join(dataset_root, "metadata.jsonl")).readlines()
        ]
        self.dataset_root = dataset_root
        self.resolution = resolution
        self.transforms = Namespace(
            resize=transforms.Resize(
                [self.resolution, self.resolution],
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            convert = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]),
        )
        
    def __len__(self):
        """返回图像的总数。"""
        return len(self.metadata)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_root, self.metadata[index]["sample"])
        cond_path = self.metadata[index]["cond"]
        target_path = self.metadata[index]["target"]
        
        image = Image.open(image_path).convert("RGB")
        if image.size != (self.resolution, self.resolution):
            image = self.transforms.resize(image)
        
        cond_image = Image.open(cond_path).convert("RGB")
        if cond_image.size != (self.resolution, self.resolution):
            cond_image = self.transforms.resize(cond_image)
            
        target_image = Image.open(target_path).convert("RGB")
        if target_image.size != (self.resolution, self.resolution):
            target_image = self.transforms.resize(target_image)
            
        image= self.transforms.convert(image)
        cond_image = self.transforms.convert(cond_image)
        target_image = self.transforms.convert(target_image)
        
        return {
            "image": image,
            "cond": cond_image,
            "target": target_image,
            "image_path": image_path,
            "cond_path": cond_path,
            "target_path": target_path,
        }