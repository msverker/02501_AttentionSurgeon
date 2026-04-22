import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor


class ADE20KDataset(Dataset):
    def __init__(self, root, split="training", image_size=224):
        self.root = root
        self.split = split
        self.image_size = image_size

        self.images_dir = os.path.join(root, "images", split)
        self.annotations_dir = os.path.join(root, "annotations", split)

        self.image_files = sorted(
            [f for f in os.listdir(self.images_dir) if f.endswith(".jpg")]
        )

        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        ann_path = os.path.join(self.annotations_dir, img_name.replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path)

        # --- Resize mask ONLY (image handled by processor) ---
        mask = TF.resize(
            mask, (self.image_size, self.image_size), interpolation=Image.NEAREST
        )

        # --- Use DINOv2 processor ---
        processed = self.processor(images=image, return_tensors="pt")

        image = processed["pixel_values"].squeeze(0)  # (3, 224, 224)

        # --- Mask processing ---
        mask = torch.from_numpy(np.array(mask)).long()

        mask = mask - 1
        mask[mask == -1] = 255  # ignore index

        return image, mask
