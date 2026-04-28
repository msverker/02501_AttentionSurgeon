import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class ADE20KDataset(Dataset):
    """
    ADE20K Semantic Segmentation Dataset.

    Folder structure expected:
        archive/
        └── ADEChallengeData2016/
            ├── images/
            │   ├── training/   *.jpg
            │   └── validation/ *.jpg
            └── annotations/
                ├── training/   *.png
                └── validation/ *.png
    """

    NUM_CLASSES = (
        150  # ADE20K has 150 semantic categories (1-indexed, 0 = background/ignore)
    )

    def __init__(
        self,
        root: str,
        split: str = "training",
        image_size: tuple[int, int] = (512, 512),
        augment: bool = False,
    ):
        assert split in (
            "training",
            "validation",
        ), "split must be 'training' or 'validation'"

        self.root = Path(root) / "ADEChallengeData2016"
        self.split = split
        self.image_size = image_size
        self.augment = augment

        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "annotations" / split

        # Pair images with their masks by stem name
        image_stems = {p.stem for p in self.image_dir.glob("*.jpg")}
        mask_stems = {p.stem for p in self.mask_dir.glob("*.png")}
        common = sorted(image_stems & mask_stems)

        if not common:
            raise FileNotFoundError(
                f"No matched image/annotation pairs found.\n"
                f"  images dir : {self.image_dir}\n"
                f"  annots dir : {self.mask_dir}"
            )

        self.samples = [
            (self.image_dir / f"{s}.jpg", self.mask_dir / f"{s}.png") for s in common
        ]

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _resize(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(
            image, self.image_size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        mask = TF.resize(
            mask, self.image_size, interpolation=transforms.InterpolationMode.NEAREST
        )
        return image, mask

    def _to_tensor(self, image: Image.Image, mask: Image.Image):
        image = TF.to_tensor(image)
        image = TF.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask - 1).clamp(min=-1)
        return image, mask

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # keep as-is (palette/L mode PNG)

        image, mask = self._resize(image, mask)

        image, mask = self._to_tensor(image, mask)

        return {
            "image": image,  # [3, H, W] float32
            "mask": mask,  # [H, W]    int64, values in [-1, 149]
            "image_path": str(img_path),
        }


class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, val_dir, solution_csv, synset_mapping, transform=None):
        self.val_dir = val_dir
        self.transform = transform

        # build synset -> index mapping
        synset_to_idx = {}
        with open(synset_mapping) as f:
            for idx, line in enumerate(f):
                synset = line.strip().split()[0]
                synset_to_idx[synset] = idx

        df = pd.read_csv(solution_csv)
        self.samples = [
            (
                row["ImageId"] + ".JPEG",
                synset_to_idx[row["PredictionString"].split()[0]],
            )
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.val_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats (works well with DINOv2)
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class CocoDataset(Dataset):
    def __init__(self, root, ann_file):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

        cats = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(cats)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(path).convert("RGB")

        # --- original size ---
        orig_w, orig_h = image.size

        # --- resize image ---
        image = image.resize((224, 224))
        image = transform(image)

        # --- scale factors ---
        scale_x = 224 / orig_w
        scale_y = 224 / orig_h

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # ✅ SCALE BOXES CORRECTLY
            x = x * scale_x
            y = y * scale_y
            w = w * scale_x
            h = h * scale_y

            boxes.append([x, y, w, h])
            labels.append(self.cat2label[ann["category_id"]])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}
