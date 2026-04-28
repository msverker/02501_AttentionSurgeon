import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor

from baseline.data import ADE20KDataset, ImageNetValDataset, CocoDataset


# -----------------------------
# Data
# -----------------------------
def get_imagenet_loaders(batch_size=64):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    train_set = datasets.ImageFolder(
        "/dtu/imagenet/ILSVRC/Data/CLS-LOC/train", transform=transform
    )
    val_set = ImageNetValDataset(
        val_dir="/dtu/imagenet/ILSVRC/Data/CLS-LOC/val",
        solution_csv="/dtu/imagenet/LOC_val_solution.csv",
        synset_mapping="/dtu/imagenet/LOC_synset_mapping.txt",
        transform=transform,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return train_loader, val_loader


def get_cifar100_loaders(batch_size=64):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    train_set = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    val_set = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader


def get_ade20k_dataloaders(
    root: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Returns {'train': DataLoader, 'val': DataLoader}."""

    train_ds = ADE20KDataset(
        root, split="training", image_size=image_size, augment=True
    )
    val_ds = ADE20KDataset(
        root, split="validation", image_size=image_size, augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def get_coco_dataloaders(batch_size=8):
    root = "/work3/s216143/02501_AttentionSurgeon/data/coco/val2017"
    ann = "/work3/s216143/02501_AttentionSurgeon/data/coco/annotations/instances_val2017.json"

    dataset = CocoDataset(root, ann)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader
