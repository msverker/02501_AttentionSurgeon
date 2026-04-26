
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor

from data import ADE20KDataset, ImageNetValDataset


# -----------------------------
# Data
# -----------------------------
def get_imagenet_loaders(batch_size=64):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    train_set = datasets.ImageFolder(
        "/dtu/imagenet/ILSVRC/Data/CLS-LOC/train", 
        transform=transform
    )
    val_set = ImageNetValDataset(
        val_dir="/dtu/imagenet/ILSVRC/Data/CLS-LOC/val",
        solution_csv="/dtu/imagenet/LOC_val_solution.csv",
        synset_mapping="/dtu/imagenet/LOC_synset_mapping.txt",
        transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

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


def get_ade20k_loaders(batch_size=16):
    root = (
        "/Users/madssverker/Documents/"
        "02501/02501_AttentionSurgeon/data/archive/ADEChallengeData2016"
    )

    train_set = ADE20KDataset(root=root, split="training")
    val_set = ADE20KDataset(root=root, split="validation")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader
