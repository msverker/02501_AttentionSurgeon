"""
train.py — Train and save linear probes for each downstream task.
Usage:
    uv run src/train.py --task cls
    uv run src/train.py --task seg
    uv run src/train.py --task det
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from baseline.backbone import (
    DinoV2Backbone,
    get_imagenet_loaders,
    get_ade20k_loaders,
    ClassificationHead,
    ADE20KHead,
    cache_features,
)

PROBE_PATHS = {
    "cls": "checkpoints/probe_imagenet.pt",
    "seg": "checkpoints/probe_ade20k.pt",
    "det": "checkpoints/probe_coco.pt",
}


def train_cls(backbone, device, epochs=5):
    print("Loading ImageNet...")
    train_loader, val_loader = get_imagenet_loaders(batch_size=64)

    print("Caching train features...")
    train_feats, train_labels = cache_features(backbone, train_loader)
    print("Caching val features...")
    val_feats, val_labels = cache_features(backbone, val_loader)

    train_cached = DataLoader(TensorDataset(train_feats, train_labels), batch_size=256, shuffle=True, num_workers=4)
    val_cached = DataLoader(TensorDataset(val_feats, val_labels), batch_size=256, shuffle=False, num_workers=4)

    probe = ClassificationHead(in_dim=768, num_classes=1000).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        probe.train()
        total_loss = 0
        for feats, labels in train_cached:
            feats, labels = feats.to(device), labels.to(device)
            loss = loss_fn(probe(feats), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # quick val accuracy
        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feats, labels in val_cached:
                feats, labels = feats.to(device), labels.to(device)
                preds = probe(feats).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch} | Loss: {total_loss:.2f} | Val Acc: {correct/total:.4f}")

    return probe


def train_seg(backbone, device, epochs=5):
    print("Loading ADE20K...")
    train_loader, val_loader = get_ade20k_loaders()

    probe = ADE20KHead(in_dim=768, num_classes=150).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        probe.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad():
                feats = backbone(imgs)
            logits = probe(feats)
            logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear")
            loss = loss_fn(logits, masks.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} | Loss: {total_loss:.2f}")

    return probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["cls", "seg", "det"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="retrain even if checkpoint exists")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_path = PROBE_PATHS[args.task]
    os.makedirs("checkpoints", exist_ok=True)

    backbone = DinoV2Backbone(device=device)

    if os.path.exists(probe_path) and not args.force:
        print(f"Checkpoint already exists at {probe_path}. Use --force to retrain.")
        return

    if args.task == "cls":
        probe = train_cls(backbone, device, args.epochs)
    elif args.task == "seg":
        probe = train_seg(backbone, device, args.epochs)
    elif args.task == "det":
        print("Detection not yet implemented")
        return

    torch.save(probe.state_dict(), probe_path)
    print(f"Saved probe to {probe_path}")


if __name__ == "__main__":
    main()