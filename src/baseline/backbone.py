import torch
import torch.nn as nn
from loaders import ADE20KDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel

# -----------------------------
# Data
# -----------------------------


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


# -----------------------------
# Backbone with head masking
# -----------------------------


class DinoV2Backbone(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            "facebook/dinov2-base", trust_remote_code=True, output_attentions=True
        )
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device
        self.to(device)

    def forward(self, x, head_mask=None, output_attentions=False):
        # head_mask shape: (num_layers, num_heads)
        outputs = self.model(
            x,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False
        )
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        patch_tokens = last_hidden_state[:, 1:, :]

        result = {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
        }

        if output_attentions:
            # tuple of (num_layers,) each (batch, num_heads, seq_len, seq_len)
            result["attentions"] = outputs.attentions
        
        return result


# -----------------------------
# Classification head
# -----------------------------


class ClassificationHead(nn.Module):
    def __init__(self, in_dim=768, num_classes=100):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feats):
        if isinstance(feats, dict):
            return self.fc(feats["cls_token"])
        return self.fc(feats)


class ADE20KHead(nn.Module):
    def __init__(self, in_dim=768, num_classes=150):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, feats):
        x = feats["patch_tokens"]
        B, N, C = x.shape
        H = W = int(N**0.5)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return self.conv(x)


class CocoHead(nn.Module):
    def __init__(self, in_dim=768, num_classes=80):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, 256, 3, padding=1)
        self.cls_head = nn.Conv2d(256, num_classes, 1)
        self.box_head = nn.Conv2d(256, 4, 1)

    def forward(self, feats):
        x = feats["patch_tokens"]
        B, N, C = x.shape
        H = W = int(N**0.5)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = torch.relu(self.conv(x))
        cls_logits = self.cls_head(x)
        box_preds = self.box_head(x)
        return cls_logits, box_preds


# -----------------------------
# Training + Eval
# -----------------------------


def train_classification(backbone, head, train_loader, val_loader, epochs=5):
    device = next(head.parameters()).device

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        head.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                feats = backbone(imgs)

            outputs = head(feats)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(backbone, head, val_loader)
        print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")


def train_segmentation(backbone, head, loader, epochs=5):
    device = next(head.parameters()).device
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    backbone.eval()
    head.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)  # (B, H, W)
            with torch.no_grad():
                feats = backbone(imgs)
            logits = head(feats)  # (B, C, h, w)
            logits = nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear"
            )
            loss = loss_fn(logits, masks.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[ADE20K] Epoch {epoch} Loss: {total_loss:.4f}")


def evaluate(backbone, head, loader):
    device = next(head.parameters()).device

    backbone.eval()
    head.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = head(backbone(imgs))
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# -----------------------------
# Feature caching (IMPORTANT for RL)
# -----------------------------


def cache_features(backbone, loader):
    device = next(backbone.parameters()).device

    features = []
    labels = []

    backbone.eval()

    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)["cls_token"].cpu()

            features.append(feats)
            labels.append(y)

    return torch.cat(features), torch.cat(labels)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train_classification",
        choices=[
            "train_classification",
            "train_segmentation",
            "train_detection",
            "test-plot",
        ],
        help="what to do when running the script (default: %(default)s)",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = DinoV2Backbone(device)

    if args.mode == "train_classification":
        head = ClassificationHead().to(device)

        train_loader, val_loader = get_cifar100_loaders()

        train_classification(backbone, head, train_loader, val_loader, epochs=5)

    elif args.mode == "train_segmentation":
        head = ADE20KHead().to(device)

        train_loader, val_loader = get_ade20k_loaders()

        train_segmentation(backbone, head, train_loader, epochs=5)
