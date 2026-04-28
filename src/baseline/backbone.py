import torch
import torch.nn as nn
from transformers import AutoModel

from baseline.loaders import (
    get_cifar100_loaders,
    get_ade20k_dataloaders,
    get_coco_dataloaders,
)
from tqdm import tqdm


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
            output_hidden_states=False,
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
        self.conv = nn.Conv2d(in_dim, num_classes, kernel_size=3, padding=1)

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
        self.obj_head = nn.Conv2d(256, 1, 1)  # NEW

    def forward(self, feats):
        x = feats["patch_tokens"]
        B, N, C = x.shape
        H = W = int(N**0.5)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = torch.relu(self.conv(x))

        cls_logits = self.cls_head(x)
        box_preds = self.box_head(x)
        obj_logits = self.obj_head(x)

        return cls_logits, box_preds, obj_logits


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
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    backbone.eval()
    head.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            imgs, masks = batch["image"], batch["mask"]
            imgs = imgs.to(device)
            masks = masks.to(device)  # (B, H, W)
            with torch.no_grad():
                feats = backbone(imgs)
            logits = head(feats)  # (B, C, h, w)
            logits = nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = loss_fn(logits, masks.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / len(loader)
        print(f"[ADE20K] Epoch {epoch} Loss: {avg_loss:.4f}")

    return head


def build_targets(targets, grid_size=14, num_classes=80, device="cuda"):
    """
    Convert COCO targets into grid targets
    """
    B = len(targets)

    obj_target = torch.zeros((B, 1, grid_size, grid_size), device=device)
    cls_target = torch.zeros((B, grid_size, grid_size), dtype=torch.long, device=device)
    box_target = torch.zeros((B, 4, grid_size, grid_size), device=device)

    for b, t in enumerate(targets):
        boxes = t["boxes"]  # (N, 4) in COCO format [x, y, w, h]
        labels = t["labels"]

        for box, label in zip(boxes, labels):
            x, y, w, h = box

            # normalize (since image is 224x224)
            cx = (x + w / 2) / 224.0
            cy = (y + h / 2) / 224.0

            gx = int(cx * grid_size)
            gy = int(cy * grid_size)

            if gx >= grid_size or gy >= grid_size:
                continue

            obj_target[b, 0, gy, gx] = 1.0
            cls_target[b, gy, gx] = label

            box_target[b, :, gy, gx] = torch.tensor(
                [cx, cy, w / 224.0, h / 224.0], device=device
            )

    return obj_target, cls_target, box_target


def train_detection(backbone, head, train_loader, val_loader, epochs=5, device="cuda"):
    backbone.eval()  # frozen
    head.train()

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
    for epoch in range(epochs):
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for images, targets in pbar:
            images = images.to(device)
            # targets = targets["boxes"]

            # forward
            with torch.no_grad():
                feats = backbone(images)

            cls_logits, box_preds, obj_logits = head(feats)

            grid_size = cls_logits.shape[-1]

            obj_t, cls_t, box_t = build_targets(
                targets, grid_size=grid_size, device=device
            )

            # losses
            obj_loss = nn.functional.binary_cross_entropy_with_logits(obj_logits, obj_t)

            cls_loss = nn.functional.cross_entropy(
                cls_logits.permute(0, 2, 3, 1).reshape(-1, 80),
                cls_t.view(-1),
                reduction="mean",
            )

            box_loss = nn.functional.l1_loss(box_preds, box_t)

            loss = obj_loss + cls_loss + box_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1} | Avg loss: {avg_loss:.4f}")
        # print(obj_loss.item(), cls_loss.item(), box_loss.item())

        # evaluate_detection(backbone, head, val_loader, device)
    return head


def evaluate_detection(backbone, head, val_loader, device="cuda"):
    backbone.eval()
    head.eval()

    total_obj = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            feats = backbone(images)
            cls_logits, box_preds, obj_logits = head(feats)

            preds = torch.sigmoid(obj_logits) > 0.5
            total_obj += preds.sum().item()

    print(f"Detected objects (rough): {total_obj}")


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
        ],
        help="what to do when running the script (default: %(default)s)",
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    backbone = DinoV2Backbone(device)

    if args.mode == "train_classification":
        head = ClassificationHead().to(device)

        train_loader, val_loader = get_cifar100_loaders()

        train_classification(backbone, head, train_loader, val_loader, epochs=5)

    elif args.mode == "train_segmentation":
        head = ADE20KHead().to(device)

        train_loader, val_loader = get_ade20k_dataloaders(
            root="./data/archive/",
            image_size=(224, 224),
            batch_size=32,
            num_workers=4,
        )

        head = train_segmentation(backbone, head, train_loader, epochs=20)
        torch.save(
            head.state_dict(),
            "/work3/s216143/02501_AttentionSurgeon/src/checkpoints/ade20k_head.pth",
        )

    elif args.mode == "train_detection":
        head = CocoHead().to(device)

        train_loader, val_loader = get_coco_dataloaders(batch_size=32)

        head = train_detection(
            backbone, head, train_loader, val_loader, epochs=20, device=device
        )
        torch.save(
            head.state_dict(),
            "/work3/s216143/02501_AttentionSurgeon/src/checkpoints/coco_head.pth",
        )
