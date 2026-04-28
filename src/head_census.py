import torch


class AttentionCensus:
    def __init__(self, backbone, num_layers=12, num_heads=12, grid_size=16):
        self.backbone = backbone
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.grid_size = grid_size  # 224 / 16 = 14 patches per side

    def compute_entropy(self, attentions):
        # attentions: list of num_layers tensors, each (B, num_heads, seq_len, seq_len)
        # return: (num_layers, num_heads)
        accum_entropy = []
        for attn in attentions:
            # attn: (B, num_heads, 197, 197)
            # slice out CLS token: attn[:, :, 1:, 1:] -> (B, num_heads, 196, 196)
            # compute entropy across the last dimension (softmax already applied in backbone)
            attn = attn[:, :, 1:, 1:]  # (B, num_heads, 196, 196)
            attn = attn / attn.sum(dim=-1, keepdim=True)  # normalize
            entropy = -torch.sum(
                attn * torch.log(attn + 1e-8), dim=-1
            )  # (B, num_heads, 196)
            entropy = entropy.mean(dim=(0, 2))  # (num_heads,)
            accum_entropy.append(entropy)

        return torch.stack(accum_entropy, dim=0)  # (num_layers, num_heads)

    def compute_distance(self, attentions):
        # attentions: same as above
        # return: (num_layers, num_heads)
        # CLS has no spatial position — slice it out: attn[:, :, 1:, 1:]
        # build a (196, 196) pairwise distance matrix from patch grid positions

        grid_coords = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(self.grid_size),
                    torch.arange(self.grid_size),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 2)
            .float()
        )
        dist_matrix = torch.cdist(grid_coords, grid_coords)
        dist_matrix = dist_matrix / dist_matrix.max()  # (196, 196)

        accum_distance = []
        for attn in attentions:
            attn = attn[:, :, 1:, 1:]  # (B, num_heads, 196, 196)
            attn = attn / attn.sum(dim=-1, keepdim=True)  # renormalize
            weighted_dist = attn * dist_matrix.to(attn.device).unsqueeze(0).unsqueeze(0)
            avg_distance = weighted_dist.sum(dim=-1).mean(dim=(0, 2))  # (num_heads,)
            accum_distance.append(avg_distance)

        return torch.stack(accum_distance, dim=0)  # (num_layers, num_heads)

    def compute_cls_entropy(self, attentions):
        # CLS token as query (row 0), attending to all patch tokens
        # Low entropy = focused CLS attention = CLS-aggregator head
        accum = []
        for attn in attentions:
            cls_attn = attn[
                :, :, 0, 1:
            ]  # (B, num_heads, 256) — CLS attending to patches
            cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True)  # renormalize
            entropy = -torch.sum(
                cls_attn * torch.log(cls_attn + 1e-8), dim=-1
            )  # (B, num_heads)
            entropy = entropy.mean(dim=0)  # (num_heads,)
            accum.append(entropy)
        return torch.stack(accum, dim=0)  # (num_layers, num_heads)

    @torch.no_grad()
    def run(self, dataloader, num_batches=50):
        self.backbone.eval()

        hooks = []
        buffer = {}
        head_dim = 768 // self.num_heads  # 64

        entropy_accum = torch.zeros(self.num_layers, self.num_heads)
        distance_accum = torch.zeros(self.num_layers, self.num_heads)
        magnitude_accum = torch.zeros(self.num_layers, self.num_heads)
        cls_entropy_accum = torch.zeros(self.num_layers, self.num_heads)

        def make_hook(layer_idx):
            def hook(module, input, output):
                ctx = output[0]  # (B, seq_len, 768)
                B, S, D = ctx.shape
                # reshape back to per-head
                ctx_per_head = ctx.view(B, S, self.num_heads, head_dim)
                ctx_per_head = ctx_per_head.permute(
                    0, 2, 1, 3
                )  # (B, num_heads, seq_len, head_dim)
                # L2 norm per head, averaged over batch and tokens
                buffer[layer_idx] = ctx_per_head.norm(dim=-1).mean(
                    dim=(0, 2)
                )  # (num_heads,)

            return hook

        # register hooks on all layers
        for i, layer in enumerate(self.backbone.model.encoder.layer):
            h = layer.attention.attention.register_forward_hook(make_hook(i))
            hooks.append(h)

        for i, (imgs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            imgs = imgs.to(self.backbone.device)
            outputs = self.backbone(imgs, output_attentions=True)
            attentions = outputs["attentions"]  # tuple of 12 x (B, 12, 197, 197)

            for layer_idx in range(self.num_layers):
                magnitude_accum[layer_idx] += buffer[layer_idx].cpu()

            entropy_accum += self.compute_entropy(attentions).cpu()
            distance_accum += self.compute_distance(attentions).cpu()
            cls_entropy_accum += self.compute_cls_entropy(attentions).cpu()

        for h in hooks:
            h.remove()

        return {
            "entropy": entropy_accum / num_batches,
            "distance": distance_accum / num_batches,
            "magnitude": magnitude_accum / num_batches,
            "cls_entropy": cls_entropy_accum / num_batches,
        }

    def compute_importance(
        self, dataloader, probe, loss_fn, task="cls", num_batches=50
    ):
        importance_accum = torch.zeros(self.num_layers, self.num_heads)
        attn_buffer = {}
        grad_buffer = {}
        hooks = []

        # temporarily unfreeze backbone so attention tensors have requires_grad=True
        for p in self.backbone.model.parameters():
            p.requires_grad_(True)

        def make_hook(layer_idx):
            def hook(module, input, output):
                attn = output[1]
                attn_buffer[layer_idx] = attn
                attn.register_hook(
                    lambda grad: grad_buffer.__setitem__(layer_idx, grad)
                )

            return hook

        for i, layer in enumerate(self.backbone.model.encoder.layer):
            h = layer.attention.attention.register_forward_hook(make_hook(i))
            hooks.append(h)

        for i, (imgs, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
            imgs = imgs.to(self.backbone.device)
            labels = labels.to(self.backbone.device)

            self.backbone.model.zero_grad()
            outputs = self.backbone(imgs, output_attentions=True)

            if task == "cls":
                logits = probe(outputs)

            loss = loss_fn(logits, labels)
            loss.backward()

            for layer_idx in range(self.num_layers):
                attn = attn_buffer[layer_idx]
                grad = grad_buffer[layer_idx]
                importance = (grad * attn).abs().sum(dim=(2, 3)).mean(dim=0)
                importance_accum[layer_idx] += importance.cpu()

        for h in hooks:
            h.remove()

        # refreeze backbone
        for p in self.backbone.model.parameters():
            p.requires_grad_(False)

        return importance_accum / num_batches


if __name__ == "__main__":
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset, TensorDataset
    from torchvision import datasets, transforms
    from transformers import AutoImageProcessor

    from baseline.backbone import (
        ClassificationHead,
        DinoV2Backbone,
        cache_features,
        get_cifar100_loaders,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    backbone = DinoV2Backbone(device=device)

    # build a small val subset directly
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )
    val_set = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )
    val_subset = Subset(val_set, indices=range(5))
    val_loader = DataLoader(val_subset, batch_size=5, shuffle=False)

    # cache just the 500 images
    print("Caching features...")
    val_feats, val_labels = cache_features(backbone, val_loader)
    val_cached = DataLoader(
        TensorDataset(val_feats, val_labels), batch_size=64, shuffle=False
    )

    # train probe on cached features
    probe = ClassificationHead(in_dim=768, num_classes=100).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        probe.train()
        for feats, labels in val_cached:
            feats, labels = feats.to(device), labels.to(device)
            loss = loss_fn(probe({"cls_token": feats}), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} done")

    # test importance
    census = AttentionCensus(backbone)
    importance = census.compute_importance(
        val_loader, probe, loss_fn, task="cls", num_batches=5
    )
    print(
        f"importance: shape={importance.shape}, min={importance.min():.4f}, max={importance.max():.4f}"
    )
