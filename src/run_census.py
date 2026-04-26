import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline.backbone import (
    ClassificationHead,
    DinoV2Backbone,
    cache_features,
    get_imagenet_loaders,
)
from head_census import AttentionCensus

device = "cuda"
backbone = DinoV2Backbone(device=device)
train_loader, val_loader = get_imagenet_loaders(batch_size=64)

# cache features
print("Caching train features...")
train_feats, train_labels = cache_features(backbone, train_loader, max_batches=800)  # cache ~50k samples for training probe
print("Caching val features...")
val_feats, val_labels = cache_features(backbone, val_loader, max_batches=200)  # cache ~10k samples for testing probe and census

train_cached = DataLoader(
    TensorDataset(train_feats, train_labels), batch_size=256, shuffle=True
)
val_cached = DataLoader(
    TensorDataset(val_feats, val_labels), batch_size=256, shuffle=False
)

# train probe
print("Training classification probe...")
probe = ClassificationHead(in_dim=768, num_classes=1000).to(device)
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    probe.train()
    for feats, labels in train_cached:
        feats, labels = feats.to(device), labels.to(device)
        loss = loss_fn(probe(feats), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} done")

# run census
print("Running census...")
census = AttentionCensus(backbone)
results = census.run(val_loader, num_batches=100)
importance = census.compute_importance(
    val_loader, probe, loss_fn, task="cls", num_batches=50
)
results["importance_cls"] = importance

# save
torch.save(results, "census_results.pt")
print("Done. Saved to census_results.pt")
for k, v in results.items():
    print(f"{k}: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

import numpy as np

np.savez(
    "head_profiles.npz",
    entropy=results["entropy"].numpy(),
    distance=results["distance"].numpy(),
    activation_mag=results["magnitude"].numpy(),
    importance=results["importance_cls"].detach().numpy(),
)
print("Saved head_profiles.npz")
