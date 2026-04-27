import gc

import numpy as np
import torch
import torch.nn as nn

from baseline.backbone import ClassificationHead, DinoV2Backbone, get_imagenet_loaders
from head_census import AttentionCensus

device = "cuda"
backbone = DinoV2Backbone(device=device)
_, val_loader = get_imagenet_loaders(batch_size=16)

probe = ClassificationHead(in_dim=768, num_classes=1000).to(device)
probe.load_state_dict(torch.load("checkpoints/probe_imagenet.pt", map_location=device))
probe.eval()

loss_fn = nn.CrossEntropyLoss()
census = AttentionCensus(backbone)

print("Running task-agnostic metrics...")
results = census.run(val_loader, num_batches=200)

del val_loader
torch.cuda.empty_cache()
gc.collect()

_, val_loader_small = get_imagenet_loaders(batch_size=4)
importance = census.compute_importance(val_loader_small, probe, loss_fn, task="cls", num_batches=400)

results["importance_cls"] = importance

np.savez(
    "head_profiles.npz",
    entropy=results["entropy"].numpy(),
    distance=results["distance"].numpy(),
    activation_mag=results["magnitude"].numpy(),
    importance=results["importance_cls"].detach().numpy(),
)
print("Saved head_profiles.npz")