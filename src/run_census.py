import torch
import torch.nn as nn
import numpy as np
from baseline.backbone import DinoV2Backbone, get_imagenet_loaders, ClassificationHead
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

torch.cuda.empty_cache()

print("Running importance scoring...")
importance = census.compute_importance(val_loader, probe, loss_fn, task="cls", num_batches=100)
results["importance_cls"] = importance

np.savez(
    "head_profiles.npz",
    entropy=results["entropy"].numpy(),
    distance=results["distance"].numpy(),
    activation_mag=results["magnitude"].numpy(),
    importance=results["importance_cls"].detach().numpy(),
)
print("Saved head_profiles.npz")