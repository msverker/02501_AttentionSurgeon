import torch
import numpy as np
from baseline.backbone import DinoV2Backbone, get_imagenet_loaders, ClassificationHead
from prune import PruningEvaluator

device = "cuda" if torch.cuda.is_available() else "mps"

# load backbone
print("Loading backbone...")
backbone = DinoV2Backbone(device=device)

# load probe
print("Loading probe...")
probe = ClassificationHead(in_dim=768, num_classes=1000).to(device)
probe.load_state_dict(torch.load("checkpoints/probe_imagenet.pt", map_location=device))
probe.eval()

# load val loader
print("Loading ImageNet val...")
_, val_loader = get_imagenet_loaders(batch_size=32)

# load census
print("Loading census...")
census = dict(np.load("results/head_profiles_cls.npz"))
census = {k: torch.tensor(v) for k, v in census.items()}

# evaluator
evaluator = PruningEvaluator(backbone, probe, val_loader, task="cls")

# baseline accuracy
print("Evaluating baseline...")
baseline_acc = evaluator.evaluate(torch.ones(12, 12), num_batches=100)
print(f"Baseline accuracy: {baseline_acc:.4f}")

# random strategy — n_runs=5 since order varies
print("Running random strategy (n_runs=5)...")
random_means, random_stds = evaluator.run_pruning_strategy(
    PruningEvaluator.random_strategy, census, n_steps=72, n_runs=5
)
print("Done.")

# magnitude strategy — deterministic, n_runs=1
print("Running magnitude strategy...")
mag_means, mag_stds = evaluator.run_pruning_strategy(
    PruningEvaluator.magnitude_strategy, census, n_steps=72, n_runs=1
)
print("Done.")

# importance strategy — deterministic, n_runs=1
print("Running importance strategy...")
imp_means, imp_stds = evaluator.run_pruning_strategy(
    PruningEvaluator.importance_strategy, census, n_steps=72, n_runs=1
)
print("Done.")

# save
np.savez(
    "results/baseline_pruning_cls_results.npz",
    baseline_acc=np.array([baseline_acc]),
    random_means=random_means.numpy(),
    random_stds=random_stds.numpy(),
    magnitude_means=mag_means.numpy(),
    magnitude_stds=mag_stds.numpy(),
    importance_means=imp_means.numpy(),
    importance_stds=imp_stds.numpy(),
)
print("Saved baseline_pruning_cls_results.npz")