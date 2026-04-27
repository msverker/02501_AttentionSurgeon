import torch
from torch import nn


class PruningEvaluator:
    def __init__(self, backbone, probe, dataloader, task="cls"):
        # stores backbone, probe, eval dataloader
        self.backbone = backbone
        self.probe = probe
        self.dataloader = dataloader
        self.task = task

    def evaluate(self, head_mask, num_batches=20):
        # head_mask: (12, 12) binary tensor
        # returns: accuracy with this mask applied
        hooks = self._apply_pruning_hooks(self.backbone, head_mask)
        correct = 0
        total = 0
        device = next(self.probe.parameters()).device
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(self.dataloader):
                if i >= num_batches:
                    break
                imgs, labels = imgs.to(device), labels.to(device)
                feats = self.backbone(imgs)["cls_token"]
                outputs = self.probe(feats)
                preds = outputs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        for h in hooks:
            h.remove()
        return correct / total

    def run_pruning_strategy(self, strategy, census, n_steps=72, n_runs=5):
        # strategy: function that given current mask + census data
        #           returns (layer_idx, head_idx) to prune next
        # returns: list of (n_pruned, accuracy, flops) at each step
        all_results = []
        all_results = []

        for run in range(n_runs):  # repeat multiple times for averaging
            mask = torch.ones(12, 12)
            run_results = []
            for step in range(n_steps):
                layer, head = strategy(mask, census)
                mask[layer, head] = 0
                acc = self.evaluate(mask)
                flops_ratio = (144 - (step + 1)) / 144
                reward = acc * flops_ratio
                run_results.append([acc, flops_ratio, reward])
            all_results.append(run_results)

        all_results = torch.tensor(all_results)  # (n_runs, n_steps, 3)
        means = all_results.mean(dim=0)  # (n_steps, 3)
        stds = all_results.std(dim=0, correction=0)  # (n_steps, 3)
        return means, stds

    @staticmethod
    def random_strategy(mask, census):
        # pick a random unpruned head
        unpruned = torch.where(mask == 1)
        idx = torch.randint(len(unpruned[0]), (1,)).item()
        return unpruned[0][idx].item(), unpruned[1][idx].item()

    @staticmethod
    def magnitude_strategy(mask, census):
        # pick lowest magnitude unpruned head
        # mask out already pruned heads by setting their score to infinity
        scores = census["activation_mag"].clone()
        scores[mask == 0] = float("inf")  # don't pick already pruned
        # pick the minimum
        idx = scores.argmin()
        return idx // 12, idx % 12

    @staticmethod
    def importance_strategy(mask, census):
        # pick lowest importance unpruned head
        scores = census["importance"].clone()
        scores[mask == 0] = float("inf")
        idx = scores.argmin()
        return idx // 12, idx % 12

    @staticmethod
    def _apply_pruning_hooks(backbone, mask):
        """
        mask: (12, 12) binary tensor — 0 = prune, 1 = keep
        returns: list of hooks to remove later
        """
        hooks = []
        head_dim = 768 // 12

        def make_hook(layer_idx, mask_row):
            def hook(module, input, output):
                # output is (B, seq_len, 768)
                B, S, _ = output.shape
                ctx = output.view(B, S, 12, head_dim)
                ctx = ctx * mask_row.to(ctx.device).view(1, 1, 12, 1)
                return ctx.view(B, S, 768)

            return hook

        for i, layer in enumerate(backbone.model.encoder.layer):
            h = layer.attention.output.register_forward_hook(make_hook(i, mask[i]))
            hooks.append(h)

        return hooks


if __name__ == "__main__":
    import torch
    import numpy as np
    from baseline.backbone import (
        DinoV2Backbone,
        get_imagenet_loaders,
        ClassificationHead,
    )

    device = "cuda" if torch.cuda.is_available() else "mps"

    backbone = DinoV2Backbone(device=device)
    probe = ClassificationHead(in_dim=768, num_classes=1000).to(device)
    probe.load_state_dict(torch.load("checkpoints/probe_imagenet.pt"))
    probe.eval()

    _, val_loader = get_imagenet_loaders(batch_size=32)

    # load census
    census = dict(np.load("results/head_profiles_cls.npz"))
    census = {k: torch.tensor(v) for k, v in census.items()}
    # rename to match strategy expectations
    census["activation_mag"] = census.pop("activation_mag")
    census["importance"] = census.pop("importance")

    evaluator = PruningEvaluator(backbone, probe, val_loader, task="cls")

    # test baseline accuracy (no pruning)
    baseline_acc = evaluator.evaluate(torch.ones(12, 12))
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # test random strategy for 5 steps
    print("Testing random strategy for 2 steps...")
    random_means, random_stds = evaluator.run_pruning_strategy(
        PruningEvaluator.random_strategy, census, n_steps=2, n_runs=2
    )
    print("Done.")
    print("Testing magnitude strategy for 2 step...")
    mag_means, _ = evaluator.run_pruning_strategy(
        PruningEvaluator.magnitude_strategy, census, n_steps=2, n_runs=1
    )
    print("Done.")
    print("Testing importance strategy for 2 step...")
    imp_means, _ = evaluator.run_pruning_strategy(
        PruningEvaluator.importance_strategy, census, n_steps=2, n_runs=1
    )
    print("Done.")
    print("Random strategy results:")
    for step in range(len(random_means)):
        acc, flops, reward = random_means[step]
        acc_std, flops_std, reward_std = random_stds[step]
        print(
            f"Step {step + 1}: Acc={acc:.4f}±{acc_std:.4f}, Flops={flops:.4f}±{flops_std:.4f}, Reward={reward:.4f}±{reward_std:.4f}"
        )
    print("Magnitude strategy results:")
    for step in range(len(mag_means)):
        acc, flops, reward = mag_means[step]
        print(f"Step {step + 1}: Acc={acc:.4f}, Flops={flops:.4f}, Reward={reward:.4f}")
    print("Importance strategy results")
    for step in range(len(imp_means)):
        acc, flops, reward = imp_means[step]
        print(f"Step {step + 1}: Acc={acc:.4f}, Flops={flops:.4f}, Reward={reward:.4f}")
