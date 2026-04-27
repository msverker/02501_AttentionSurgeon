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
        
    def run_pruning_strategy(self, strategy, census, n_steps=72):
        # strategy: function that given current mask + census data
        #           returns which head to prune next
        # returns: list of (n_pruned, accuracy, flops) at each step
        mask = torch.ones(12, 12)
        results = []
        for step in range(n_steps):
            layer, head = strategy(mask, census)
            mask[layer, head] = 0
            acc = self.evaluate(mask)
            flops_ratio = (144 - (step + 1)) / 144
            reward = acc * flops_ratio
            results.append((step + 1, acc, flops_ratio, reward))
        return results

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
    from baseline.backbone import DinoV2Backbone, get_imagenet_loaders, ClassificationHead
    from pruning_evaluator import PruningEvaluator

    device = "cuda" if torch.cuda.is_available() else "mps"

    backbone = DinoV2Backbone(device=device)
    probe = ClassificationHead(in_dim=768, num_classes=1000).to(device)
    probe.load_state_dict(torch.load("checkpoints/probe_imagenet.pt"))
    probe.eval()

    _, val_loader = get_imagenet_loaders(batch_size=32)

    # load census
    census = dict(np.load("head_profiles.npz"))
    census = {k: torch.tensor(v) for k, v in census.items()}
    # rename to match strategy expectations
    census["activation_mag"] = census.pop("activation_mag")
    census["importance"] = census.pop("importance")

    evaluator = PruningEvaluator(backbone, probe, val_loader, task="cls")

    # test baseline accuracy (no pruning)
    baseline_acc = evaluator.evaluate(torch.ones(12, 12))
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # test random strategy for 5 steps
    results = evaluator.run_pruning_strategy(
        PruningEvaluator.random_strategy, census, n_steps=5
    )
    for n_pruned, acc, flops, reward in results:
        print(f"Pruned {n_pruned:3d} | Acc: {acc:.4f} | FLOPs: {flops:.3f} | Reward: {reward:.4f}")