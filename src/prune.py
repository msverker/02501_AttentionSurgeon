import torch
from torch import nn
from pruning_agent import TransformerPruningEnv

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
        hist = None # For segmentation mIoU
        device = next(self.probe.parameters()).device
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= num_batches:
                    break
                
                if self.task == "cls":
                    imgs, labels = batch
                    imgs, labels = imgs.to(device), labels.to(device)
                    feats = self.backbone(imgs)["cls_token"]
                    outputs = self.probe(feats)
                    preds = outputs.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                elif self.task == "seg":
                    import torch.nn.functional as F
                    imgs, labels = batch["image"], batch["mask"]
                    imgs, labels = imgs.to(device), labels.to(device)
                    feats = self.backbone(imgs)
                    outputs = self.probe(feats)
                    outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    preds = outputs.argmax(dim=1)
                    valid = labels != -1
                    
                    if hist is None:
                        num_classes = outputs.shape[1]
                        hist = torch.zeros(num_classes, num_classes, device=device)
                    
                    label_v = labels[valid]
                    pred_v = preds[valid]
                    inds = label_v * num_classes + pred_v
                    hist += torch.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes)
                    
                elif self.task == "det":
                    imgs, targets = batch
                    imgs = imgs.to(device)

                    feats = self.backbone(imgs)
                    cls_logits, box_preds, obj_logits = self.probe(feats)

                    # ---- 1. Convert predictions ----
                    obj_scores = obj_logits.sigmoid()                  # (B, 1, S, S)
                    cls_probs = cls_logits.softmax(dim=1)              # (B, C, S, S)
                    cls_scores, cls_preds = cls_probs.max(dim=1)       # (B, S, S)

                    scores = obj_scores.squeeze(1) * cls_scores        # combine obj + cls

                    # ---- 2. Decode boxes (YOU must match your encoding) ----
                    pred_boxes = self.decode_boxes(box_preds)          # (B, S, S, 4)

                    # ---- 3. Loop per image ----
                    for b in range(imgs.size(0)):
                        gt_boxes = targets[b]["boxes"].to(device)      # (N, 4)
                        gt_labels = targets[b]["labels"].to(device)

                        # flatten predictions
                        pred_b = pred_boxes[b].reshape(-1, 4)
                        pred_s = scores[b].reshape(-1)
                        pred_c = cls_preds[b].reshape(-1)

                        # ---- 4. Confidence filtering ----
                        keep = pred_s > 0.3
                        pred_b = pred_b[keep]
                        pred_s = pred_s[keep]
                        pred_c = pred_c[keep]

                        if pred_b.numel() == 0:
                            total += len(gt_boxes)   # all missed
                            continue

                        # ---- 5. IoU matching ----
                        ious = self.box_iou(pred_b, gt_boxes)  # (num_preds, num_gt)

                        matched_gt = set()
                        tp = 0

                        for i in range(pred_b.size(0)):
                            max_iou, j = ious[i].max(dim=0)

                            if max_iou > 0.5 and j.item() not in matched_gt:
                                if pred_c[i] == gt_labels[j]:
                                    tp += 1
                                    matched_gt.add(j.item())

                        fp = pred_b.size(0) - tp
                        fn = len(gt_boxes) - tp

                        correct += tp
                        total += (tp + fp + fn)
        for h in hooks:
            h.remove()
            
        if self.task == "seg":
            iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist) + 1e-6)
            valid_classes = hist.sum(dim=1) > 0
            return iou[valid_classes].mean().item()
            
        return correct / max(total, 1)

    # def run_pruning_strategy(self, strategy, census, n_steps=72, n_runs=5):
    #     # strategy: function that given current mask + census data
    #     #           returns (layer_idx, head_idx) to prune next
    #     # returns: list of (n_pruned, accuracy, flops) at each step
    #     all_results = []
    #     all_results = []

    #     for run in range(n_runs):  # repeat multiple times for averaging
    #         mask = torch.ones(12, 12)
    #         run_results = []
    #         for step in range(n_steps):
    #             layer, head = strategy(mask, census)
    #             mask[layer, head] = 0
    #             acc = self.evaluate(mask)
    #             flops_ratio = (144 - (step + 1)) / 144
    #             reward = acc / flops_ratio
    #             run_results.append([acc, flops_ratio, reward])
    #         all_results.append(run_results)
            

    #     all_results = torch.tensor(all_results)  # (n_runs, n_steps, 3)
    #     means = all_results.mean(dim=0)  # (n_steps, 3)
    #     stds = all_results.std(dim=0, correction=0)  # (n_steps, 3)
    #     return means, stds
    
    @staticmethod
    def random_strategy(mask, census, max_per_layer=12):
        valid_mask = mask.clone()
        for l in range(12):
            if (mask[l] == 0).sum() >= max_per_layer:
                valid_mask[l] = 0
        unpruned = torch.where(valid_mask == 1)
        idx = torch.randint(len(unpruned[0]), (1,)).item()
        
        return unpruned[0][idx].item(), unpruned[1][idx].item()
    
    @staticmethod
    def importance_strategy(mask, census, max_per_layer=12):
        scores = census["importance"].clone()
        scores[mask == 0] = float("inf")
        for layer in range(12):
            if (mask[layer] == 0).sum() >= max_per_layer:
                scores[layer] = float("inf")
        idx = scores.argmin()
        return idx // 12, idx % 12

    @staticmethod
    def magnitude_strategy(mask, census, max_per_layer=12):
        scores = census["activation_mag"].clone()
        scores[mask == 0] = float("inf")
        for layer in range(12):
            if (mask[layer] == 0).sum() >= max_per_layer:
                scores[layer] = float("inf")
        idx = scores.argmin()
        return idx // 12, idx % 12
    
    @staticmethod
    def uniform_strategy(mask, census):
        # prune from the layer with the most remaining heads
        # within that layer, pick the least important head
        heads_per_layer = mask.sum(dim=1)  # (12,)
        
        # pick layer with most remaining heads (break ties by lowest index)
        layer = heads_per_layer.argmax().item()
        
        # within that layer, pick lowest importance
        scores = census["importance"][layer].clone()
        scores[mask[layer] == 0] = float("inf")
        head = scores.argmin().item()
        
        return layer, head


    def run_pruning_strategy(self, strategy, census, n_steps=72, n_runs=5, dataset_name=None):
        all_results = []

        for run in range(n_runs):  
            mask = torch.ones(12, 12)
            run_results = []
            
            if hasattr(strategy, 'reset'):
                strategy.reset()
            
            for step in range(n_steps):
                layer, head = strategy(mask, census)
                mask[layer, head] = 0
                acc = self.evaluate(mask)
                                
                flops_ratio = (144 - (step + 1)) / 144
                reward = acc * flops_ratio
                
                # Feed the new proxy-loss/reward back to the RL agent for its next state
                if hasattr(strategy, 'update_state'):
                    loss_proxy = 1.0 - acc # the proxy loss since evaluators return acc natively
                    strategy.update_state()
                    
                run_results.append([acc, flops_ratio, reward])
                
            all_results.append(run_results)
            
        if dataset_name is not None and hasattr(strategy, 'update_state'):
            import os
            import numpy as np
            os.makedirs("results", exist_ok=True)
            strat_name = strategy.__name__ if hasattr(strategy, "__name__") else strategy.__class__.__name__
            save_path = f"results/final_mask_{dataset_name}_{strat_name}.npz"
            np.savez(save_path, mask=mask.cpu().numpy())
            print(f"Saved final mask to {save_path}")


        all_results = torch.tensor(all_results)  
        means = all_results.mean(dim=0)  
        stds = all_results.std(dim=0, correction=0)  
        return means, stds

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
                ctx = output[0]  # (B, S, 768)
                B, S, _ = ctx.shape
                ctx = ctx.view(B, S, 12, head_dim)
                ctx = ctx * mask_row.to(ctx.device).view(1, 1, 12, 1)
                return (ctx.view(B, S, 768),) + output[1:]
            return hook

        for i, layer in enumerate(backbone.model.encoder.layer):
            h = layer.attention.attention.register_forward_hook(make_hook(i, mask[i]))
            hooks.append(h)

        return hooks
    
class PPOAgentStrategy:
    def __init__(self, policy_network, device="cuda", env=None):
        self.policy_network = policy_network
        self.policy_network.eval()
        self.device = device
        self.env: TransformerPruningEnv = env
        self.reset()
        
    def reset(self):
        self.current_state = self.env.reset()

    def __call__(self, mask, census):
        """Matches the signature: strategy(mask, census) -> (layer, head)"""
        state = self.current_state
            
        stop_action_valid = torch.tensor([False], dtype=torch.bool, device=self.device) # STOP FUNCTION OFF (False)
        head_mask = (self.env.mask.flatten() == 1.0).to(torch.bool)
        valid_mask = torch.cat([head_mask, stop_action_valid])
        
        with torch.no_grad():
            action_logits = self.policy_network.actor(state)
            
            # Mask out already-pruned heads & the stop action
            action_logits[~valid_mask] = -1e8
            
            # For EVALUATION, we don't sample randomly. We pick the absolute best action.
            action_idx = torch.argmax(action_logits).item()
            
        layer = action_idx // 12
        head = action_idx % 12
        
        self.last_action_idx = action_idx
            
        return layer, head
        
    def update_state(self, *args, **kwargs):
            # ignore proxies, step the actual environment using last action
            self.current_state, _ , _ = self.env.step(self.last_action_idx)


if __name__ == "__main__":
    # quick test to verify pruning hooks work as intended
    import torch
    import numpy as np
    from pruning_agent import PPOActorCritic
    from baseline.backbone import (
        DinoV2Backbone,
        ClassificationHead,
    )
    from baseline.loaders import get_imagenet_loaders

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


    print("Loading trained RL Agent...")
    # 2. Initialize the empty brain
    agent_net = PPOActorCritic(input_dim=147, action_dim=145).to(device)
    # 3. Load the trained weights
    agent_net.load_state_dict(torch.load("checkpoints/rl_agent_ppo.pt", map_location=device))
    
    # 3.5. Load training environment for accurate real-time state extraction
    import sys
    sys.path.append(".")
    from config.agents_config import DATASET_CONFIGS
    from pruning_agent import TransformerPruningEnv
    env_rl = TransformerPruningEnv(
        backbone=backbone, 
        probe=probe, 
        dataloader=val_loader, 
        device=device,
        config=DATASET_CONFIGS,
        dataset_name="imagenet"
    )
    
    # 4. Wrap it in your strategy class
    rl_strategy = PPOAgentStrategy(agent_net, device=device, env=env_rl)

    # 5. Run the evaluation
    print("Testing RL Agent strategy for 2 steps...")
    rl_means, _ = evaluator.run_pruning_strategy(
        rl_strategy, census, n_steps=2, n_runs=1
    )
    print("Done.")

    # 6. Print the results
    print("RL Agent strategy results:")
    for step in range(len(rl_means)):
        acc, flops, reward = rl_means[step]
        print(f"Step {step + 1}: Acc={acc:.4f}, Flops={flops:.4f}, Reward={reward:.4f}")

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