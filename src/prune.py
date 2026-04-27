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
    def random_strategy(mask, census, max_per_layer=6):
        valid_mask = mask.clone()
        for l in range(12):
            if (mask[l] == 0).sum() >= max_per_layer:
                valid_mask[l] = 0
        unpruned = torch.where(valid_mask == 1)
        idx = torch.randint(len(unpruned[0]), (1,)).item()
        return unpruned[0][idx].item(), unpruned[1][idx].item()
    
    @staticmethod
    def importance_strategy(mask, census, max_per_layer=6):
        scores = census["importance"].clone()
        scores[mask == 0] = float("inf")
        for layer in range(12):
            if (mask[layer] == 0).sum() >= max_per_layer:
                scores[layer] = float("inf")
        idx = scores.argmin()
        return idx // 12, idx % 12

    @staticmethod
    def magnitude_strategy(mask, census, max_per_layer=6):
        scores = census["activation_mag"].clone()
        scores[mask == 0] = float("inf")
        for layer in range(12):
            if (mask[layer] == 0).sum() >= max_per_layer:
                scores[layer] = float("inf")
        idx = scores.argmin()
        return idx // 12, idx % 12


    def run_pruning_strategy(self, strategy, census, n_steps=72, n_runs=5):
        all_results = []

        for run in range(n_runs):  
            mask = torch.ones(12, 12)
            run_results = []
            
            for step in range(n_steps):
                layer, head = strategy(mask, census)
                mask[layer, head] = 0
                acc = self.evaluate(mask)
                
                # --- ADD THIS LINE ---
                # Feed the new accuracy back to the RL agent for its next state
                if hasattr(strategy, 'update_acc'):
                    strategy.update_acc(acc)
                # ---------------------
                
                flops_ratio = (144 - (step + 1)) / 144
                reward = acc * flops_ratio
                run_results.append([acc, flops_ratio, reward])
                
            all_results.append(run_results)

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
    
class PPOAgentStrategy:
    def __init__(self, policy_network, device="cuda"):
        self.policy_network = policy_network
        self.policy_network.eval()
        self.device = device
        # Track accuracy for the state vector
        self.current_acc = 1.0 

    def __call__(self, mask, census):
        """Matches the signature: strategy(mask, census) -> (layer, head)"""
        # 1. Calculate current FLOPs
        current_flops = mask.sum().item() / 144.0
        
        # 2. Build the state vector [Mask (144) | Acc (1) | FLOPs (1)]
        state = torch.cat([
            mask.flatten().to(self.device), 
            torch.tensor([self.current_acc, current_flops]).to(self.device)
        ])
        
        valid_mask = (mask.flatten() == 1.0).to(self.device)
        
        # 3. Get action probabilities from the Actor network
        with torch.no_grad():
            action_logits = self.policy_network.actor(state)
            
            # Mask out already-pruned heads
            action_logits[~valid_mask] = -1e8
            
            # For EVALUATION, we don't sample randomly. We pick the absolute best action.
            action_idx = torch.argmax(action_logits).item()
            
        layer = action_idx // 12
        head = action_idx % 12
        
        return layer, head
        
    def update_acc(self, new_acc):
        self.current_acc = new_acc


if __name__ == "__main__":
    # quick test to verify pruning hooks work as intended
    import torch
    import numpy as np
    from pruning_agent import PPOActorCritic
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


    print("Loading trained RL Agent...")
    # 2. Initialize the empty brain
    agent_net = PPOActorCritic(input_dim=146, action_dim=144).to(device)
    # 3. Load the trained weights
    agent_net.load_state_dict(torch.load("checkpoints/rl_agent_ppo.pt", map_location=device))
    
    # 4. Wrap it in your strategy class
    rl_strategy = PPOAgentStrategy(agent_net, device=device)

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
