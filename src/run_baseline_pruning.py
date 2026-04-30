import argparse
import torch
import numpy as np
from pathlib import Path
from baseline.backbone import DinoV2Backbone, ClassificationHead, ADE20KHead, CocoHead
from baseline.loaders import get_imagenet_loaders, get_cifar100_loaders, get_ade20k_dataloaders, get_coco_dataloaders
from pruning_agent import PPOActorCritic
from prune import PruningEvaluator, PPOAgentStrategy

def main():
    parser = argparse.ArgumentParser(description="Run baseline pruning strategies")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "cifar100", "ade20k", "coco"], help="Dataset to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for validation loader")
    parser.add_argument("--probe-ckpt", type=str, default=None, help="Path to head checkpoint")
    parser.add_argument("--census", type=str, default="results/head_profiles_cls.npz", help="Path to head profiles numpy file")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--rl-agent-ckpt", type=str, default="checkpoints/rl_agent_ppo.pt", help="Path to RL agent checkpoint")
    parser.add_argument("--run-agent", action="store_true", help="Whether to also run the RL pruning agent")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up dataset specific parameters
    if args.dataset == "imagenet":
        num_classes = 1000
        get_loaders_fn = get_imagenet_loaders
        probe_ckpt = args.probe_ckpt or "checkpoints/probe_imagenet.pt"
        output_file = args.output or "results/baseline_pruning_imagenet_results.npz"
        head_cls = ClassificationHead
        task = "cls"
    elif args.dataset == "cifar100":
        num_classes = 100
        get_loaders_fn = get_cifar100_loaders
        probe_ckpt = args.probe_ckpt or "checkpoints/probe_cifar100.pt"
        output_file = args.output or "results/baseline_pruning_cifar100_results.npz"
        head_cls = ClassificationHead
        task = "cls"
    elif args.dataset == "ade20k":
        num_classes = 150
        get_loaders_fn = lambda batch_size: get_ade20k_dataloaders(root="./data/archive", batch_size=batch_size)
        probe_ckpt = args.probe_ckpt or "checkpoints/ade20k_head.pth"
        output_file = args.output or "results/baseline_pruning_ade20k_results.npz"
        head_cls = ADE20KHead
        task = "seg"
    elif args.dataset == "coco":
        num_classes = 80
        get_loaders_fn = get_coco_dataloaders
        probe_ckpt = args.probe_ckpt or "checkpoints/coco_head.pth"
        output_file = args.output or "results/baseline_pruning_coco_results.npz"
        head_cls = CocoHead
        task = "det"
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # backbone and probe
    backbone = DinoV2Backbone(device=device)
    probe = head_cls(in_dim=768, num_classes=num_classes).to(device)
    
    if Path(probe_ckpt).exists():
        probe.load_state_dict(torch.load(probe_ckpt, map_location=device))
        print(f"Loaded probe: {probe_ckpt}")
    else:
        print(f"Warning: No probe at {probe_ckpt}")
    probe.eval()

    # loader
    if args.dataset == "ade20k":
        loaders = get_loaders_fn(batch_size=args.batch_size)
        val_loader = loaders["val"] if isinstance(loaders, dict) else loaders[1]
    else:
        _, val_loader = get_loaders_fn(batch_size=args.batch_size)

    # census
    census = dict(np.load(args.census))
    census = {k: torch.tensor(v) for k, v in census.items()}

    evaluator = PruningEvaluator(backbone, probe, val_loader, task=task)

    # Baseline performance
    baseline_acc = evaluator.evaluate(torch.ones(12, 12), num_batches=100)
    print(f"Baseline Accuracy ({task}): {baseline_acc:.4f}")

    # Results dictionary
    results = {
        "baseline_acc": np.array([baseline_acc])
    }

    # # Run standard baselines
    # for name, strategy, runs in [
    #     ("random", PruningEvaluator.random_strategy, 5),
    #     ("magnitude", PruningEvaluator.magnitude_strategy, 1),
    #     ("importance", PruningEvaluator.importance_strategy, 1),
    #     ("uniform", PruningEvaluator.uniform_strategy, 1)
    # ]:
    #     print(f"Running {name} strategy...")
    #     means, stds = evaluator.run_pruning_strategy(strategy, census, n_steps=72, n_runs=runs)
    #     results[f"{name}_means"] = means.numpy()
    #     results[f"{name}_stds"] = stds.numpy()

    # Optional RL Agent block
    if args.run_agent:
        import sys
        sys.path.append(".")
        from config.agents_config import DATASET_CONFIGS
        from pruning_agent import TransformerPruningEnv
        
        print("Loading PPO RL agent...")
        ppo_net = PPOActorCritic(input_dim=147, action_action_dim=145 if "rl_agent_ppo" in args.rl_agent_ckpt else 145) # Assuming action_dim=145 for 147 input dim
        ppo_net = ppo_net.to(device) # Redoing to ensure proper creation
        
        ppo_net = PPOActorCritic(input_dim=147, action_dim=145).to(device)
        
        if Path(args.rl_agent_ckpt).exists():
            ppo_net.load_state_dict(torch.load(args.rl_agent_ckpt, map_location=device))
            print(f"RL Agent loaded from {args.rl_agent_ckpt}")
        else:
            print(f"Warning: RL Agent not found, using random weights.")
            
        print("Initializing identical RL environment state...")
        env_rl = TransformerPruningEnv(
            backbone=backbone, 
            probe=probe, 
            dataloader=val_loader, 
            device=device,
            config=DATASET_CONFIGS,
            dataset_name=args.dataset
        )
        
        ppo_strategy = PPOAgentStrategy(ppo_net, device=device, env=env_rl)
        print("Running RL (PPO) strategy...")
        rl_means, rl_stds = evaluator.run_pruning_strategy(ppo_strategy, census, n_steps=72, n_runs=1)
        results["rl_means"] = rl_means.numpy()
        results["rl_stds"] = rl_stds.numpy()

    # Save
    print(f"Saving results to {output_file}...")
    np.savez(output_file, **results)
    print("All done.")

if __name__ == "__main__":
    main()