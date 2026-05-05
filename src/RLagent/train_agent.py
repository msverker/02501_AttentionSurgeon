"""
train_agent.py — Train the pruning agent using MaskablePPO from sb3-contrib.

Replaces the hand-rolled PPO in pruning_agent.py.
MaskablePPO is identical to standard PPO but respects the action_masks()
method on the env, so the agent never wastes actions on already-pruned heads.

Usage:
    python train_agent.py --dataset imagenet --timesteps 100000
    python train_agent.py --dataset ade20k   --timesteps 100000
    python train_agent.py --dataset coco     --timesteps 100000
"""

import argparse
import os
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from pruning_env import HeadPruningEnv
from baseline.backbone import DinoV2Backbone, ClassificationHead, ADE20KHead, CocoHead
from baseline.loaders import get_imagenet_loaders, get_ade20k_dataloaders, get_coco_dataloaders
from config.agents_config import DATASET_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, default="imagenet",
                        choices=["imagenet", "ade20k", "coco"])
    parser.add_argument("--timesteps",  type=int, default=100_000,
                        help="Total environment steps to train for")
    parser.add_argument("--census",     type=str, default=None,
                        help="Path to head profiles .npz — defaults to task-specific file")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = (
        torch.device("cuda")  if torch.cuda.is_available() else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    # ── Dataset-specific setup ────────────────────────────────────────────
    # Each task needs its own probe and dataloader so the agent learns
    # pruning decisions that are actually useful for that task.
    if args.dataset == "imagenet":
        probe       = ClassificationHead(in_dim=768, num_classes=1000).to(device)
        probe.load_state_dict(torch.load("checkpoints/probe_imagenet.pt", map_location=device))
        _, loader   = get_imagenet_loaders()
        census_path = args.census or "results/head_profiles_imagenet.npz"

    elif args.dataset == "ade20k":
        probe       = ADE20KHead(in_dim=768, num_classes=150).to(device)
        probe.load_state_dict(torch.load("checkpoints/ade20k_head.pth", map_location=device))
        _, loader   = get_ade20k_dataloaders(root="data/archive")
        census_path = args.census or "results/head_profiles_ade20k.npz"

    elif args.dataset == "coco":
        probe       = CocoHead(in_dim=768, num_classes=80).to(device)
        probe.load_state_dict(torch.load("checkpoints/coco_head.pth", map_location=device))
        _, loader   = get_coco_dataloaders(batch_size=8)
        census_path = args.census or "results/head_profiles_coco.npz"

    probe.eval()
    backbone = DinoV2Backbone().to(device)

    # ── Census ────────────────────────────────────────────────────────────
    # Load task-specific head profiles so the agent's observations reflect
    # what each head does for THIS task, not classification by default.
    print(f"Loading census from: {census_path}")
    census_np = dict(np.load(census_path))
    census    = {k: torch.tensor(v) for k, v in census_np.items()}

    # ── Build env ─────────────────────────────────────────────────────────
    # DummyVecEnv wraps the env in the vectorised format SB3 expects.
    # We use a lambda so SB3 can re-instantiate it if needed.
    def make_env():
        return HeadPruningEnv(
            backbone    = backbone,
            probe       = probe,
            dataloader  = loader,
            census      = census,
            device      = device,
            config      = DATASET_CONFIGS,
            dataset_name= args.dataset,
        )

    env = DummyVecEnv([make_env])

    # ── MaskablePPO ───────────────────────────────────────────────────────
    # MlpPolicy builds a standard MLP actor-critic automatically.
    # The policy sees the 578-dim observation and outputs over 145 actions.
    # MaskablePPO calls env.action_masks() every step to block invalid actions.
    model = MaskablePPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = 3e-4,
        n_steps       = 144,          # collect one full episode before updating
        batch_size    = 64,
        n_epochs      = 10,           # how many times to reuse each collected batch
        gamma         = 0.99,         # discount factor
        clip_range    = 0.2,          # PPO clip parameter
        ent_coef      = 0.01,         # entropy bonus — encourages exploration
        verbose       = 1,
        tensorboard_log = f"logs/{args.dataset}/"
    )

    # ── Checkpoint callback ───────────────────────────────────────────────
    # Saves the model every 10k steps so you don't lose progress
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = "checkpoints/",
        name_prefix = f"rl_agent_{args.dataset}",
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"Training for {args.timesteps:,} timesteps on {args.dataset}...")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)

    # ── Save final model ──────────────────────────────────────────────────
    save_path = f"checkpoints/rl_agent_ppo_{args.dataset}"
    model.save(save_path)
    print(f"Saved to {save_path}.zip")


if __name__ == "__main__":
    main()