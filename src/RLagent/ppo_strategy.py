"""
ppo_strategy.py — Drop-in replacement for PPOAgentStrategy in prune.py.

Loads a trained SB3 MaskablePPO model and wraps it in the same
strategy(mask, census) -> (layer, head) interface that PruningEvaluator expects.
Nothing else in prune.py or run_baselines.py needs to change.

Usage in run_baselines.py:
    from ppo_strategy import load_ppo_strategy
    ppo_strategy = load_ppo_strategy(args.rl_agent_ckpt, backbone, probe, val_loader, census, device, args.dataset)
    rl_means, rl_stds = evaluator.run_pruning_strategy(ppo_strategy, census, n_steps=72, n_runs=1)
"""

import torch
import numpy as np
from sb3_contrib import MaskablePPO
from pruning_env import HeadPruningEnv
from config.agents_config import DATASET_CONFIGS


class PPOAgentStrategy:
    """
    Wraps a trained MaskablePPO model so it can be called as a standard
    pruning strategy: strategy(mask, census) -> (layer, head).

    Internally it maintains the Gymnasium env so the agent always gets
    a consistent state — real loss after each prune, not a proxy.
    """

    def __init__(self, model, env):
        """
        Args:
            model: loaded MaskablePPO instance
            env:   HeadPruningEnv instance (same backbone/probe/dataloader as evaluator)
        """
        self.model = model
        self.env   = env
        self.reset()

    def reset(self):
        """Reset the env at the start of each evaluation run."""
        self.obs, _ = self.env.reset()

    def __call__(self, mask, census):
        """
        Pick the next head to prune.

        Args:
            mask:   (12, 12) binary tensor — current pruning state
            census: dict of head profile tensors — not used directly here
                    since the env already has census in its observations

        Returns:
            (layer, head) indices
        """
        # Build the valid action mask:
        # True  = action is allowed (head still active)
        # False = action is blocked (head already pruned)
        # Stop action is DISABLED during evaluation — we want the agent
        # to keep pruning for exactly n_steps so we can plot the curve.
        action_masks = self.env.action_masks()
        action_masks[-1] = False  # disable stop (action 144)

        # Ask the policy for the best action given current obs + valid actions.
        # deterministic=True means greedy — no random sampling during evaluation.
        action, _ = self.model.predict(
            self.obs,
            action_masks = action_masks,
            deterministic= True,
        )

        action = int(action)

        # Safety check — should never happen with stop disabled,
        # but guard against it to avoid out-of-bounds mask access
        if action >= 144:
            unpruned = (mask == 1.0).nonzero(as_tuple=False)
            layer, head = unpruned[0].tolist()
            self.last_action = layer * 12 + head
            return layer, head

        # Decode flat action index to (layer, head)
        layer = action // 12
        head  = action % 12

        # Store action so update_state can step the env
        self.last_action = action

        return layer, head

    def update_state(self):
        """
        Step the internal env with the last action so the agent's next
        observation reflects the real task loss after pruning.
        Called by run_pruning_strategy after each evaluate() call.
        """
        self.obs, _, terminated, truncated, _ = self.env.step(self.last_action)

        # If the env ended for any reason, reset it
        if terminated or truncated:
            self.reset()


def load_ppo_strategy(ckpt_path, backbone, probe, dataloader, census, device, dataset_name):
    """
    Convenience function — loads the model and builds the strategy in one call.

    Args:
        ckpt_path:    path to the .zip file saved by train_agent.py
        backbone:     DinoV2Backbone instance
        probe:        task head (ClassificationHead / ADE20KHead / CocoHead)
        dataloader:   validation dataloader for the task
        census:       dict of head profile tensors
        device:       torch device string
        dataset_name: "imagenet" | "ade20k" | "coco"

    Returns:
        PPOAgentStrategy instance ready to pass to run_pruning_strategy
    """
    # Build the same env the agent was trained in
    env = HeadPruningEnv(
        backbone     = backbone,
        probe        = probe,
        dataloader   = dataloader,
        census       = census,
        device       = device,
        config       = DATASET_CONFIGS,
        dataset_name = dataset_name,
    )

    # Load the trained policy
    model = MaskablePPO.load(ckpt_path, device=device)
    print(f"Loaded RL agent from {ckpt_path}")

    return PPOAgentStrategy(model=model, env=env)