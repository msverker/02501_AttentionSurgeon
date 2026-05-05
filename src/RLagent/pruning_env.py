"""
pruning_env.py — Gymnasium environment for RL-based attention head pruning.

Wraps the DINOv2 backbone + task probe into a standard Gymnasium interface
so any SB3 algorithm (MaskablePPO, SAC, etc.) can train on it without changes.

State space (578-dim):
  [0:144]   binary mask        — which heads are still active
  [144:288] importance scores  — precomputed per-head task sensitivity
  [288:432] activation mag     — precomputed per-head activity level
  [432:576] entropy scores     — precomputed per-head attention focus
  [576]     current loss       — task loss with current pruning mask applied
  [577]     sparsity           — fraction of heads pruned so far

Action space:
  Discrete(145) — actions 0-143 prune head (layer=a//12, head=a%12), action 144 = STOP
"""

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
from gymnasium import spaces


class HeadPruningEnv(gymnasium.Env):

    def __init__(self, backbone, probe, dataloader, census, device, config, dataset_name="imagenet"):
        super().__init__()

        # ── Core components ──────────────────────────────────────────────
        self.backbone = backbone
        self.probe = probe
        self.dataloader = dataloader
        self.device = device
        self.dataset_name = dataset_name

        # ── Reward hyperparameters from config ───────────────────────────
        cfg = config[dataset_name]
        self.alpha   = cfg["alpha"]    # weight on retention loss (performance)
        self.beta    = cfg["beta"]     # weight on sparsity (efficiency)
        self.gamma   = cfg["gamma"]    # weight on penalty (catastrophic drops)
        self.epsilon = cfg["epsilon"]  # tolerance before penalty kicks in

        # ── Census: fixed per-head features precomputed on the full model ─
        # Shape of each: (12, 12) → flatten to (144,) for the state vector.
        # These never change during an episode — they describe what each head
        # does on average, giving the agent context for its pruning decisions.
        self.census_importance = census["importance"].float().flatten().cpu()   # task sensitivity
        self.census_magnitude  = census["activation_mag"].float().flatten().cpu()  # head activity
        self.census_entropy    = census["entropy"].float().flatten().cpu()      # attention focus

        # ── Spaces ───────────────────────────────────────────────────────
        # Observation: mask(144) + importance(144) + magnitude(144) + entropy(144) + loss(1) + sparsity(1)
        obs_dim = 144 + 144 + 144 + 144 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Actions: 144 heads to prune + 1 stop action
        self.action_space = spaces.Discrete(145)

        # Internal state — initialise mask to all-ones BEFORE computing
        # baseline loss, since _compute_loss uses hooks that read self.mask.
        self.mask         = torch.ones(12, 12, device=self.device)
        self.step_count   = 0
        self.sparsity     = 0.0
        self.current_loss = 0.0

        # ── Compute baseline loss once (unpruned model) ──────────────────
        print("Computing baseline loss...")
        self.baseline_loss = self._compute_loss(num_batches=4)
        print(f"Baseline loss: {self.baseline_loss:.4f}")

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Called at the start of every episode.
        Resets mask to all-ones (no heads pruned) and recomputes clean loss.
        Returns (observation, info_dict).
        """
        super().reset(seed=seed)

        # Reset pruning state — all heads active
        self.mask       = torch.ones(12, 12, device=self.device)
        self.step_count = 0
        self.sparsity   = 0.0

        # Recompute loss with clean mask (slight variance from baseline due to batch sampling)
        self.current_loss = self._compute_loss(num_batches=4)

        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one pruning action.

        Args:
            action: int 0–143 (prune a specific head) or 144 (stop)

        Returns:
            obs, reward, terminated, truncated, info
        """
        # ── STOP action ──────────────────────────────────────────────────
        if action == 144:
            reward = self._compute_reward()
            return self._get_obs(), reward, True, False, {}

        # ── Decode head index ─────────────────────────────────────────────
        layer = action // 12
        head  = action % 12

        # Penalise trying to prune an already-pruned head
        # (MaskablePPO should prevent this, but just in case)
        if self.mask[layer, head] == 0.0:
            return self._get_obs(), -1.0, False, False, {}

        # ── Prune the head ────────────────────────────────────────────────
        self.mask[layer, head] = 0.0
        self.step_count += 1
        self.sparsity = self.step_count / 144

        # Evaluate task loss with updated mask
        self.current_loss = self._compute_loss(num_batches=4)

        reward     = self._compute_reward()
        terminated = self.step_count >= 144  # pruned everything — force end

        return self._get_obs(), reward, terminated, False, {}

    def action_masks(self):
        """
        Called by MaskablePPO every step to know which actions are legal.
        Returns bool array of shape (145,): True = allowed, False = blocked.

        Unpruned heads + stop are always valid.
        Already-pruned heads are masked out so the agent can't re-prune them.
        """
        head_valid = self.mask.flatten().bool().cpu().numpy()  # (144,) — 1 if still active
        stop_valid = np.array([True])                          # stop always allowed
        return np.concatenate([head_valid, stop_valid])

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_obs(self):
        """
        Build the state vector the agent sees.

        Concatenates:
          - Current mask (which heads are alive)
          - Census features (what each head does — fixed throughout episode)
          - Current task loss and sparsity (dynamic signals)
        """
        mask_flat = self.mask.flatten().float().cpu().numpy()

        obs = np.concatenate([
            mask_flat,                                                    # [0:144]   active heads
            self.census_importance.numpy(),                               # [144:288] task sensitivity
            self.census_magnitude.numpy(),                                # [288:432] head activity
            self.census_entropy.numpy(),                                  # [432:576] attention focus
            np.array([self.current_loss, self.sparsity], dtype=np.float32)  # [576:578] dynamics
        ]).astype(np.float32)

        return obs

    def _compute_reward(self):
        """
        R = α·ΔL + β·S − γ·P²

        ΔL  retention loss  — positive when pruning didn't hurt performance
        S   sparsity        — fraction of heads removed (efficiency signal)
        P   penalty         — excess loss beyond tolerance ε (catastrophic drop guard)
        """
        retention = (self.baseline_loss - self.current_loss) / (self.baseline_loss + 1e-8)
        excess    = max(0.0, (self.current_loss - self.baseline_loss) - self.epsilon)
        reward    = self.alpha * retention + self.beta * self.sparsity - self.gamma * excess**2
        return float(reward)

    def _compute_loss(self, num_batches=4):
        """
        Forward pass through backbone + probe with the current pruning mask applied.
        Uses forward hooks to zero out pruned heads without modifying model weights.
        """
        self.backbone.eval()
        self.probe.eval()

        hooks    = self._register_pruning_hooks()
        total    = 0
        loss_sum = 0.0
        device   = next(self.probe.parameters()).device

        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= num_batches:
                    break

                if self.dataset_name == "ade20k":
                    imgs, labels = batch["image"].to(device), batch["mask"].to(device)
                    feats  = self.backbone(imgs)
                    logits = self.probe(feats)
                    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    if labels.dim() == 4:
                        labels = labels.squeeze(1)
                    loss = F.cross_entropy(logits, labels.long(), ignore_index=-1)

                elif self.dataset_name == "coco":
                    from baseline.backbone import build_targets
                    images  = batch[0].to(device)
                    targets = [batch[1]] if isinstance(batch[1], dict) else batch[1]
                    feats   = self.backbone(images)
                    cls_logits, box_preds, obj_logits = self.probe(feats)
                    grid_size = cls_logits.shape[-1]
                    obj_t, cls_t, box_t = build_targets(targets, grid_size=grid_size, device=device)
                    loss = (
                        F.binary_cross_entropy_with_logits(obj_logits, obj_t)
                        + F.cross_entropy(cls_logits.permute(0,2,3,1).reshape(-1,80), cls_t.view(-1))
                        + F.l1_loss(box_preds, box_t)
                    )

                else:  # imagenet / cifar
                    imgs, labels = batch
                    imgs, labels = imgs.to(device), labels.to(device)
                    feats  = self.backbone(imgs)
                    logits = self.probe(feats)
                    loss   = F.cross_entropy(logits, labels)

                loss_sum += loss.item()
                total    += 1

        for h in hooks:
            h.remove()

        return loss_sum / total if total > 0 else 0.0

    def _register_pruning_hooks(self):
        """
        Attach a forward hook to each transformer layer that zeroes out
        the output channels belonging to pruned heads.

        This lets us simulate pruning without actually modifying model weights —
        we just multiply the head outputs by 0 or 1 according to self.mask.
        """
        hooks    = []
        head_dim = 768 // 12  # 64 dims per head for ViT-B

        def make_hook(layer_idx, mask_row):
            def hook(module, input, output):
                # output[0] is (B, SeqLen, 768) — the attention layer output
                ctx = output[0] if isinstance(output, tuple) else output
                B, S, _ = ctx.shape
                # Reshape to separate heads: (B, S, 12, 64)
                ctx = ctx.view(B, S, 12, head_dim)
                # Zero out pruned heads using the mask row for this layer
                ctx = ctx * mask_row.to(ctx.device).view(1, 1, 12, 1)
                # Reshape back to (B, S, 768)
                ctx = ctx.view(B, S, 768)
                return (ctx,) + output[1:] if isinstance(output, tuple) else ctx
            return hook

        for i, layer in enumerate(self.backbone.model.encoder.layer):
            h = layer.attention.attention.register_forward_hook(make_hook(i, self.mask[i]))
            hooks.append(h)

        return hooks