import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from baseline.loaders import get_imagenet_loaders  # Replace with your actual data loader import
from baseline.backbone import DinoV2Backbone  # Replace with your actual DINOv2 backbone import
import numpy as np

# ==========================================
# 1. MOCK COMPONENTS (Replace with yours)
# ==========================================
class ClassificationHead(nn.Module):
    def __init__(self, in_dim=768, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

# ==========================================
# 2. RL ENVIRONMENT (Unchanged)
# ==========================================
class TransformerPruningEnv:
    def __init__(self, backbone, probe, dataloader, device):
        self.backbone = backbone
        self.probe = probe
        self.dataloader = dataloader
        self.device = device
        self.num_layers = 12
        self.num_heads = 12
        self.total_heads = self.num_layers * self.num_heads
        
        self.reset()

    def reset(self):
        self.mask = torch.ones(self.num_layers, self.num_heads).to(self.device)
        self.current_acc = self._get_current_accuracy()
        self.current_flops = 1.0
        return self._get_state()

    def _get_state(self):
        return torch.cat([self.mask.flatten(), 
                          torch.tensor([self.current_acc, self.current_flops]).to(self.device)])

    def _get_current_accuracy(self, num_batches=2):
        self.backbone.eval()
        self.probe.eval()
        correct, total = 0, 0
        device = next(self.probe.parameters()).device
        
        hooks = []
        head_dim = 768 // self.num_heads

        def make_hook(layer_idx, mask_row):
            def hook(module, input, output):
                ctx = output[0] if isinstance(output, tuple) else output
                B, S, _ = ctx.shape
                ctx = ctx.view(B, S, 12, head_dim)
                ctx = ctx * mask_row.to(ctx.device).view(1, 1, 12, 1)
                ctx = ctx.view(B, S, 768)
                if isinstance(output, tuple):
                    return (ctx,) + output[1:]
                return ctx
            return hook

        for i, layer in enumerate(self.backbone.model.encoder.layer):
            h = layer.attention.attention.register_forward_hook(make_hook(i, self.mask[i]))
            hooks.append(h)

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(self.dataloader):
                if i >= num_batches: 
                    break
                imgs, labels = imgs.to(device), labels.to(device)
                feats = self.backbone(imgs)["cls_token"] 
                logits = self.probe(feats)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)

        for h in hooks:
            h.remove()
            
        return correct / total if total > 0 else 0.0

    def step(self, action_idx):
        layer = action_idx // self.num_heads
        head = action_idx % self.num_heads
        
        if self.mask[layer, head] == 0.0:
            return self._get_state(), -1.0, False
        
        self.mask[layer, head] = 0.0
        self.current_flops -= (1.0 / self.total_heads)
        self.current_acc = self._get_current_accuracy()
        
        reward = self.current_acc * (1.0 - self.current_flops)
        done = self.current_flops <= 0.1 or self.current_acc < 0.5
        
        return self._get_state(), reward, done


# ==========================================
# 3. PPO AGENT & TRAINING LOOP
# ==========================================
class PPOActorCritic(nn.Module):
    def __init__(self, input_dim=146, action_dim=144):
        super().__init__()
        
        # Actor: Decides which action to take (Outputs Logits)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        
        # Critic: Estimates how good the current state is (Outputs Value)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, valid_mask):
        """Returns action, log probability, and estimated state value"""
        action_logits = self.actor(state)
        
        # Action Masking: Set invalid actions to a massive negative number
        # so their Softmax probability becomes essentially 0.
        action_logits[~valid_mask] = -1e8 
        
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action.item(), action_logprob, state_val.squeeze()
        
    def evaluate(self, states, actions, valid_masks):
        """Used during the PPO update step to re-evaluate collected trajectories"""
        action_logits = self.actor(states)
        action_logits[~valid_masks] = -1e8
        
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        
        return action_logprobs, state_values.squeeze(), dist_entropy


def train_ppo_agent(env, episodes=50):
    device = env.device
    policy = PPOActorCritic(input_dim=env.total_heads + 2, action_dim=env.total_heads).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # PPO Hyperparameters
    gamma = 0.99            # Discount factor
    eps_clip = 0.2          # PPO clip parameter
    K_epochs = 4            # Number of times to update the network per episode
    
    print(f"Starting PPO Training for {episodes} episodes...")
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Memory buffers for one episode rollout
        buffer_states, buffer_actions, buffer_logprobs = [], [], []
        buffer_rewards, buffer_values, buffer_masks = [], [], []
        
        # --- PHASE 1: DATA COLLECTION (ROLLOUT) ---
        while not done:
            # Get valid actions mask
            valid_mask = (env.mask.flatten() == 1.0)
            
            with torch.no_grad():
                action, logprob, state_val = policy.act(state, valid_mask)
            
            next_state, reward, done = env.step(action)

            # --- ADD THIS PRINT BLOCK HERE ---
            layer = action // env.num_heads
            head = action % env.num_heads
            print(f"  -> Step {steps + 1:03d} | Pruned L{layer:02d} H{head:02d} | "
                  f"Acc: {env.current_acc:.4f} | FLOPs: {env.current_flops*100:.1f}% | "
                  f"Reward: {reward:.4f}")
            # ---------------------------------
            
            # Store data
            buffer_states.append(state)
            buffer_actions.append(torch.tensor(action, dtype=torch.long, device=device))
            buffer_logprobs.append(logprob)
            buffer_rewards.append(reward)
            buffer_values.append(state_val)
            buffer_masks.append(valid_mask)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps > env.total_heads * 2: break
            
        # --- PHASE 2: PPO UPDATE ---
        # 1. Calculate Discounted Rewards (Returns)
        returns = []
        discounted_reward = 0
        for reward in reversed(buffer_rewards):
            discounted_reward = reward + (gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7) # Normalize
        
        # 2. Convert lists to tensors for batch updating
        old_states = torch.stack(buffer_states).detach()
        old_actions = torch.stack(buffer_actions).detach()
        old_logprobs = torch.stack(buffer_logprobs).detach()
        old_valid_masks = torch.stack(buffer_masks).detach()
        old_values = torch.stack(buffer_values).detach()
        
        # Calculate Advantages
        advantages = returns.detach() - old_values.detach()

        # 3. Optimize policy for K epochs
        for _ in range(K_epochs):
            # Re-evaluate the stored trajectory with the newly updated weights
            logprobs, state_values, dist_entropy = policy.evaluate(old_states, old_actions, old_valid_masks)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            
            # Actor Loss (Maximize surrogate) + Critic Loss (MSE) + Entropy Bonus (Exploration)
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Episode {ep + 1}/{episodes} | Steps: {steps} | "
              f"Final FLOPs: {env.current_flops*100:.1f}% | "
              f"Final Acc: {env.current_acc:.2f} | "
              f"Total Reward: {total_reward:.2f}")
              
    return policy


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    backbone = DinoV2Backbone().to(device)
    probe = ClassificationHead(in_dim=768, num_classes=1000).to(device)
    probe.load_state_dict(torch.load("checkpoints/probe_imagenet.pt", map_location=device))
    _, dataloader = get_imagenet_loaders()

    env = TransformerPruningEnv(
        backbone=backbone, 
        probe=probe, 
        dataloader=dataloader, 
        device=device
    )

    trained_policy = train_ppo_agent(env, episodes=500)
    print("Training complete.")
    
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(trained_policy.state_dict(), "checkpoints/rl_agent_ppo.pt")
    print("Agent saved to checkpoints/rl_agent_ppo.pt")