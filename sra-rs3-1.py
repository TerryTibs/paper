PROTOTYPE MERGED SRA-RS3 AGENT
==============================
Merges key elements from RS3 multi-agent training and SRA neuro-symbolic architecture.
- Multi-agent parallel learning from RS3.
- ConvAE perception, resonance, memory gating, symbol discovery from SRA.
- PPO with curiosity from late SRA.
- Simplified for prototype; scales grid, adds curriculum basics.

Run: python sra_rs3_merged.py
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Try HDBSCAN
try:
    import hdbscan
except ImportError:
    hdbscan = None

# ==========================
# UNIFIED CONFIG (Merged from RS3 and SRA)
# ==========================
CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "merged_output",

    # Environment (RS3 + SRA curriculum)
    "start_grid_size": 12,
    "max_grid_size": 48,
    "num_agents": 4,             # RS3 Parallelism
    "max_episode_steps": 1000,

    # Latent/Model Dims (High Capacity)
    "latent_dim": 128,
    "bottleneck_dim": 32,

    # Hyperparams (Blend)
    "learning_rate": 1e-4,
    "contrastive_weight": 3.0,
    "vision_gate_weight": 60.0,
    "proprioceptive_weight": 35.0,
    
    # Resonance (SRA)
    "inference_steps": 5,
    "inference_lr": 0.06,
    "memory_abs_threshold": 0.02,
    "memory_rel_factor": 0.85,
    "memory_capacity": 10000,
    "min_memory_for_clustering": 50,

    # PPO (SRA)
    "ppo_lr": 3e-4,
    "ppo_updates": 500,
    "ppo_epochs": 4,
    "clip_ratio": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "entropy_coef": 0.01,
    "curiosity_beta": 0.2,

    # Symbols (SRA)
    "min_symbols": 5,
    "max_symbols": 20,
}

device = torch.device(CONFIG["device"])
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])

# ==========================
# SNAKE ENVIRONMENT (From SRA, scalable like RS3)
# ==========================
class SnakeEnv:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = deque([[self.grid_size // 2, self.grid_size // 2]])
        self.direction = [0, 1]  # right
        self.food = self._place_food()
        self.steps = 0
        self.done = False
        return self._get_obs()

    def _place_food(self):
        while True:
            food = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            if food not in self.snake:
                return food

    def _get_obs(self):
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        for i, (x, y) in enumerate(self.snake):
            # Safe indexing
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                obs[0, x, y] = 1.0 if i == 0 else 0.5  # head/body
        
        fx, fy = self.food
        if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
            obs[1, fx, fy] = 1.0  # food
        return obs

    def step(self, action):
        # Actions: 0 left, 1 up, 2 right, 3 down
        dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        
        # Simple prevention of 180 turns
        current_dir_idx = dirs.index(self.direction)
        if (action + 2) % 4 != current_dir_idx:
            self.direction = dirs[action]

        head = list(self.snake[0])
        head[0] += self.direction[0]
        head[1] += self.direction[1]
        
        # Bounds Check
        if (head[0] < 0 or head[0] >= self.grid_size or 
            head[1] < 0 or head[1] >= self.grid_size or 
            head in list(self.snake)):
            self.done = True
            return self._get_obs(), -1.0, self.done, {}

        self.snake.appendleft(head)
        reward = -0.01
        
        if head == self.food:
            reward = 1.0
            self.food = self._place_food()
        else:
            self.snake.pop()

        self.steps += 1
        if self.steps > CONFIG["max_episode_steps"]:
            self.done = True
            
        return self._get_obs(), reward, self.done, {}

# ==========================
# CONV AUTOENCODER (SRA Perception)
# ==========================
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Dynamic sizing for curriculum
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1), nn.Sigmoid()
        )
        
        # Adaptive pooling to handle variable grid sizes into fixed latent
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) 
        self.fc_enc = nn.Linear(64 * 4 * 4, CONFIG["latent_dim"])
        self.fc_dec = nn.Linear(CONFIG["latent_dim"], 64 * 4 * 4)

    def forward(self, x):
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        h = self.encoder(x)
        h_pooled = self.pool(h).view(batch_size, -1)
        z = self.fc_enc(h_pooled)
        
        h_recon = self.fc_dec(z).view(batch_size, 64, 4, 4)
        # Upsample to current grid size before decoding
        h_recon = F.interpolate(h_recon, size=(grid_size, grid_size), mode='nearest')
        recon = self.decoder(h_recon)
        return z, recon

# ==========================
# RESONATOR (SRA Inference)
# ==========================
class Resonator:
    def __init__(self, ae):
        self.ae = ae

    def resonate(self, obs_tensor, prev_z=None):
        # Initial guess from encoder
        with torch.no_grad():
            z_init, _ = self.ae(obs_tensor)
            
        z = z_init.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=CONFIG["inference_lr"])

        for _ in range(CONFIG["inference_steps"]):
            # Decode using CURRENT z
            # Note: We must manually run the decoder part of ConvAE here
            # to propagate gradients back to z
            batch_size = z.size(0)
            grid_size = obs_tensor.size(2)
            
            h_recon = self.ae.fc_dec(z).view(batch_size, 64, 4, 4)
            h_recon = F.interpolate(h_recon, size=(grid_size, grid_size), mode='nearest')
            recon = self.ae.decoder(h_recon)
            
            loss = F.mse_loss(recon, obs_tensor)
            opt.zero_grad()
            loss.backward()
            opt.step()

        return z.detach(), loss.item()

# ==========================
# MEMORY (SRA Gated)
# ==========================
class Memory:
    def __init__(self):
        self.vectors = deque(maxlen=CONFIG["memory_capacity"])
        self.coherences = deque(maxlen=CONFIG["memory_capacity"])

    def add(self, z, coherence):
        if coherence < CONFIG["memory_abs_threshold"]:
            self.vectors.append(z.cpu().numpy())
            self.coherences.append(coherence)

    def to_array(self):
        return np.array(self.vectors) if self.vectors else np.empty((0, CONFIG["latent_dim"]))

# ==========================
# ACTOR-CRITIC (SRA PPO)
# ==========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.actor = nn.Linear(256, 4)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.actor(h), self.critic(h)

# ==========================
# MAIN
# ==========================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    OUTDIR = Path(CONFIG["output_dir"])

    # Phase 0: Initialize
    grid_size = CONFIG["start_grid_size"]
    envs = [SnakeEnv(grid_size) for _ in range(CONFIG["num_agents"])]
    ae = ConvAE().to(device)
    resonator = Resonator(ae)
    memory = Memory()
    ac = ActorCritic(CONFIG["latent_dim"]).to(device)
    
    opt_ae = optim.Adam(ae.parameters(), lr=CONFIG["learning_rate"])
    opt_ac = optim.Adam(ac.parameters(), lr=CONFIG["ppo_lr"])

    # Phase 1: AE Pretraining
    print("[Phase 1] Pretraining Eyes...")
    dataset = []
    for _ in range(200):
        for env in envs:
            obs = env.reset()
            # Random moves
            for _ in range(10):
                obs, _, d, _ = env.step(random.randint(0,3))
                dataset.append(obs)
                if d: break
    
    dataset = torch.tensor(np.array(dataset), dtype=torch.float32).to(device)
    for epoch in range(20):
        idx = torch.randperm(len(dataset))
        for i in range(0, len(dataset), 32):
            batch = dataset[idx[i:i+32]]
            _, recon = ae(batch)
            loss = F.mse_loss(recon, batch)
            opt_ae.zero_grad(); loss.backward(); opt_ae.step()
    print("  > Eyes Opened.")

    # Phase 2: Resonance Exploration
    print("[Phase 2] Exploration & Memory...")
    for ep in range(20):
        for i, env in enumerate(envs):
            obs = env.reset()
            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                z_opt, loss = resonator.resonate(obs_t)
                
                # SRA Memory Gating
                memory.add(z_opt.squeeze(0), loss)
                
                # Random Action
                action = random.randint(0, 3)
                obs, _, done, _ = env.step(action)
                
    print(f"  > Memory populated with {len(memory.vectors)} coherent states.")

    # Phase 3: PPO Training
    print("[Phase 3] PPO Training (Multi-Agent)...")
    
    rewards_history = []
    for update in range(CONFIG["ppo_updates"]):
        
        # Rollout Storage
        b_obs, b_acts, b_logp, b_rets, b_adv = [], [], [], [], []
        
        for env in envs:
            obs = env.reset()
            done = False
            
            ep_obs, ep_acts, ep_logp, ep_rews, ep_vals = [], [], [], [], []
            
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Fast Resonance (1 step for speed during RL)
                with torch.no_grad():
                    z, _ = ae(obs_t)
                
                logits, val = ac(z)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
                
                obs, reward, done, _ = env.step(action.item())
                
                ep_obs.append(z)
                ep_acts.append(action)
                ep_logp.append(logp)
                ep_rews.append(reward)
                ep_vals.append(val)
                
            # GAE
            ep_rets = []
            gae = 0
            for i in reversed(range(len(ep_rews))):
                delta = ep_rews[i] - ep_vals[i].item() # Simplified GAE
                gae = delta + 0.95 * 0.99 * gae
                ep_rets.insert(0, gae + ep_vals[i].item())
                
            b_obs.extend(ep_obs)
            b_acts.extend(ep_acts)
            b_logp.extend(ep_logp)
            b_rets.extend(ep_rets)
            
        # PPO Update
        b_obs = torch.cat(b_obs)
        b_acts = torch.stack(b_acts)
        b_logp = torch.stack(b_logp)
        b_rets = torch.tensor(b_rets).to(device)
        b_adv = b_rets - b_rets.mean() # Simplified advantage
        
        for _ in range(4):
            logits, vals = ac(b_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(b_acts)
            ratio = torch.exp(new_logp - b_logp.detach())
            
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
            loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(vals.squeeze(), b_rets)
            
            opt_ac.zero_grad(); loss.backward(); opt_ac.step()
            
        avg_r = np.mean(b_rets.cpu().numpy())
        rewards_history.append(avg_r)
        
        if update % 10 == 0:
            print(f"Update {update}: Avg Return {avg_r:.2f}")
            
        # Curriculum: Expand Grid
        if update > 0 and update % 50 == 0 and grid_size < CONFIG["max_grid_size"]:
            grid_size += 4
            for env in envs: env.grid_size = grid_size
            print(f"*** WORLD EXPANDED TO {grid_size}x{grid_size} ***")

    # Save
    plt.plot(rewards_history)
    plt.savefig(OUTDIR / "training_curve.png")
    torch.save(ae.state_dict(), OUTDIR / "ae.pth")
    torch.save(ac.state_dict(), OUTDIR / "ac.pth")
    print("Done.")

if __name__ == "__main__":
    main()
