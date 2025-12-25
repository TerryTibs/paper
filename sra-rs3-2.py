#!/usr/bin/env python3
"""
PROTOTYPE MERGED SRA-RS3 AGENT
==============================
Merges key elements from RS3 multi-agent training and SRA neuro-symbolic architecture.
- Multi-agent parallel learning from RS3.
- ConvAE perception, resonance, memory gating, symbol discovery from SRA.
- PPO with curiosity from late SRA.
- Simplified for prototype; scales grid, adds curriculum basics.

Run: python merged_prototype.py
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
from pathlib import Path

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
    "max_grid_size": 12,
    "num_agents": 3,  # From RS3
    "max_episode_steps": 60001,  # RS3 long training

    # Latent/Model Dims (Later RS3/SRA)
    "latent_dim": 128,
    "bottleneck_dim": 32,

    # Hyperparams (Blend)
    "learning_rate": 1e-4,
    "contrastive_weight": 3.0,
    "spectral_weight": 0.8,
    "vision_gate_weight": 60.0,
    "alignment_weight": 40.0,
    "proprioceptive_weight": 35.0,
    "field_coupling": 0.40,  # RS3 autonomy
    "oracle_steps": 5000,
    "base_entropy": 0.4,

    # Resonance (SRA)
    "inference_steps": 8,
    "inference_lr": 0.06,
    "memory_abs_threshold": 0.02,
    "memory_rel_factor": 0.85,
    "memory_capacity": 10000,
    "min_memory_for_clustering": 50,

    # PPO (SRA)
    "ppo_lr": 1e-3,
    "ppo_updates": 100,
    "ppo_epochs": 4,
    "clip_ratio": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "entropy_coef": 0.01,
    "curiosity_beta": 0.2,

    # Symbols (SRA)
    "min_symbols": 5,
    "max_symbols": 20,
    "n_symbols_fallback": 10,
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
        self.starved = False
        return self._get_obs()

    def _place_food(self):
        while True:
            food = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            if food not in self.snake:
                return food

    def _get_obs(self):
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        for i, (x, y) in enumerate(self.snake):
            obs[0, x, y] = 1.0 if i == 0 else 0.5  # head/body
        obs[1, self.food[0], self.food[1]] = 1.0  # food
        return obs

    def step(self, action):
        # Actions: 0 left, 1 up, 2 right, 3 down
        dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        self.direction = dirs[action]

        head = list(self.snake[0])
        head[0] += self.direction[1]  # Note: swapped for grid coords
        head[1] += self.direction[0]
        head = [int(np.clip(head[0], 0, self.grid_size-1)), int(np.clip(head[1], 0, self.grid_size-1))]

        reward = -0.01  # survival cost
        self.done = head in list(self.snake) or self.steps > self.grid_size*10  # starve
        if self.done and self.steps > self.grid_size*10:
            self.starved = True
            reward = -1.0

        self.snake.appendleft(head)
        if head == self.food:
            reward = 1.0
            self.food = self._place_food()
        else:
            self.snake.pop()

        self.steps += 1
        return self._get_obs(), reward, self.done, {}

    def expand(self, increment):
        self.grid_size += increment
        return self.grid_size

# ==========================
# CONV AUTOENCODER (SRA Perception)
# ==========================
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (CONFIG["start_grid_size"]//4)**2, CONFIG["latent_dim"])
        )
        self.decoder = nn.Sequential(
            nn.Linear(CONFIG["latent_dim"], 64 * (CONFIG["start_grid_size"]//4)**2),
            nn.ReLU(),
            nn.Unflatten(1, (64, CONFIG["start_grid_size"]//4, CONFIG["start_grid_size"]//4)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

# ==========================
# RESONATOR (SRA Inference)
# ==========================
class Resonator:
    def __init__(self, ae):
        self.ae = ae

    def resonate(self, obs_tensor, prev_z=None):
        # FIX: Force gradients enabled so optimization works even inside torch.no_grad()
        with torch.enable_grad():
            z = torch.randn(1, CONFIG["latent_dim"], requires_grad=True, device=device)
            if prev_z is not None:
                z = prev_z.detach().clone().requires_grad_(True)
            opt = optim.Adam([z], lr=CONFIG["inference_lr"])

            for _ in range(CONFIG["inference_steps"]):
                recon = self.ae.decoder(z)
                loss = F.mse_loss(recon, obs_tensor)
                opt.zero_grad()
                loss.backward()
                opt.step()

        return z.detach(), loss.item()

# ==========================
# MEMORY (SRA Gated + RS3 Telemetry)
# ==========================
class Memory:
    def __init__(self):
        self.vectors = deque(maxlen=CONFIG["memory_capacity"])
        self.coherences = deque(maxlen=CONFIG["memory_capacity"])

    def add(self, z, coherence, z0_loss):
        if coherence < CONFIG["memory_abs_threshold"] or coherence < CONFIG["memory_rel_factor"] * z0_loss:
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
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, 4)  # 4 actions
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.actor(h), self.critic(h)

# ==========================
# SYMBOL DISCOVERY (SRA)
# ==========================
def discover_symbols(mem_arr, min_k, max_k):
    if hdbscan:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        labels = clusterer.fit_predict(mem_arr)
    else:
        k = min(max_k, max(min_k, len(mem_arr)//10))
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(mem_arr)
        centers = kmeans.cluster_centers_
    unique_labels = np.unique(labels[labels >= 0])
    centers = [np.mean(mem_arr[labels == l], axis=0) for l in unique_labels]
    return np.vstack(centers), labels, len(unique_labels)

# ==========================
# MAIN (Phased: SRA + RS3 Multi-Agent)
# ==========================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    OUTDIR = Path(CONFIG["output_dir"])

    # Phase 0: Initialize Envs, Agents, Memories (RS3 Multi + SRA)
    grid_size = CONFIG["start_grid_size"]
    envs = [SnakeEnv(grid_size) for _ in range(CONFIG["num_agents"])]
    ae = ConvAE().to(device)
    resonators = [Resonator(ae) for _ in range(CONFIG["num_agents"])]
    memories = [Memory() for _ in range(CONFIG["num_agents"])]
    ac = ActorCritic(CONFIG["latent_dim"]).to(device)  # Shared policy
    opt_ae = optim.Adam(ae.parameters(), lr=CONFIG["learning_rate"])
    opt_ac = optim.Adam(ac.parameters(), lr=CONFIG["ppo_lr"])

    # Pretrain AE (SRA Birth)
    print("Phase 0: Pretraining AE")
    dataset = []
    for _ in range(1000):  # Collect random states
        for env in envs:
            obs = env.reset()
            dataset.append(obs)
    dataset = torch.tensor(np.array(dataset)).to(device)
    for epoch in range(10):
        for batch in torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True):
            _, recon = ae(batch)
            loss = F.mse_loss(recon, batch)
            opt_ae.zero_grad(); loss.backward(); opt_ae.step()
    print("AE Pretrained.")

    # Phase 1: Exploration/Childhood (Resonance + Memory, Multi-Agent)
    print("Phase 1: Exploration")
    for ep in range(50):
        for i in range(CONFIG["num_agents"]):
            env = envs[i]
            resonator = resonators[i]
            memory = memories[i]
            obs = env.reset()
            done = False
            prev_z = None
            while not done:
                obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
                z_opt, loss = resonator.resonate(obs_tensor, prev_z)
                # Random action for exploration
                action = random.randint(0, 3)
                next_obs, reward, done, _ = env.step(action)
                # Gate memory
                with torch.no_grad():
                    recon = ae(obs_tensor)[1]
                    z0_loss = F.mse_loss(recon, obs_tensor).item()
                memory.add(z_opt.squeeze(0), loss, z0_loss)
                prev_z = z_opt
                obs = next_obs
        if ep % 10 == 0:
            print(f"Ep {ep}: Memories {[len(m.vectors) for m in memories]}")

    # Phase 2: Symbol Discovery (SRA)
    print("Phase 2: Symbol Discovery")
    all_mem = np.vstack([m.to_array() for m in memories if len(m.vectors) > 0])
    if len(all_mem) >= CONFIG["min_memory_for_clustering"]:
        centers, labels, k = discover_symbols(all_mem, CONFIG["min_symbols"], CONFIG["max_symbols"])
        print(f"Discovered {k} symbols.")
        np.save(OUTDIR / "symbols.npy", centers)
    else:
        print("Not enough memories for symbols.")
        return

    # Phase 3: PPO Training (SRA + RS3 Multi-Agent + Curiosity)
    print("Phase 3: PPO Training")
    forward_model = nn.Sequential(  # Curiosity
        nn.Linear(CONFIG["latent_dim"] + 4, 256), nn.ReLU(),  # state + action
        nn.Linear(256, CONFIG["latent_dim"])
    ).to(device)
    opt_fwd = optim.Adam(forward_model.parameters(), lr=CONFIG["ppo_lr"])

    rewards_history = []
    for update in range(CONFIG["ppo_updates"]):
        batch_obs = []
        batch_acts = []
        batch_old_logprobs = []
        batch_rews = []
        batch_adv = []
        batch_rets = []
        for i in range(CONFIG["num_agents"]):
            env = envs[i]
            resonator = resonators[i]
            obs = env.reset()
            done = False
            ep_obs = []
            ep_acts = []
            ep_logprobs = []
            ep_rews = []
            ep_vals = []
            ep_next_obs = []
            with torch.no_grad():
                while not done:
                    obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
                    # resonator now handles enable_grad internally
                    z, _ = resonator.resonate(obs_tensor)
                    logits, value = ac(z)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
                    logprob = dist.log_prob(torch.tensor(action, device=device)).detach()
                    next_obs, rew, done, _ = env.step(action)

                    ep_obs.append(z.detach())
                    ep_acts.append(action)
                    ep_logprobs.append(logprob)
                    ep_rews.append(rew)
                    ep_vals.append(value.detach().item())
                    ep_next_obs.append(next_obs)
                    obs = next_obs

            # Compute GAE and returns per episode
            ep_adv = []
            gae = 0
            ep_returns = []
            with torch.no_grad():
                last_obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
                last_z = ae.encoder(last_obs_tensor)
                next_value = 0 if done else ac(last_z)[1].item()
            for t in reversed(range(len(ep_rews))):
                delta = ep_rews[t] + CONFIG["gamma"] * next_value - ep_vals[t]
                gae = delta + CONFIG["gamma"] * CONFIG["gae_lambda"] * gae
                ep_adv.insert(0, gae)
                ret = gae + ep_vals[t]
                ep_returns.insert(0, ret)
                next_value = ep_vals[t]

            # Curiosity intrinsic
            for t in range(len(ep_rews)):
                act_oh = F.one_hot(torch.tensor(ep_acts[t]), num_classes=4).float().to(device)
                z_t = ep_obs[t].squeeze(0)
                z_next_pred = forward_model(torch.cat([z_t, act_oh]))
                next_obs_tensor = torch.tensor(ep_next_obs[t]).unsqueeze(0).to(device)
                z_next_true = ae.encoder(next_obs_tensor).squeeze(0)
                int_rew = CONFIG["curiosity_beta"] * F.mse_loss(z_next_pred, z_next_true.detach())
                ep_rews[t] += int_rew.item()
                opt_fwd.zero_grad(); int_rew.backward(); opt_fwd.step()
                # Update adv and returns with int_rew? For demo, skip recompute

            batch_obs.extend(ep_obs)
            batch_acts.extend(ep_acts)
            batch_old_logprobs.extend(ep_logprobs)
            batch_rews.extend(ep_rews)
            batch_adv.extend(ep_adv)
            batch_rets.extend(ep_returns)

        # PPO Update
        # FIX: cat instead of stack to ensure shape is (N, 128) not (N, 1, 128)
        batch_obs = torch.cat(batch_obs, dim=0) 
        batch_acts = torch.tensor(batch_acts, device=device)
        batch_old_logprobs = torch.stack(batch_old_logprobs).detach()
        batch_adv = torch.tensor(batch_adv, device=device)
        batch_rets = torch.tensor(batch_rets, device=device)

        for _ in range(CONFIG["ppo_epochs"]):
            logits, vals = ac(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(batch_acts)
            ratio = torch.exp(new_logprobs - batch_old_logprobs)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - CONFIG["clip_ratio"], 1 + CONFIG["clip_ratio"]) * batch_adv
            loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(vals.squeeze(), batch_rets)
            opt_ac.zero_grad(); loss.backward(); opt_ac.step()

        rewards_history.append(sum(batch_rews) / CONFIG["num_agents"])
        if update % 10 == 0:
            print(f"Update {update}: Avg Reward {np.mean(rewards_history[-10:]):.2f}")

    # Save
    torch.save(ae.state_dict(), OUTDIR / "ae.pth")
    torch.save(ac.state_dict(), OUTDIR / "ac.pth")
    plt.plot(rewards_history)
    plt.savefig(OUTDIR / "curve.png")
    print("Prototype Run Complete.")

if __name__ == "__main__":
    main()
