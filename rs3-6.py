import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import sys
import os
from collections import deque

# =============================================================================
# 1. SYSTEM HYPERPARAMETERS
# =============================================================================
CONFIG = {
    "grid_size": 12,
    "num_agents": 6,
    "latent_dim": 32,
    "bottleneck_dim": 8,
    "total_steps": 10001,
    "learning_rate": 8e-5,
    "contrastive_weight": 2.5,     # Weight for symmetric representation consistency
    "spectral_weight": 0.6,        # Weight for weight-rank regularization (SVD entropy)
    "vision_gate_weight": 55.0,    # Weight for target-noise discrimination
    "alignment_weight": 35.0,      # Weight for policy-coordinate alignment
    "proprioceptive_weight": 30.0, # Weight for ego-centric awareness
    "field_coupling": 0.85,        # Coefficient for shared global manifold
    "oracle_steps": 2000,          # Steps for supervised coordinate grounding
    "base_entropy": 0.4,           # Coefficient for stochastic exploration
    "model_path": "geometric_standard_model.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
device = torch.device(CONFIG["device"])

# =============================================================================
# 2. PERFORMANCE MONITORING SUITE
# =============================================================================
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.snr_history = deque(maxlen=100)

    def sparkline(self, data, length=12):
        if len(data) < 2: return " " * length
        chars = " ▂▃▄▅▆▇█"
        min_v, max_v = min(data), max(data)
        if max_v == min_v: return "─" * length
        res = ""
        for i in range(length):
            idx = int((i / length) * len(data))
            char_idx = int((data[idx] - min_v) / (max_v - min_v + 1e-9) * (len(chars) - 1))
            res += chars[char_idx]
        return res

    def log_status(self, step, succ, mort, envs, agents, telemetry):
        elapsed = time.time() - self.start_time
        weight_health = [a.spectral_entropy for a in agents]
        ratio = succ / max(1, mort)
        
        print(f"\n[STEP {step:04d}] " + "="*85)
        print(f" LOGISTICS  | Succ: {succ:5d} | Mort: {mort:5d} | Ratio: {ratio:.4f} | TPS: {step/max(0.1,elapsed):.1f}")
        print(f" SPECTRUM   | Spectral Entropy: [{self.sparkline(weight_health)}] {np.mean(weight_health):.3f}/3.46")
        
        avg_tg = np.mean([t['target_gain'] for t in telemetry])
        avg_pa = np.mean([t['proprioception'] for t in telemetry])
        snr = avg_tg / max(0.001, avg_pa)
        self.snr_history.append(snr)
        
        print(f" VISION     | SNR: {snr:.2f} [{self.sparkline(list(self.snr_history))}] | Target: {avg_tg*100:2.1f}% | Proprioception: {avg_pa*100:2.1f}%")
        print(f" GEOMETRY   | Target_V: [X:{telemetry[0]['rv'][0]:+.2f} Y:{telemetry[0]['rv'][1]:+.2f}] | Ego_V: [X:{telemetry[0]['sv'][0]:+.2f} Y:{telemetry[0]['sv'][1]:+.2f}]")
        print(f" AGENT_0    | Param Resets: {agents[0].reset_count} | Lag: {envs[0].lag:02d} | PolicyCert: {telemetry[0]['ct']:.2f}")
        
        if step % 100 == 0:
            print("\n--- ENVIRONMENT RENDER (AGENT 0) ---")
            print(envs[0].render_ascii())
        print("-" * 105)

# =============================================================================
# 3. NEURAL ARCHITECTURE: BINOCULAR GEOMETRIC ENCODER
# =============================================================================
class AgentNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid = config["grid_size"]
        lat = config["latent_dim"]
        
        # Dual-Channel Saliency CNN
        self.eye_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 2, kernel_size=1) 
        )
        
        # Symmetric Latent Projection Paths
        self.path_int = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        self.path_ext = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        self.consistency_model = nn.Linear(lat, lat)
        
        # Latent Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(lat, config["bottleneck_dim"]), nn.ReLU(), nn.Linear(config["bottleneck_dim"], lat)
        )
        
        self.policy_head = nn.Linear(lat + 4, 4) 
        self.value_head = nn.Linear(lat, 1)
        
        self.reset_count, self.death_streak, self.spectral_entropy = 0, 0, 3.46

    def compute_centroids(self, logits, head_pos):
        batch_size = logits.shape[0]
        # Coordinate extraction via high-temperature spatial softmax
        target_mask = F.softmax(logits[:, 0, :, :].view(batch_size, -1) * 150.0, dim=-1)
        ego_mask = F.softmax(logits[:, 1, :, :].view(batch_size, -1) * 150.0, dim=-1)
        
        indices = torch.linspace(0, self.grid - 1, self.grid).to(device)
        
        # Geometric reduction
        tm = target_mask.view(batch_size, self.grid, self.grid)
        tx, ty = (tm.sum(dim=2) * indices).sum(dim=1), (tm.sum(dim=1) * indices).sum(dim=1)
        target_v = (torch.stack([tx, ty], dim=-1) - head_pos) / (self.grid / 2.0)
        
        em = ego_mask.view(batch_size, self.grid, self.grid)
        ex, ey = (em.sum(dim=2) * indices).sum(dim=1), (em.sum(dim=1) * indices).sum(dim=1)
        ego_v = (torch.stack([ex, ey], dim=-1) - head_pos) / (self.grid / 2.0)
        
        return target_v, ego_v, target_mask, ego_mask

    def forward(self, x, head_pos):
        x_proc = x.clone()
        # Feature subtraction for improved signal separation
        x_proc[:, 2, :, :] -= x_proc[:, 0, :, :]
        x_proc[:, 2, :, :] = torch.clamp(x_proc[:, 2, :, :], 0, 1) * 100.0
        
        # FIXED: Corrected attribute name call to eye_cnn
        logits = self.eye_cnn(x_proc)
        target_v, ego_v, t_mask, e_mask = self.compute_centroids(logits, head_pos)
        
        target_gain = (t_mask * x[:, 2, :, :].view(x.shape[0], -1)).sum()
        ego_awareness = (e_mask * x[:, 0, :, :].view(x.shape[0], -1)).sum()
        
        masked_input = (x[:, 2, :, :].view(x.shape[0], -1) * t_mask)
        z_i, z_e = torch.tanh(self.path_int(masked_input)), torch.tanh(self.path_ext(masked_input))
        z_v = self.bottleneck(z_i)
        
        return z_i, z_e, self.consistency_model(z_i), z_v, target_v, ego_v, target_gain, ego_awareness

# =============================================================================
# 4. ENVIRONMENT: GRID-WORLD SIMULATOR
# =============================================================================
class SnakeEnv:
    def __init__(self, size):
        self.size = size
        self.reset()
    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.food = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.done, self.lag = False, 0
        return self.get_obs()
    def step(self, action):
        if self.done: return self.get_obs(), 0.0, True, False, 0, 0
        self.lag += 1
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        hx, hy = self.snake[0][0] + dx, self.snake[0][1] + dy
        old_d = abs(self.snake[0][0]-self.food[0]) + abs(self.snake[0][1]-self.food[1])
        if hx < 0 or hy < 0 or hx >= self.size or hy >= self.size or (hx, hy) in self.snake:
            self.done = True; return self.get_obs(), -20.0, True, False, 0, old_d
        self.snake.insert(0, (hx, hy))
        new_d = abs(hx-self.food[0]) + abs(hy-self.food[1])
        if (hx, hy) == self.food:
            self.food = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            self.lag = 0; return self.get_obs(), 1000.0, False, True, old_d-new_d, 0
        if len(self.snake) > 5: self.snake.pop()
        return self.get_obs(), -0.1, False, False, old_d-new_d, new_d
    def get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        for i, (sx, sy) in enumerate(self.snake): obs[0, sx, sy] = 1.0 if i == 0 else 0.5 
        obs[1, :, :] = 0.1 
        obs[2, self.food[0], self.food[1]] = 1.0 
        return obs
    def render_ascii(self):
        grid = np.full((self.size, self.size), "·")
        grid[self.food[0], self.food[1]] = "★"
        for i, (sx, sy) in enumerate(self.snake): grid[sx, sy] = "S"
        return "\n".join([" ".join(row) for row in grid])

# =============================================================================
# 5. TRAINING AND EXECUTION
# =============================================================================
def train():
    print(f"--- INITIALIZING RS-3 TRAINING ON {str(device).upper()} ---")
    monitor = PerformanceMonitor()
    agents = [AgentNetwork(CONFIG).to(device) for _ in range(CONFIG["num_agents"])]
    opts = [optim.Adam(agents[i].parameters(), lr=CONFIG["learning_rate"]) for i in range(CONFIG["num_agents"])]
    envs = [SnakeEnv(CONFIG["grid_size"]) for _ in range(CONFIG["num_agents"])]
    obs_batch = [torch.tensor(e.reset()).unsqueeze(0).to(device) for e in envs]
    latent_field = torch.zeros(1, CONFIG["latent_dim"], device=device)
    total_succ, total_mort = 0, 0

    for step in range(CONFIG["total_steps"]):
        latent_stack, winners, step_logs = [], [], []
        
        for i, agent in enumerate(agents):
            if agent.death_streak > 20:
                nn.init.xavier_uniform_(agent.policy_head.weight); agent.reset_count += 1; agent.death_streak = 0

            h_pos = torch.tensor([[float(envs[i].snake[0][0]), float(envs[i].snake[0][1])]], device=device)
            z_i, z_e, z_p, z_v, target_v, ego_v, tg, sa = agent(obs_batch[i], h_pos)
            
            # Heuristic Bias Injection (Directional Prior)
            bias = torch.zeros(1, 4, device=device)
            tx, ty = target_v[0,0].item(), target_v[0,1].item()
            if tx < -0.05: bias[0,0] = 12.0 # North
            if tx > 0.05:  bias[0,2] = 12.0 # South
            if ty < -0.05: bias[0,3] = 12.0 # West
            if ty > 0.05:  bias[0,1] = 12.0 # East

            z_coupled = torch.cat([z_v + CONFIG["field_coupling"] * latent_field, target_v, ego_v], dim=-1)
            logits = agent.policy_head(z_coupled) + bias
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            
            n_obs, rew, done, ate, d_delta, d_raw = envs[i].step(action)
            if ate: total_succ += 1; agent.death_streak = 0; winners.append(i)
            if done: total_mort += 1; agent.death_streak += 1
            latent_stack.append(z_v)

            # --- LOSS SYNTHESIS ---
            true_v = (torch.tensor([[float(envs[i].food[0]), float(envs[i].food[1])]], device=device) - h_pos) / (CONFIG["grid_size"] / 2.0)
            l_coord = 300.0 * F.mse_loss(target_v, true_v) if step < CONFIG["oracle_steps"] else torch.tensor(0.0, device=device)
            l_unity = CONFIG["contrastive_weight"] * (F.mse_loss(z_i, z_e) + F.mse_loss(z_i, z_p))
            l_ortho = CONFIG["spectral_weight"] * F.mse_loss(torch.mm(agent.path_int.weight, agent.path_int.weight.t()), torch.eye(32, device=device))
            l_rl = -agent.value_head(z_v).mean() * (rew + d_delta) + CONFIG["base_entropy"] * dist.entropy().mean()
            
            total_loss = l_rl + l_unity + l_coord + l_ortho + CONFIG["proprioceptive_weight"]*(1.0-sa) + CONFIG["vision_gate_weight"]*(0.0-tg)
            opts[i].zero_grad(); total_loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.1); opts[i].step()

            # Weight Health Check (SVD entropy)
            with torch.no_grad():
                s = torch.linalg.svdvals(agent.path_int.weight); p = s / (s.sum() + 1e-9)
                agent.spectral_entropy = -(p * torch.log(p + 1e-9)).sum().item()

            agent.last_cert = 1.0-(dist.entropy().mean().item()/1.38)
            obs_batch[i] = torch.tensor(envs[i].reset() if done else n_obs).unsqueeze(0).to(device)
            
            # FIXED: Synchronized keys with log_status logic
            step_logs.append({
                'target_gain': tg.item(), 
                'proprioception': sa.item(), 
                'rv': [tx, ty], 
                'sv': ego_v[0].detach().cpu().numpy(), 
                'dr': d_raw, 
                'ct': agent.last_cert
            })

        if len(winners) > 0: latent_field = 0.8 * latent_field + 0.2 * torch.cat(latent_stack, 0)[winners].mean(dim=0, keepdim=True).detach()
        if step % 50 == 0: monitor.log_status(step, total_succ, total_mort, envs, agents, step_logs)

    torch.save(agents[0].state_dict(), CONFIG["model_path"])

if __name__ == "__main__":
    train()
