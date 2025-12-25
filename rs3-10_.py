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
# 1. SYSTEM CONFIGURATION (48x48 High-Capacity)
# =============================================================================
CONFIG = {
    "grid_size": 12,
    "num_agents": 3,               
    "latent_dim": 128,             
    "bottleneck_dim": 32,          
    "total_steps": 60001,          
    "learning_rate": 1e-4,         
    "contrastive_weight": 3.0,     
    "spectral_weight": 0.8,        
    "vision_gate_weight": 60.0,    
    "alignment_weight": 40.0,      
    "proprioceptive_weight": 35.0, 
    "field_coupling": 0.40,        # Autonomous Search Regime
    "oracle_steps": 5000,          
    "base_entropy": 0.3,           
    "bias_decay_end": 10000,       
    "starvation_limit": 500,       
    "model_path": "geometric_smart_model.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
device = torch.device(CONFIG["device"])

# =============================================================================
# 2. OMNISCIENCE MONITOR (Complete Data Suite)
# =============================================================================
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.snr_history = deque(maxlen=50)
        self.succ_history = deque(maxlen=50)
        self.mort_history = deque(maxlen=50)

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

    def log_status(self, step, succs, morts, starve_counts, envs, agents, telemetry):
        elapsed = time.time() - self.start_time
        avg_tg = np.mean([t['target_gain'] for t in telemetry])
        avg_pa = np.mean([t['proprioception'] for t in telemetry])
        snr = avg_tg / max(0.001, avg_pa)
        
        self.snr_history.append(snr)
        self.succ_history.append(sum(succs))
        self.mort_history.append(sum(morts))

        print(f"\n[STEP {step:05d}] " + "="*110)
        print(f" GLOBAL STATS | Total Succ: {sum(succs):5d} [{self.sparkline(list(self.succ_history))}]")
        print(f" LOGISTICS    | Total Mort: {sum(morts):5d} [{self.sparkline(list(self.mort_history))}] | Starved: {sum(starve_counts)}")
        print(f" VISION SNR   | {snr:7.2f} [{self.sparkline(list(self.snr_history))}] | Oracle: {'ON' if step < CONFIG['oracle_steps'] else 'OFF'} | TPS: {step/max(0.1,elapsed):.1f}")
        print("-" * 123)
        print(f" ID  | LEN | SUCC | LAG | CERT | GAZE (X,   Y)   | OPTIMISM | SPECTRAL | STATUS")
        print("-" * 123)

        for i in range(CONFIG["num_agents"]):
            t = telemetry[i]
            e = envs[i]
            a = agents[i]
            gx, gy = t['gaze']
            
            # Status Logic
            status = "WANDERING"
            if e.lag > 400: status = "STARVING "
            elif t['val'] > 10.0: status = "SUPER-AG"
            elif t['val'] > 1.0: status = "TARGETED "
            elif abs(gx) > 0.05 or abs(gy) > 0.05: status = "TRACKING "

            print(f" A{i}  | {len(e.snake):03d} | {succs[i]:4d} | {e.lag:03d} | {t['ct']:.2f} | [{gx:+.2f}, {gy:+.2f}] | {t['val']:+8.4f} |  {a.spectral_entropy:.2f}    | {status}")

        if step % 500 == 0:
            # Dynamic Leader Selection for Rendering
            leader_idx = int(np.argmax(succs))
            print(f"\n--- SPATIAL PROJECTION (LEADER: A{leader_idx}) ---")
            print(envs[leader_idx].render_ascii())
        print("=" * 123)

# =============================================================================
# 3. NEURAL ARCHITECTURE
# =============================================================================
class AgentNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid = config["grid_size"]
        lat = config["latent_dim"]

        self.eye_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 2, kernel_size=1)
        )

        self.path_int = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        self.path_ext = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        self.consistency_model = nn.Linear(lat, lat)

        self.bottleneck = nn.Sequential(
            nn.Linear(lat, config["bottleneck_dim"]), nn.ReLU(), nn.Linear(config["bottleneck_dim"], lat)
        )

        self.policy_head = nn.Linear(lat + 4, 4)
        self.value_head = nn.Linear(lat, 1)
        self.spectral_entropy = 4.85

    def compute_centroids(self, logits, head_pos):
        batch_size = logits.shape[0]
        target_mask = F.softmax(logits[:, 0, :, :].view(batch_size, -1) * 150.0, dim=-1)
        ego_mask = F.softmax(logits[:, 1, :, :].view(batch_size, -1) * 150.0, dim=-1)
        indices = torch.linspace(0, self.grid - 1, self.grid).to(device)
        tm = target_mask.view(batch_size, self.grid, self.grid)
        tx, ty = (tm.sum(dim=2) * indices).sum(dim=1), (tm.sum(dim=1) * indices).sum(dim=1)
        target_v = (torch.stack([tx, ty], dim=-1) - head_pos) / (self.grid / 2.0)
        em = ego_mask.view(batch_size, self.grid, self.grid)
        ex, ey = (em.sum(dim=2) * indices).sum(dim=1), (em.sum(dim=1) * indices).sum(dim=1)
        ego_v = (torch.stack([ex, ey], dim=-1) - head_pos) / (self.grid / 2.0)
        return target_v, ego_v, target_mask, ego_mask

    def forward(self, x, head_pos):
        x_proc = x.clone()
        x_proc[:, 2, :, :] -= x_proc[:, 0, :, :] 
        x_proc[:, 2, :, :] = torch.clamp(x_proc[:, 2, :, :], 0, 1) * 100.0
        logits = self.eye_cnn(x_proc)
        target_v, ego_v, t_mask, e_mask = self.compute_centroids(logits, head_pos)
        target_gain = (t_mask * x[:, 2, :, :].view(x.shape[0], -1)).sum()
        ego_awareness = (e_mask * x[:, 0, :, :].view(x.shape[0], -1)).sum()
        masked_input = (x[:, 2, :, :].view(x.shape[0], -1) * t_mask)
        z_i, z_e = torch.tanh(self.path_int(masked_input)), torch.tanh(self.path_ext(masked_input))
        z_v = self.bottleneck(z_i)
        return z_i, z_e, self.consistency_model(z_i), z_v, target_v, ego_v, target_gain, ego_awareness

# =============================================================================
# 4. ENVIRONMENT: 48x48 GRID WITH STARVATION
# =============================================================================
class SnakeEnv:
    def __init__(self, size, starvation_limit):
        self.size = size
        self.starvation_limit = starvation_limit
        self.reset()
    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.food = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.done, self.lag = False, 0
        return self.get_obs()
    def step(self, action):
        if self.done: return self.get_obs(), 0.0, True, False, 0, False
        self.lag += 1
        # Starvation
        if self.lag > self.starvation_limit:
            self.done = True; return self.get_obs(), -15.0, True, False, 0, True
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        hx, hy = self.snake[0][0] + dx, self.snake[0][1] + dy
        old_d = abs(self.snake[0][0]-self.food[0]) + abs(self.snake[0][1]-self.food[1])
        # Collision
        if hx < 0 or hy < 0 or hx >= self.size or hy >= self.size or (hx, hy) in self.snake:
            self.done = True; return self.get_obs(), -50.0, True, False, 0, False
        self.snake.insert(0, (hx, hy))
        new_d = abs(hx-self.food[0]) + abs(hy-self.food[1])
        # Eating
        if (hx, hy) == self.food:
            while True:
                self.food = (random.randint(0, self.size-1), random.randint(0, self.size-1))
                if self.food not in self.snake: break
            self.lag = 0; return self.get_obs(), 200.0, False, True, old_d-new_d, False
        self.snake.pop()
        return self.get_obs(), -0.1, False, False, old_d-new_d, False
    def get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        for i, (sx, sy) in enumerate(self.snake): obs[0, sx, sy] = 1.0 if i == 0 else 0.5
        obs[1, :, :] = 0.1
        obs[2, self.food[0], self.food[1]] = 10.0  # INCREASED: Brighter Food (Lighthouse Effect)
        return obs
    def render_ascii(self):
        grid = np.full((self.size, self.size), "·")
        grid[self.food[0], self.food[1]] = "★"
        for i, (sx, sy) in enumerate(self.snake): grid[sx, sy] = "H" if i==0 else "S"
        return "\n".join([" ".join(row) for row in grid])

# =============================================================================
# 5. TRAINING LOOP (AUTO-RESUME)
# =============================================================================
def train():
    monitor = PerformanceMonitor()
    agents = [AgentNetwork(CONFIG).to(device) for _ in range(CONFIG["num_agents"])]
    
    # LOAD CHECKPOINT IF EXISTS
    if os.path.exists(CONFIG["model_path"]):
        print(f"--- RESUMING FROM {CONFIG['model_path']} ---")
        try:
            loaded_state = torch.load(CONFIG["model_path"], map_location=device)
            # Handle potential shape mismatch if configs changed previously
            # This logic assumes structure is compatible
            for a in agents:
                a.load_state_dict(loaded_state, strict=False)
            print("--- MODEL LOADED SUCCESSFULLY ---")
        except Exception as e:
            print(f"--- LOAD FAILED (Starting Fresh): {e} ---")
    else:
        print("--- NO CHECKPOINT FOUND: STARTING NEW RUN ---")

    opts = [optim.Adam(agents[i].parameters(), lr=CONFIG["learning_rate"]) for i in range(CONFIG["num_agents"])]
    envs = [SnakeEnv(CONFIG["grid_size"], CONFIG["starvation_limit"]) for _ in range(CONFIG["num_agents"])]
    
    agent_succs = [0] * CONFIG["num_agents"]
    agent_morts = [0] * CONFIG["num_agents"]
    agent_starves = [0] * CONFIG["num_agents"]
    obs_batch = [torch.tensor(e.reset()).unsqueeze(0).to(device) for e in envs]
    latent_field = torch.zeros(1, CONFIG["latent_dim"], device=device)

    for step in range(CONFIG["total_steps"]):
        bias_strength = max(0.0, 12.0 * (1 - step / CONFIG["bias_decay_end"]))
        latent_stack, winners, step_logs = [], [], []

        for i, agent in enumerate(agents):
            h_pos = torch.tensor([[float(envs[i].snake[0][0]), float(envs[i].snake[0][1])]], device=device)
            z_i, z_e, z_p, z_v, target_v, ego_v, tg, sa = agent(obs_batch[i], h_pos)
            pred_val = agent.value_head(z_v).item()

            bias = torch.zeros(1, 4, device=device)
            tx, ty = target_v[0,0].item(), target_v[0,1].item()
            if tx < -0.02: bias[0,0] = bias_strength 
            if tx > 0.02:  bias[0,2] = bias_strength 
            if ty < -0.02: bias[0,3] = bias_strength 
            if ty > 0.02:  bias[0,1] = bias_strength 

            z_coupled = torch.cat([z_v + CONFIG["field_coupling"] * latent_field, target_v, ego_v], dim=-1)
            logits = agent.policy_head(z_coupled) + bias
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

            n_obs, rew, done, ate, d_delta, starved = envs[i].step(action)
            if ate: agent_succs[i] += 1; winners.append(i)
            if done: 
                agent_morts[i] += 1
                if starved: agent_starves[i] += 1
            latent_stack.append(z_v)

            # --- LOSS SYNTHESIS ---
            true_v = (torch.tensor([[float(envs[i].food[0]), float(envs[i].food[1])]], device=device) - h_pos) / (CONFIG["grid_size"] / 2.0)
            l_coord = 400.0 * F.mse_loss(target_v, true_v) if step < CONFIG["oracle_steps"] else torch.tensor(0.0, device=device)
            l_unity = CONFIG["contrastive_weight"] * (F.mse_loss(z_i, z_e) + F.mse_loss(z_i, z_p))
            l_ortho = CONFIG["spectral_weight"] * F.mse_loss(torch.mm(agent.path_int.weight, agent.path_int.weight.t()), torch.eye(CONFIG["latent_dim"], device=device))
            l_rl = -agent.value_head(z_v).mean() * (rew + d_delta * 10.0) + CONFIG["base_entropy"] * dist.entropy().mean()

            total_loss = l_rl + l_unity + l_coord + l_ortho + CONFIG["proprioceptive_weight"]*(1.0-sa) + CONFIG["vision_gate_weight"]*(0.0-tg)
            opts[i].zero_grad(); total_loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.1); opts[i].step()

            # Spectral Check
            with torch.no_grad():
                s = torch.linalg.svdvals(agent.path_int.weight); p = s / (s.sum() + 1e-9)
                agent.spectral_entropy = -(p * torch.log(p + 1e-9)).sum().item()

            obs_batch[i] = torch.tensor(envs[i].reset() if done else n_obs).unsqueeze(0).to(device)
            step_logs.append({
                'target_gain': tg.item(), 
                'proprioception': sa.item(), 
                'ct': 1.0-(dist.entropy().mean().item()/1.38), 
                'val': pred_val, 
                'gaze': [tx, ty]
            })

        if len(winners) > 0: latent_field = 0.8 * latent_field + 0.2 * torch.cat(latent_stack, 0)[winners].mean(dim=0, keepdim=True).detach()
        if step % 100 == 0: 
            monitor.log_status(step, agent_succs, agent_morts, agent_starves, envs, agents, step_logs)
            # Safety Save every 1000 steps
            if step % 60000 == 0: torch.save(agents[0].state_dict(), CONFIG["model_path"])

    torch.save(agents[0].state_dict(), CONFIG["model_path"])

if __name__ == "__main__":
    train()
