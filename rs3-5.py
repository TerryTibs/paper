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
# 1. SYSTEM HYPERPARAMETERS (Academic Standard)
# =============================================================================
CONFIG = {
    "grid_size": 12,
    "num_agents": 6,
    "latent_dim": 32,
    "bottleneck_dim": 8,
    "total_steps": 5001,
    "learning_rate": 8e-5,
    "contrastive_weight": 2.5,     # Dual-path representation consistency
    "orthogonal_weight": 0.6,      # Weight matrix rank regularization
    "vision_gate_weight": 55.0,    # Signal discrimination pressure
    "alignment_weight": 35.0,      # Heuristic action prior pressure
    "proprioceptive_weight": 30.0, # Ego-channel awareness weight
    "field_coupling": 0.85,        # Global latent integration
    "oracle_steps": 1000,          # Supervised coordinate warm-up
    "base_entropy": 0.4,           # Exploration coefficient
    "model_path": "geometric_rl_agent.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
device = torch.device(CONFIG["device"])

# =============================================================================
# 2. DIAGNOSTIC MONITORING
# =============================================================================
class DiagnosticOracle:
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

    def log_telemetry(self, step, succ, mort, envs, agents, logs):
        elapsed = time.time() - self.start_time
        weight_health = [a.spectral_entropy for a in agents]
        ratio = succ / max(1, mort)
        
        print(f"\n\033[1;36m[STEP {step:04d}]\033[0m " + "="*85)
        print(f" LOGISTICS  | Succ: {succ:5d} | Mort: {mort:5d} | Ratio: {ratio:.4f} | TPS: {step/max(0.1,elapsed):.1f}")
        print(f" SPECTRUM   | Spectral Entropy: [{self.sparkline(weight_health)}] {np.mean(weight_health):.3f}/3.46")
        
        avg_tg = np.mean([l['target_gain'] for l in logs])
        avg_sa = np.mean([l['proprioception'] for l in logs])
        snr = avg_tg / max(0.001, avg_sa)
        self.snr_history.append(snr)
        
        print(f" VISION     | SNR: {snr:.2f} [{self.sparkline(list(self.snr_history))}] | Target: {avg_tg*100:2.1f}% | Ego: {avg_sa*100:2.1f}%")
        print(f" GEOMETRY   | Target_V: [X:{logs[0]['rv'][0]:+.2f} Y:{logs[0]['rv'][1]:+.2f}] | Ego_V: [X:{logs[0]['sv'][0]:+.2f} Y:{logs[0]['sv'][1]:+.2f}]")
        print(f" AGENT_0    | Repents: {agents[0].repent_count} | Lag: {envs[0].lag:02d} | PolicyCert: {logs[0]['ct']:.2f}")
        
        if step % 100 == 0:
            print("\n--- SPATIAL GRID RENDER (AGENT 0) ---")
            print(envs[0].render_ascii())
        print("-" * 105)

# =============================================================================
# 3. ARCHITECTURE: BINOCULAR GEOMETRIC ENCODER
# =============================================================================
class BinocularAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid = config["grid_size"]
        lat = config["latent_dim"]
        
        # Dual-Channel Visual Saliency
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 2, kernel_size=1) 
        )
        
        # Contrastive Latent Path Weights
        self.path_internal = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        self.path_external = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        self.identity_predictor = nn.Linear(lat, lat)
        
        self.bottleneck = nn.Sequential(
            nn.Linear(lat, config["bottleneck_dim"]), nn.ReLU(), nn.Linear(config["bottleneck_dim"], lat)
        )
        
        # Output Heads
        self.policy_head = nn.Linear(lat + 4, 4) # Latent + 2 Geometric Vectors
        self.value_head = nn.Linear(lat, 1)
        
        self.temp, self.repent_count, self.death_streak, self.spectral_entropy = 1.0, 0, 0, 3.46

    def get_geometric_centroids(self, logits, head_pos):
        batch_size = logits.shape[0]
        # Softmax temperature (150.0) forces coordinate lock
        target_mask = F.softmax(logits[:, 0, :, :].view(batch_size, -1) * 150.0, dim=-1)
        ego_mask = F.softmax(logits[:, 1, :, :].view(batch_size, -1) * 150.0, dim=-1)
        
        coords = torch.linspace(0, self.grid - 1, self.grid).to(device)
        
        # Target Centroid
        tm = target_mask.view(batch_size, self.grid, self.grid)
        tx, ty = (tm.sum(dim=2) * coords).sum(dim=1), (tm.sum(dim=1) * coords).sum(dim=1)
        target_v = (torch.stack([tx, ty], dim=-1) - head_pos) / (self.grid / 2.0)
        
        # Ego Centroid
        em = ego_mask.view(batch_size, self.grid, self.grid)
        ex, ey = (em.sum(dim=2) * coords).sum(dim=1), (em.sum(dim=1) * coords).sum(dim=1)
        ego_v = (torch.stack([ex, ey], dim=-1) - head_pos) / (self.grid / 2.0)
        
        return target_v, ego_v, target_mask, ego_mask

    def forward(self, x, head_pos):
        x_processed = x.clone()
        # Subtractive filter: suppress ego signal in target channel
        x_processed[:, 2, :, :] -= x_processed[:, 0, :, :]
        x_processed[:, 2, :, :] = torch.clamp(x_processed[:, 2, :, :], 0, 1) * 100.0
        
        logits = self.vision_cnn(x_processed)
        target_v, ego_v, t_mask, e_mask = self.get_geometric_centroids(logits, head_pos)
        
        target_gain = (t_mask * x[:, 2, :, :].view(x.shape[0], -1)).sum()
        ego_awareness = (e_mask * x[:, 0, :, :].view(x.shape[0], -1)).sum()
        
        masked_input = (x[:, 2, :, :].view(x.shape[0], -1) * t_mask)
        z_i, z_e = torch.tanh(self.path_internal(masked_input)), torch.tanh(self.path_external(masked_input))
        z_v = self.bottleneck(z_i)
        
        return z_i, z_e, self.identity_predictor(z_i), z_v, target_v, ego_v, target_gain, ego_awareness

# =============================================================================
# 4. ENVIRONMENT: GEOMETRIC GRID-WORLD
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
        old_d = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        if hx < 0 or hy < 0 or hx >= self.size or hy >= self.size or (hx, hy) in self.snake:
            self.done = True; return self.get_obs(), -20.0, True, False, 0, old_d
        self.snake.insert(0, (hx, hy))
        new_d = abs(hx - self.food[0]) + abs(hy - self.food[1])
        if (hx, hy) == self.food:
            self.food = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            self.lag = 0; return self.get_obs(), 1000.0, False, True, old_d - new_d, 0
        if len(self.snake) > 5: self.snake.pop()
        return self.get_obs(), -0.1, False, False, old_d - new_d, new_d
    def get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        for i, (sx, sy) in enumerate(self.snake): obs[0, sx, sy] = 1.0 if i == 0 else 0.5
        obs[1, :, :] = 0.1 
        obs[2, self.food[0], self.food[1]] = 1.0 
        return obs
    def render_ascii(self):
        grid = np.full((self.size, self.size), "·")
        grid[self.food[0], self.food[1]] = "★"
        for i, (sx, sy) in enumerate(self.snake): grid[sx, sy] = "S" if i == 0 else "x"
        return "\n".join([" ".join(row) for row in grid])

# =============================================================================
# 5. EXECUTION PIPELINE
# =============================================================================
def train():
    print(f"--- PROJECT RS-3 STARTING ON {str(device).upper()} ---")
    oracle = DiagnosticOracle()
    agents = [BinocularAgent(CONFIG).to(device) for _ in range(CONFIG["num_agents"])]
    for a in agents:
        a.policy_head = nn.Linear(CONFIG["latent_dim"] + 4, 4).to(device)
        a.value_head = nn.Linear(CONFIG["latent_dim"], 1).to(device)
    
    opts = [optim.Adam(agents[i].parameters(), lr=CONFIG["learning_rate"]) for i in range(CONFIG["num_agents"])]
    envs = [SnakeEnv(CONFIG["grid_size"]) for _ in range(CONFIG["num_agents"])]
    obs_batch = [torch.tensor(e.reset()).unsqueeze(0).to(device) for e in envs]
    latent_field = torch.zeros(1, CONFIG["latent_dim"], device=device)
    total_succ, total_mort = 0, 0

    for step in range(CONFIG["total_steps"]):
        latent_stack, winners, step_logs = [], [], []
        
        for i, agent in enumerate(agents):
            if agent.death_streak > 20:
                nn.init.xavier_uniform_(agent.policy_head.weight); agent.repent_count += 1; agent.death_streak = 0
                
            h_pos = torch.tensor([[float(envs[i].snake[0][0]), float(envs[i].snake[0][1])]], device=device)
            z_i, z_e, z_p, z_v, target_v, ego_v, tg, sa = agent(obs_batch[i], h_pos)
            
            # Action Space Bias (Mechanical Homing)
            cb = torch.zeros(1, 4, device=device)
            tx, ty = target_v[0, 0].item(), target_v[0, 1].item()
            if tx < -0.05: cb[0, 0] = 12.0
            if tx > 0.05:  cb[0, 2] = 12.0
            if ty < -0.05: cb[0, 3] = 12.0
            if ty > 0.05:  cb[0, 1] = 12.0
            
            z_integrated = torch.cat([z_v + CONFIG["field_coupling"] * latent_field, target_v, ego_v], dim=-1)
            logits = agent.policy_head(z_integrated) + cb
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            
            n_obs, rew, done, ate, d_delta, d_raw = envs[i].step(action)
            if ate: total_succ += 1; agent.death_streak = 0; winners.append(i)
            if done: total_mort += 1; agent.death_streak += 1
            latent_stack.append(z_v)
            
            # --- SPECTRAL INFORMATION OPTIMIZATION ---
            t_rel = (torch.tensor([[float(envs[i].food[0]), float(envs[i].food[1])]], device=device) - h_pos) / (CONFIG["grid_size"] / 2.0)
            l_coord = 300.0 * F.mse_loss(target_v, t_rel) if step < CONFIG["oracle_steps"] else 0.0
            l_unity = CONFIG["contrastive_weight"] * (F.mse_loss(z_i, z_e) + F.mse_loss(z_i, z_p))
            l_ortho = CONFIG["orthogonal_weight"] * F.mse_loss(torch.mm(agent.path_internal.weight, agent.path_internal.weight.t()), torch.eye(32, device=device))
            l_rl = -agent.value_head(z_v).mean() * (rew + d_delta) + CONFIG["base_entropy"] * dist.entropy().mean()
            
            total_loss = l_rl + l_unity + l_coord + l_ortho + CONFIG["proprioceptive_weight"]*(1.0-sa) + CONFIG["vision_gate_weight"]*(0.0-tg)
            
            opts[i].zero_grad(); total_loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.1); opts[i].step()
            
            with torch.no_grad():
                s = torch.linalg.svdvals(agent.path_internal.weight); p = s / (s.sum() + 1e-9)
                agent.spectral_entropy = -(p * torch.log(p + 1e-9)).sum().item()
            
            obs_batch[i] = torch.tensor(envs[i].reset() if done else n_obs).unsqueeze(0).to(device)
            step_logs.append({
                'target_gain': tg.item(), 'proprioception': sa.item(), 
                'rv': [tx, ty], 'sv': ego_v[0].detach().cpu().numpy(), 
                'dr': d_raw, 'ct': 1.0 - (dist.entropy().mean().item() / 1.38)
            })
            
        if len(winners) > 0: latent_field = 0.8 * latent_field + 0.2 * torch.cat(latent_stack, 0)[winners].mean(dim=0, keepdim=True).detach()
        if step % 50 == 0: oracle.log_telemetry(step, total_succ, total_mort, envs, agents, step_logs)

    torch.save(agents[0].state_dict(), CONFIG["model_path"])

def play_demo():
    if not os.path.exists(CONFIG["model_path"]): return
    agent = BinocularAgent(CONFIG).to(device)
    agent.load_state_dict(torch.load(CONFIG["model_path"]))
    agent.eval()
    env = SnakeEnv(CONFIG["grid_size"])
    obs = torch.tensor(env.reset()).unsqueeze(0).to(device)
    for _ in range(50):
        os.system('cls' if os.name == 'nt' else 'clear'); print(env.render_ascii())
        h_pos = torch.tensor([[float(env.snake[0][0]), float(env.snake[0][1])]], device=device)
        with torch.no_grad():
            _, _, _, z_v, target_v, ego_v, _, _ = agent(obs, h_pos)
            cb = torch.zeros(1, 4, device=device)
            tx, ty = target_v[0,0].item(), target_v[0,1].item()
            if tx < -0.05: cb[0,0] = 12.0
            if tx > 0.05:  cb[0,2] = 12.0
            if ty < -0.05: cb[0,3] = 12.0
            if ty > 0.05:  cb[0,1] = 12.0
            logits = agent.policy_head(torch.cat([z_v, target_v, ego_v], dim=-1)) + cb
            action = torch.argmax(logits).item()
        next_obs, _, done, _, _, _ = env.step(action)
        obs = torch.tensor(env.reset() if done else next_obs).unsqueeze(0).to(device); time.sleep(0.2)

if __name__ == "__main__":
    train()
    play_demo()
