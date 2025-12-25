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
# 1. CONFIGURATION
# =============================================================================
CONFIG = {
    "grid_size": 12,
    "num_agents": 6,
    "latent_dim": 32,
    "bottleneck_dim": 8,
    "total_steps": 5001,
    "learning_rate": 8e-5,
    "contrastive_weight": 2.5,
    "orthogonal_weight": 0.6,
    "vision_gate_weight": 55.0,
    "alignment_weight": 35.0,
    "self_awareness_weight": 30.0,
    "field_coupling": 0.85,
    "oracle_steps": 1000,
    "base_entropy": 0.4,
    "model_path": "rs3_omega_standard.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
device = torch.device(CONFIG["device"])

# =============================================================================
# 2. DIAGNOSTICS CLASS
# =============================================================================
class Diagnostics:
    def __init__(self):
        self.start_time = time.time()
        self.hist_snr = deque(maxlen=100)

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

    def print_dashboard(self, step, succ, mort, envs, agents, telemetry):
        elapsed = time.time() - self.start_time
        jar_ents = [a.weight_entropy for a in agents]
        ratio = succ / max(1, mort)
        
        print(f"\n[STEP {step:04d}] " + "=" * 85)
        print(f" LOGISTICS  | Succ: {succ:5d} | Mort: {mort:5d} | Ratio: {ratio:.4f} | TPS: {step / max(0.1, elapsed):.1f}")
        print(f" SPECTRUM   | Weight Entropy: [{self.sparkline(jar_ents)}] {np.mean(jar_ents):.3f}/3.46")
        
        avg_tg = np.mean([t['target_gain'] for t in telemetry])
        avg_sa = np.mean([t['self_awareness'] for t in telemetry])
        snr = avg_tg / max(0.001, avg_sa)
        self.hist_snr.append(snr)
        
        print(f" VISION     | SNR: {snr:.2f} [{self.sparkline(list(self.hist_snr))}] | Target: {avg_tg * 100:2.1f}% | Self Aware: {avg_sa * 100:2.1f}%")
        print(f" VECTORS    | Target_V: [X:{telemetry[0]['rv'][0]:+.2f} Y:{telemetry[0]['rv'][1]:+.2f}] | Self_V: [X:{telemetry[0]['sv'][0]:+.2f} Y:{telemetry[0]['sv'][1]:+.2f}]")
        print(f" AGENT_0    | Repents: {agents[0].repents} | Lag: {envs[0].lag:02d} | Cert: {telemetry[0]['ct']:.2f}")
        
        if step % 100 == 0:
            print("\n--- ENVIRONMENT RENDER (AGENT 0) ---")
            print(envs[0].render_ascii())
        print("-" * 105)

# =============================================================================
# 3. NEURAL ARCHITECTURE
# =============================================================================
class AgentNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid = config["grid_size"]
        lat = config["latent_dim"]
        
        # Feature Extraction CNN
        self.eye_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 2, kernel_size=1) 
        )
        
        # Dual-Path Latent Projection
        self.path_int = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        self.path_ext = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        
        # Recursive Self-Model
        self.self_model = nn.Linear(lat, lat)
        
        # Dimensionality Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(lat, config["bottleneck_dim"]),
            nn.ReLU(),
            nn.Linear(config["bottleneck_dim"], lat)
        )

        # FIXED: Explicitly define heads inside the class
        self.policy_head = nn.Linear(lat + 4, 4)
        self.value_head = nn.Linear(lat, 1)
        
        self.temp, self.repents, self.death_streak, self.weight_entropy = 1.0, 0, 0, 3.46

    def calculate_centroids(self, logits, head_pos):
        batch_size = logits.shape[0]
        target_mask = F.softmax(logits[:, 0, :, :].view(batch_size, -1) * 150.0, dim=-1)
        self_mask = F.softmax(logits[:, 1, :, :].view(batch_size, -1) * 150.0, dim=-1)
        indices = torch.linspace(0, self.grid - 1, self.grid).to(device)
        
        tm = target_mask.view(batch_size, self.grid, self.grid)
        tx, ty = (tm.sum(dim=2) * indices).sum(dim=1), (tm.sum(dim=1) * indices).sum(dim=1)
        target_v = (torch.stack([tx, ty], dim=-1) - head_pos) / (self.grid / 2.0)
        
        sm = self_mask.view(batch_size, self.grid, self.grid)
        sx, sy = (sm.sum(dim=2) * indices).sum(dim=1), (sm.sum(dim=1) * indices).sum(dim=1)
        self_v = (torch.stack([sx, sy], dim=-1) - head_pos) / (self.grid / 2.0)
        
        return target_v, self_v, target_mask, self_mask

    def forward(self, x, head_pos):
        x_proc = x.clone()
        x_proc[:, 2, :, :] -= x_proc[:, 0, :, :]
        x_proc[:, 2, :, :] = torch.clamp(x_proc[:, 2, :, :], 0, 1) * 100.0
        
        logits = self.eye_cnn(x_proc)
        target_v, self_v, t_mask, s_mask = self.calculate_centroids(logits, head_pos)
        
        target_gain = (t_mask * x[:, 2, :, :].view(x.shape[0], -1)).sum()
        self_awareness = (s_mask * x[:, 0, :, :].view(x.shape[0], -1)).sum()
        
        masked_input = (x[:, 2, :, :].view(x.shape[0], -1) * t_mask)
        z_i, z_e = torch.tanh(self.path_int(masked_input)), torch.tanh(self.path_ext(masked_input))
        z_v = self.bottleneck(z_i)
        
        return z_i, z_e, self.self_model(z_i), z_v, target_v, self_v, target_gain, self_awareness

# =============================================================================
# 4. ENVIRONMENT CLASS
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
# 5. TRAINING AND DEMO
# =============================================================================
def train():
    print(f"--- RS-3 OMEGA (FIXED HEADS) ON {str(device).upper()} ---")
    diag = Diagnostics()
    agents = [AgentNetwork(CONFIG).to(device) for _ in range(CONFIG["num_agents"])]
    opts = [optim.Adam(agents[i].parameters(), lr=CONFIG["learning_rate"]) for i in range(CONFIG["num_agents"])]
    envs = [SnakeEnv(CONFIG["grid_size"]) for _ in range(CONFIG["num_agents"])]
    obs_batch = [torch.tensor(e.reset()).unsqueeze(0).to(device) for e in envs]
    treasure_field = torch.zeros(1, CONFIG["latent_dim"], device=device)
    total_succ, total_mort = 0, 0
    
    for step in range(CONFIG["total_steps"]):
        latents_list, winners, telemetry = [], [], []
        for i, agent in enumerate(agents):
            if agent.death_streak > 20:
                nn.init.xavier_uniform_(agent.policy_head.weight); agent.repents += 1; agent.death_streak = 0
            h_pos = torch.tensor([[float(envs[i].snake[0][0]), float(envs[i].snake[0][1])]], device=device)
            z_i, z_e, z_p, z_v, target_v, self_v, tg, sa = agent(obs_batch[i], h_pos)
            cb = torch.zeros(1, 4, device=device)
            vx, vy = target_v[0, 0].item(), target_v[0, 1].item()
            if vx < -0.05: cb[0, 0] = 12.0
            if vx > 0.05:  cb[0, 2] = 12.0
            if vy < -0.05: cb[0, 3] = 12.0
            if vy > 0.05:  cb[0, 1] = 12.0
            z_coupled = torch.cat([z_v + CONFIG["field_coupling"] * treasure_field, target_v, self_v], dim=-1)
            logits = agent.policy_head(z_coupled) + cb
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            n_obs, rew, done, ate, d_delta, d_raw = envs[i].step(action)
            if ate: total_succ += 1; agent.death_streak = 0; winners.append(i)
            if done: total_mort += 1; agent.death_streak += 1
            latents_list.append(z_v)
            t_rel = (torch.tensor([[float(envs[i].food[0]), float(envs[i].food[1])]], device=device) - h_pos) / (CONFIG["grid_size"] / 2.0)
            l_rl = -agent.value_head(z_v).mean() * (rew + d_delta) + CONFIG["base_entropy"] * dist.entropy().mean()
            total_loss = l_rl + CONFIG["contrastive_weight"]*(F.mse_loss(z_i, z_e) + F.mse_loss(z_i, z_p)) + 300.0*F.mse_loss(target_v, t_rel) + CONFIG["orthogonal_weight"]*F.mse_loss(torch.mm(agent.path_int.weight, agent.path_int.weight.t()), torch.eye(32, device=device)) + CONFIG["self_awareness_weight"]*(1.0-sa) + CONFIG["vision_gate_weight"]*(0.0-tg)
            opts[i].zero_grad(); total_loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.1); opts[i].step()
            with torch.no_grad():
                s = torch.linalg.svdvals(agent.path_int.weight); p = s / (s.sum() + 1e-9)
                agent.weight_entropy = -(p * torch.log(p + 1e-9)).sum().item()
            agent.last_cert = 1.0 - (dist.entropy().mean().item() / 1.38)
            obs_batch[i] = torch.tensor(envs[i].reset() if done else n_obs).unsqueeze(0).to(device)
            telemetry.append({'target_gain': tg.item(), 'self_awareness': sa.item(), 'rv': [vx, vy], 'sv': self_v[0].detach().cpu().numpy(), 'dr': d_raw, 'ct': agent.last_cert})
        if len(winners) > 0: treasure_field = 0.8 * treasure_field + 0.2 * torch.cat(latents_list, 0)[winners].mean(dim=0, keepdim=True).detach()
        if step % 50 == 0: diag.print_dashboard(step, total_succ, total_mort, envs, agents, telemetry)
    torch.save(agents[0].state_dict(), CONFIG["model_path"])

def play_demo():
    if not os.path.exists(CONFIG["model_path"]): return
    agent = AgentNetwork(CONFIG).to(device)
    agent.load_state_dict(torch.load(CONFIG["model_path"]))
    agent.eval()
    env = SnakeEnv(CONFIG["grid_size"])
    obs = torch.tensor(env.reset()).unsqueeze(0).to(device)
    for _ in range(50):
        os.system('cls' if os.name == 'nt' else 'clear'); print(env.render_ascii())
        h_pos = torch.tensor([[float(env.snake[0][0]), float(env.snake[0][1])]], device=device)
        with torch.no_grad():
            _, _, _, z_v, target_v, self_v, _, _ = agent(obs, h_pos)
            cb = torch.zeros(1, 4, device=device)
            vx, vy = target_v[0,0].item(), target_v[0,1].item()
            if vx < -0.05: cb[0,0] = 12.0
            if vx > 0.05:  cb[0,2] = 12.0
            if vy < -0.05: cb[0,3] = 12.0
            if vy > 0.05:  cb[0,1] = 12.0
            logits = agent.policy_head(torch.cat([z_v, target_v, self_v], dim=-1)) + cb
            action = torch.argmax(logits).item()
        next_obs, _, done, _, _, _ = env.step(action)
        obs = torch.tensor(env.reset() if done else next_obs).unsqueeze(0).to(device); time.sleep(0.2)

if __name__ == "__main__":
    train()
    play_demo()
