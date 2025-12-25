import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import sys
from collections import deque

# =============================================================================
# 1. CONFIGURATION (normal language)
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
    "vision_gate_weight": 50.0,
    "alignment_weight": 35.0,
    "self_awareness_weight": 30.0,
    "field_coupling": 0.85,
    "oracle_steps": 1000,
    "base_entropy": 0.4,
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
        if len(data) < 2:
            return " " * length
        chars = " ▂▃▄▅▆▇█"
        min_v, max_v = min(data), max(data)
        if max_v == min_v:
            return "─" * length
        res = ""
        for i in range(length):
            idx = int((i / length) * len(data))
            char_idx = int((data[idx] - min_v) / (max_v - min_v + 1e-9) * (len(chars) - 1))
            res += chars[char_idx]
        return res

    def print_dashboard(self, step, succ, mort, envs, agents, telemetry):
        elapsed = time.time() - self.start_time
        jar_ents = [a.jar_val for a in agents]
        ratio = succ / max(1, mort)
        
        print(f"\n[STEP {step:04d}] " + "=" * 85)
        print(f" LOGISTICS  | Succ: {succ:5d} | Mort: {mort:5d} | Ratio: {ratio:.4f} | TPS: {step / max(0.1, elapsed):.1f}")
        print(f" METABOLIC  | Jar: [{self.sparkline(jar_ents)}] {np.mean(jar_ents):.3f}/3.46 | AvgTemp: {np.mean([a.temp for a in agents]):.2f}")
        
        avg_tg = np.mean([t['tf'] for t in telemetry])
        avg_sa = np.mean([t['sa'] for t in telemetry])
        snr = avg_tg / max(0.001, avg_sa)
        self.hist_snr.append(snr)
        
        print(f" VISION     | SNR: {snr:.2f} [{self.sparkline(list(self.hist_snr))}] | TargetGain: {avg_tg * 100:2.1f}% | Awareness: {avg_sa * 100:2.1f}%")
        print(f" PROPHESY   | Food_V: [X:{telemetry[0]['rv'][0]:+.2f} Y:{telemetry[0]['rv'][1]:+.2f}] | Self_V: [X:{telemetry[0]['sv'][0]:+.2f} Y:{telemetry[0]['sv'][1]:+.2f}]")
        print(f" COGNITION  | Repents: {sum([a.repents for a in agents])} | Cert: {np.mean([t['ct'] for t in telemetry]):.3f}")
        
        if step % 100 == 0:
            print("\n--- AGENT_0 VISUALIZATION ---")
            print(envs[0].render_ascii())
        print("-" * 105)

# =============================================================================
# 3. AGENT NETWORK
# =============================================================================
class AgentNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid = config["grid_size"]
        lat = config["latent_dim"]
        
        # CNN for vision processing
        self.eye_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 2, kernel_size=1)
        )
        
        # Latent paths
        self.path_int = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        self.path_ext = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        
        # Self-model
        self.self_model = nn.Linear(lat, lat)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(lat, config["bottleneck_dim"]),
            nn.ReLU(),
            nn.Linear(config["bottleneck_dim"], lat)
        )
        
        self.temp, self.repents, self.death_streak = 1.0, 0, 0
        self.jar_val = 3.46

    def calculate_anchors(self, logits, head_pos):
        batch_size = logits.shape[0]
        seek_mask = F.softmax(logits[:, 0, :, :].view(batch_size, -1) * 80.0, dim=-1)
        guard_mask = F.softmax(logits[:, 1, :, :].view(batch_size, -1) * 80.0, dim=-1)
        
        pos_indices = torch.linspace(0, self.grid - 1, self.grid).to(device)
        
        sm = seek_mask.view(batch_size, self.grid, self.grid)
        sx, sy = (sm.sum(dim=2) * pos_indices).sum(dim=1), (sm.sum(dim=1) * pos_indices).sum(dim=1)
        seek_v = (torch.stack([sx, sy], dim=-1) - head_pos) / (self.grid / 2.0)
        
        gm = guard_mask.view(batch_size, self.grid, self.grid)
        gx, gy = (gm.sum(dim=2) * pos_indices).sum(dim=1), (gm.sum(dim=1) * pos_indices).sum(dim=1)
        guard_v = (torch.stack([gx, gy], dim=-1) - head_pos) / (self.grid / 2.0)
        
        return seek_v, guard_v, seek_mask, guard_mask

    def forward(self, x, head_pos):
        x_vision = x.clone()
        # Veil of non-duality: subtract body channel from treasure channel
        x_vision[:, 2, :, :] -= x_vision[:, 0, :, :]
        x_vision[:, 2, :, :] = torch.clamp(x_vision[:, 2, :, :], 0, 1) * 100.0
        
        logits = self.eye_cnn(x_vision)
        seek_v, guard_v, s_mask, g_mask = self.calculate_anchors(logits, head_pos)
        
        tf_gain = (s_mask * x[:, 2, :, :].view(x.shape[0], -1)).sum()
        self_aware = (g_mask * x[:, 0, :, :].view(x.shape[0], -1)).sum()
        
        masked_vision = (x[:, 2, :, :].view(x.shape[0], -1) * s_mask)
        z_i, z_e = torch.tanh(self.path_int(masked_vision)), torch.tanh(self.path_ext(masked_vision))
        
        return z_i, z_e, self.self_model(z_i), self.bottleneck(z_i), seek_v, guard_v, tf_gain, self_aware

# =============================================================================
# 4. ENVIRONMENT CLASS
# =============================================================================
class SnakeEnv:
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.food = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        self.done, self.lag = False, 0
        return self.get_obs()

    def step(self, action):
        if self.done:
            return self.get_obs(), 0.0, True, False, 0, 0
        self.lag += 1
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        hx, hy = self.snake[0][0] + dx, self.snake[0][1] + dy
        old_d = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        
        if hx < 0 or hy < 0 or hx >= self.size or hy >= self.size or (hx, hy) in self.snake:
            self.done = True
            return self.get_obs(), -20.0, True, False, 0, old_d
            
        self.snake.insert(0, (hx, hy))
        new_d = abs(hx - self.food[0]) + abs(hy - self.food[1])
        
        if (hx, hy) == self.food:
            self.food = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            self.lag = 0
            return self.get_obs(), 1000.0, False, True, old_d - new_d, 0
            
        if len(self.snake) > 5:
            self.snake.pop()
        return self.get_obs(), -0.1, False, False, old_d - new_d, new_d

    def get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        for i, (sx, sy) in enumerate(self.snake):
            obs[0, sx, sy] = 1.0 if i == 0 else 0.5
        obs[1, :, :] = 0.1
        obs[2, self.food[0], self.food[1]] = 1.0
        return obs

    def render_ascii(self):
        grid = np.full((self.size, self.size), "·")
        grid[self.food[0], self.food[1]] = "★"
        for i, (sx, sy) in enumerate(self.snake):
            grid[sx, sy] = "S" if i == 0 else "x"
        return "\n".join([" ".join(row) for row in grid])

# =============================================================================
# 5. TRAINING FUNCTION
# =============================================================================
def train():
    print(f"--- PROJECT RS-3 STARTING ON {str(device).upper()} ---")
    diag = Diagnostics()
    agents = [AgentNetwork(CONFIG).to(device) for _ in range(CONFIG["num_agents"])]
    
    for a in agents:
        a.policy = nn.Linear(CONFIG["latent_dim"] + 4, 4).to(device)  # Seek_v + Guard_v
        a.value = nn.Linear(CONFIG["latent_dim"], 1).to(device)
        
    opts = [optim.Adam(agents[i].parameters(), lr=CONFIG["learning_rate"]) for i in range(CONFIG["num_agents"])]
    envs = [SnakeEnv(CONFIG["grid_size"]) for _ in range(CONFIG["num_agents"])]
    obs_batch = [torch.tensor(e.reset()).unsqueeze(0).to(device) for e in envs]
    treasure_field = torch.zeros(1, CONFIG["latent_dim"], device=device)
    total_succ, total_mort = 0, 0
    
    for step in range(CONFIG["total_steps"]):
        latents_list, winners, telemetry = [], [], []
        
        for i, agent in enumerate(agents):
            if agent.death_streak > 15:
                for layer in agent.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                agent.repents += 1
                agent.death_streak = 0
                
            h_pos = torch.tensor([[float(envs[i].snake[0][0]), float(envs[i].snake[0][1])]], device=device)
            z_i, z_e, z_p, z_v, seek_v, guard_v, tf, sa = agent(obs_batch[i], h_pos)
            
            # Compass Bias Injection
            cb = torch.zeros(1, 4, device=device)
            sx, sy = seek_v[0, 0].item(), seek_v[0, 1].item()
            if sx < -0.05: cb[0, 0] = 8.0
            if sx > 0.05:  cb[0, 2] = 8.0
            if sy < -0.05: cb[0, 3] = 8.0
            if sy > 0.05:  cb[0, 1] = 8.0
            
            z_coupled = torch.cat([z_v + CONFIG["field_coupling"] * treasure_field, seek_v, guard_v], dim=-1)
            logits = agent.policy(z_coupled) + cb
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            
            n_obs, rew, done, ate, d_delta, d_raw = envs[i].step(action)
            if ate:
                total_succ += 1
                agent.death_streak = 0
                st = "FIND"
            else:
                st = "MOVE"
            
            if done:
                total_mort += 1
                agent.death_streak += 1
            
            latents_list.append(z_v)
            
            # Loss synthesis
            true_rel_v = (torch.tensor([[float(envs[i].food[0]), float(envs[i].food[1])]], device=device) - h_pos) / (CONFIG["grid_size"] / 2.0)
            l_oracle = 300.0 * F.mse_loss(seek_v, true_rel_v) if step < CONFIG["oracle_steps"] else torch.tensor(0.0, device=device)
            l_aware = CONFIG["self_awareness_weight"] * (1.0 - sa)
            l_unity = CONFIG["contrastive_weight"] * (F.mse_loss(z_i, z_e) + F.mse_loss(z_i, z_p))
            l_ortho = CONFIG["orthogonal_weight"] * F.mse_loss(torch.mm(agent.path_int.weight, agent.path_int.weight.t()), torch.eye(32, device=device))
            l_rl = -agent.value(z_v).mean() * (rew + d_delta) + CONFIG["base_entropy"] * dist.entropy().mean()
            
            total_loss = l_rl + l_unity + l_oracle + l_aware + l_ortho + CONFIG["vision_gate_weight"] * (0.0 - tf)
            
            opts[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.1)
            opts[i].step()
            
            # Jar Entropy Audit
            with torch.no_grad():
                s = torch.linalg.svdvals(agent.path_int.weight)
                p = s / (s.sum() + 1e-9)
                agent.jar_val = -(p * torch.log(p + 1e-9)).sum().item()
            
            agent.last_cert = 1.0 - (dist.entropy().mean().item() / 1.38)
            obs_batch[i] = torch.tensor(envs[i].reset() if done else n_obs).unsqueeze(0).to(device)
            telemetry.append({
                'st': st, 'tf': tf.item(), 'sa': sa.item(), 'rv': [sx, sy], 
                'sv': guard_v[0].detach().cpu().numpy(), 'dr': d_raw, 
                'ct': agent.last_cert, 'det': seek_v[0].detach().cpu().numpy()
            })
            
        if len(winners) > 0:
            treasure_field = 0.8 * treasure_field + 0.2 * torch.cat(latents_list, 0)[winners].mean(dim=0, keepdim=True).detach()
            
        if step % 50 == 0:
            diag.print_dashboard(step, total_succ, total_mort, envs, agents, telemetry)
        elif step % 10 == 0:
            sys.stdout.write(f"\rStep {step:04d} | SNR: {telemetry[0]['tf']/max(0.01, telemetry[0]['sa']):.2f} | Lag: {envs[0].lag:02d} | C: {telemetry[0]['ct']:.2f}")
            sys.stdout.flush()

    print("\n--- RS-3 PROTOCOL COMPLETE ---")

if __name__ == "__main__":
    train()
