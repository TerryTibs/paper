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
# 1. HYPERPARAMETERS AND CONFIGURATION
# =============================================================================
CONFIG = {
    "grid_size": 12,
    "num_agents": 6,
    "latent_dim": 32,
    "bottleneck_dim": 8,
    "total_steps": 5001,
    "learning_rate": 8e-5,
    "contrastive_weight": 2.5,     # Weight for symmetric representation loss
    "orthogonal_weight": 0.6,      # Weight for weight-rank regularization
    "vision_gate_weight": 55.0,    # Weight for target-noise discrimination
    "alignment_weight": 35.0,      # Weight for policy-target alignment
    "self_awareness_weight": 30.0, # Weight for proprioceptive tail-avoidance
    "field_coupling": 0.85,        # Weight for shared global latent field
    "oracle_steps": 1000,          # Steps for supervised coordinate grounding
    "base_entropy": 0.4,           # Coefficient for entropy-driven exploration
    "smoothing_factor": 0.15,      # EMA alpha for visual mask smoothing
    "model_path": "rs3_agent.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
device = torch.device(CONFIG["device"])

# =============================================================================
# 2. TRAINING MONITORING AND VISUALIZATION
# =============================================================================
class TrainingMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.snr_history = deque(maxlen=100)

    def generate_sparkline(self, data, length=12):
        """Generates an ASCII sparkline representing data distribution."""
        if len(data) < 2: return " " * length
        chars = " ▂▃▄▅▆▇█"
        min_v, max_v = min(data), max(data)
        if max_v == min_v: return "─" * length
        res = ""
        for i in range(length):
            idx = int((i/length)*len(data))
            char_idx = int((data[idx]-min_v)/(max_v-min_v+1e-9)*(len(chars)-1))
            res += chars[char_idx]
        return res

    def log_status(self, step, succ, mort, envs, agents, telemetry):
        elapsed = time.time() - self.start_time
        jar_ents = [a.weight_entropy for a in agents]
        ratio = succ / max(1, mort)
        
        print(f"\n[STEP {step:04d}] " + "="*85)
        print(f" LOGISTICS  | Succ: {succ:5d} | Mort: {mort:5d} | Ratio: {ratio:.4f} | TPS: {step/max(0.1,elapsed):.1f}")
        print(f" SPECTRUM   | Weight Entropy: [{self.generate_sparkline(jar_ents)}] {np.mean(jar_ents):.3f}/3.46")
        
        avg_tg = np.mean([t['target_gain'] for t in telemetry])
        avg_sa = np.mean([t['self_aware'] for t in telemetry])
        snr = avg_tg / max(0.001, avg_sa)
        self.snr_history.append(snr)
        
        print(f" VISION     | SNR: {snr:.2f} [{self.generate_sparkline(list(self.snr_history))}] | Target: {avg_tg*100:2.1f}% | Self: {avg_sa*100:2.1f}%")
        print(f" VECTORS    | Target_V: [X:{telemetry[0]['rv'][0]:+.2f} Y:{telemetry[0]['rv'][1]:+.2f}] | Self_V: [X:{telemetry[0]['sv'][0]:+.2f} Y:{telemetry[0]['sv'][1]:+.2f}]")
        print(f" AGENT_0    | Deaths: {agents[0].death_streak} | Cert: {telemetry[0]['ct']:.2f}")
        
        if step % 100 == 0:
            print("\n--- ENVIRONMENT RENDER (AGENT 0) ---")
            print(envs[0].render_ascii())
        print("-" * 105)

# =============================================================================
# 3. NEURAL ARCHITECTURE: BINOCULAR SYMMETRIC ENCODER
# =============================================================================
class AgentNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid = config["grid_size"]
        lat = config["latent_dim"]
        
        # Saliency Heads: Channel 0 (Target Detection), Channel 1 (Self Detection)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 2, kernel_size=1) 
        )
        
        # Symmetric Internal Representation Paths
        self.path_internal = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        self.path_external = nn.utils.spectral_norm(nn.Linear(self.grid**2, lat))
        self.self_predictor = nn.Linear(lat, lat)
        
        # Bottleneck for Feature Compression
        self.bottleneck = nn.Sequential(
            nn.Linear(lat, config["bottleneck_dim"]), nn.ReLU(), nn.Linear(config["bottleneck_dim"], lat)
        )
        
        # Policy Head: Input includes Latent state + Target Vector + Self Vector
        self.policy_head = nn.Linear(lat + 4, 4)
        self.value_head = nn.Linear(lat, 1)
        
        self.repents, self.death_streak, self.weight_entropy = 0, 0, 3.46

    def calculate_centroids(self, logits, head_pos):
        """Calculates spatial center-of-mass for visual attention channels."""
        batch_size = logits.shape[0]
        # Hard attention using high-temperature Softmax
        target_mask = F.softmax(logits[:, 0, :, :].view(batch_size, -1) * 100.0, dim=-1)
        self_mask = F.softmax(logits[:, 1, :, :].view(batch_size, -1) * 100.0, dim=-1)
        
        indices = torch.linspace(0, self.grid - 1, self.grid).to(device)
        
        # Target coordinate reduction
        tm = target_mask.view(batch_size, self.grid, self.grid)
        tx, ty = (tm.sum(dim=2) * indices).sum(dim=1), (tm.sum(dim=1) * indices).sum(dim=1)
        target_v = (torch.stack([tx, ty], dim=-1) - head_pos) / (self.grid / 2.0)
        
        # Self coordinate reduction
        sm = self_mask.view(batch_size, self.grid, self.grid)
        sx, sy = (sm.sum(dim=2) * indices).sum(dim=1), (sm.sum(dim=1) * indices).sum(dim=1)
        self_v = (torch.stack([sx, sy], dim=-1) - head_pos) / (self.grid / 2.0)
        
        return target_v, self_v, target_mask, self_mask

    def forward(self, x, head_pos):
        # Subtract self-channel from target-channel to remove visual artifacts
        x_proc = x.clone()
        x_proc[:, 2, :, :] = torch.clamp(x_proc[:, 2, :, :] - x_proc[:, 0, :, :], 0, 1) * 100.0
        
        logits = self.encoder_cnn(x_proc)
        target_v, self_v, t_mask, s_mask = self.calculate_centroids(logits, head_pos)
        
        # Saliency analytics
        target_gain = (t_mask * x[:, 2, :, :].view(x.shape[0], -1)).sum()
        self_awareness = (s_mask * x[:, 0, :, :].view(x.shape[0], -1)).sum()
        
        # Representation path
        masked_input = (x[:, 2, :, :].view(x.shape[0], -1) * t_mask)
        z_i, z_e = torch.tanh(self.path_internal(masked_input)), torch.tanh(self.path_external(masked_input))
        z_v = self.bottleneck(z_i)
        
        return z_i, z_e, self.self_predictor(z_i), z_v, target_v, self_v, target_gain, self_awareness

# =============================================================================
# 4. ENVIRONMENT: REINFORCEMENT LEARNING SIMULATOR
# =============================================================================
class SnakeEnv:
    def __init__(self, size):
        self.size = size
        self.reset()
    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.food = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.done, self.lag, self.life_steps = False, 0, 0
        return self.get_obs()
    def step(self, action):
        if self.done: return self.get_obs(), 0.0, True, False, 0, 0
        self.lag += 1; self.life_steps += 1
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        hx, hy = self.snake[0][0] + dx, self.snake[0][1] + dy
        old_d = abs(self.snake[0][0]-self.food[0]) + abs(self.snake[0][1]-self.food[1])
        
        # Terminal state check
        if hx < 0 or hy < 0 or hx >= self.size or hy >= self.size or (hx, hy) in self.snake:
            self.done = True; return self.get_obs(), -20.0, True, False, 0, old_d
            
        self.snake.insert(0, (hx, hy))
        new_d = abs(hx-self.food[0]) + abs(hy-self.food[1])
        d_delta = old_d - new_d # Continuous reward signal
        
        if (hx, hy) == self.food:
            self.food = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            self.lag = 0; return self.get_obs(), 1000.0, False, True, d_delta, 0
        if len(self.snake) > 5: self.snake.pop()
        return self.get_obs(), -0.1, False, False, d_delta, new_d

    def get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        for i, (sx, sy) in enumerate(self.snake): obs[0, sx, sy] = 1.0 if i == 0 else 0.5 
        obs[1, :, :] = 0.1 
        obs[2, self.food[0], self.food[1]] = 1.0 
        return obs

    def render_ascii(self):
        grid = np.full((self.size, self.size), "·")
        grid[self.food[0], self.food[1]] = "★"
        for sx, sy in self.snake: grid[sx, sy] = "S"
        return "\n".join([" ".join(row) for row in grid])

# =============================================================================
# 5. TRAINING AND INFERENCE PIPELINE
# =============================================================================
def train():
    print(f"--- INITIALIZING DUAL-PATH REINFORCE TRAINING ON {str(device).upper()} ---")
    monitor = TrainingMonitor()
    agents = [AgentNetwork(CONFIG).to(device) for _ in range(CONFIG["num_agents"])]
    optimizers = [optim.Adam(agents[i].parameters(), lr=CONFIG["learning_rate"]) for i in range(CONFIG["num_agents"])]
    envs = [SnakeEnv(CONFIG["grid_size"]) for _ in range(CONFIG["num_agents"])]
    obs_batch = [torch.tensor(e.reset()).unsqueeze(0).to(device) for e in envs]
    global_field = torch.zeros(1, CONFIG["latent_dim"], device=device)
    total_succ, total_mort = 0, 0

    for step in range(CONFIG["total_steps"]):
        latents_list, winners, telemetry = [], [], []
        
        for i, agent in enumerate(agents):
            # Parameter re-initialization if performance plateaus
            if agent.death_streak > 20:
                nn.init.xavier_uniform_(agent.policy_head.weight); agent.repents += 1; agent.death_streak = 0

            h_pos = torch.tensor([[float(envs[i].snake[0][0]), float(envs[i].snake[0][1])]], device=device)
            z_i, z_e, z_p, z_v, target_v, self_v, tg, sa = agent(obs_batch[i], h_pos)
            
            # Directional Prior (Heuristic compass bias)
            bias = torch.zeros(1, 4, device=device)
            tx, ty = target_v[0,0].item(), target_v[0,1].item()
            if tx < -0.05: bias[0,0] = 8.0 
            if tx > 0.05:  bias[0,2] = 8.0 
            if ty < -0.05: bias[0,3] = 8.0 
            if ty > 0.05:  bias[0,1] = 8.0 

            # Policy integration with global shared latent field
            policy_input = torch.cat([z_v + CONFIG["field_coupling"] * global_field, target_v, self_v], dim=-1)
            logits = agent.policy_head(policy_input) + bias
            action = torch.distributions.Categorical(logits=logits).sample().item()
            
            n_obs, rew, done, ate, d_delta, d_raw = envs[i].step(action)
            if ate: total_succ += 1; agent.death_streak = 0
            if done: total_mort += 1; agent.death_streak += 1
            latents_list.append(z_v)

            # --- Multi-Objective Loss Synthesis ---
            t_rel = (torch.tensor([[float(envs[i].food[0]), float(envs[i].food[1])]], device=device) - h_pos) / (CONFIG["grid_size"] / 2.0)
            l_ortho = CONFIG["orthogonal_weight"] * F.mse_loss(torch.mm(agent.path_internal.weight, agent.path_internal.weight.t()), torch.eye(32, device=device))
            l_coord = 300.0 * F.mse_loss(target_v, t_rel) if step < CONFIG["oracle_steps"] else 0.0
            l_unity = CONFIG["contrastive_weight"] * (F.mse_loss(z_i, z_e) + F.mse_loss(z_i, z_p))
            l_rl = -agent.value_head(z_v).mean() * (rew + d_delta) + CONFIG["base_entropy"] * torch.distributions.Categorical(logits=logits).entropy().mean()
            
            total_loss = l_rl + l_unity + l_coord + l_ortho + CONFIG["self_awareness_weight"]*(1.0-sa) + CONFIG["vision_gate_weight"]*(0.0-tg)
            optimizers[i].zero_grad(); total_loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.1); optimizers[i].step()

            # Rank Entropy Calculation
            with torch.no_grad():
                s = torch.linalg.svdvals(agent.path_internal.weight)
                p = s / (s.sum() + 1e-9)
                agent.weight_entropy = -(p * torch.log(p + 1e-9)).sum().item()

            agent.last_cert = 1.0-(torch.distributions.Categorical(logits=logits).entropy().mean().item()/1.38)
            obs_batch[i] = torch.tensor(envs[i].reset() if done else n_obs).unsqueeze(0).to(device)
            telemetry.append({'st': "FIND" if ate else "MOVE", 'tf': tg.item(), 'sa': sa.item(), 'rv': [tx, ty], 'sv': self_v[0].detach().cpu().numpy(), 'dr': d_raw, 'ct': agent.last_cert, 'det': target_v[0].detach().cpu().numpy()})

        # Shared field update via mean of individual latents
        if len(winners) > 0: global_field = 0.8 * global_field + 0.2 * torch.cat(latents_list, 0)[winners].mean(dim=0, keepdim=True).detach()
        if step % 50 == 0: monitor.log_status(step, total_succ, total_mort, envs, agents, telemetry)

    torch.save(agents[0].state_dict(), CONFIG["model_path"])
    print(f"\n--- TRAINING COMPLETE. MODEL SAVED TO {CONFIG['model_path']} ---")

def play_demo():
    """Inference loop for real-time model evaluation."""
    if not os.path.exists(CONFIG["model_path"]): return
    agent = AgentNetwork(CONFIG).to(device)
    agent.load_state_dict(torch.load(CONFIG["model_path"]))
    agent.eval()
    env = SnakeEnv(CONFIG["grid_size"])
    obs = torch.tensor(env.reset()).unsqueeze(0).to(device)
    for _ in range(50):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(env.render_ascii())
        h_pos = torch.tensor([[float(env.snake[0][0]), float(env.snake[0][1])]], device=device)
        with torch.no_grad():
            _, _, _, z_v, target_v, self_v, _, _ = agent(obs, h_pos)
            bias = torch.zeros(1, 4, device=device)
            tx, ty = target_v[0,0].item(), target_v[0,1].item()
            if tx < -0.05: bias[0,0] = 8.0 
            if tx > 0.05:  bias[0,2] = 8.0 
            if ty < -0.05: bias[0,3] = 8.0 
            if ty > 0.05:  bias[0,1] = 8.0 
            logits = agent.policy_head(torch.cat([z_v, target_v, self_v], dim=-1)) + bias
            action = torch.argmax(logits).item()
        next_obs, _, done, _, _, _ = env.step(action)
        obs = torch.tensor(env.reset() if done else next_obs).unsqueeze(0).to(device)
        time.sleep(0.2)

if __name__ == "__main__":
    train()
    play_demo()
