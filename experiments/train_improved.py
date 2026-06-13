"""Improved training for Rainbow DQN on River Raid."""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.agents.rainbow import RainbowAgent

# Compact CPU config
rb_cfg = RainbowConfig()
rb_cfg.hidden_dim = 256
rb_cfg.num_atoms = 51
rb_cfg.v_min = -5
rb_cfg.v_max = 30
rb_cfg.min_replay_size = 5000
rb_cfg.buffer_capacity = 100000
rb_cfg.batch_size = 32
rb_cfg.target_update_freq = 2500
rb_cfg.train_freq = 4
rb_cfg.learning_rate = 0.00025
rb_cfg.gamma = 0.99
rb_cfg.epsilon_decay_steps = 100000
rb_cfg.beta_frames = 100000
rb_cfg.n_step = 3

TOTAL_STEPS = 200000
EVAL_FREQ = 25000
EVAL_EPISODES = 10
env_cfg = EnvConfig()

print("=" * 60)
print("RIVER RAID - IMPROVED RAINBOW")
print(f"Steps: {TOTAL_STEPS:,}  |  N-Step: {rb_cfg.n_step}  |  Atoms: {rb_cfg.num_atoms}")
print(f"Buffer: {rb_cfg.buffer_capacity}  |  MinReplay: {rb_cfg.min_replay_size}")
print(f"Reward clipping: DISABLED")
print("=" * 60)

agent = RainbowAgent(env_cfg, rb_cfg, 6, "cpu")
agent.q_network.train()
print(f"Params: {sum(p.numel() for p in agent.q_network.parameters()):,}")
print()

env = make_riverraid_env(env_cfg, clip_rewards=False)
state, info = env.reset()
episode_num = 0
best_mean_reward = float("-inf")
t_start = time.time()
last_log = 0

for step in range(TOTAL_STEPS):
    action = agent.act(np.array(state), training=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    agent.memory.push(state, action, reward, next_state, terminated or truncated)
    state = next_state

    if step >= rb_cfg.min_replay_size and step % rb_cfg.train_freq == 0:
        metrics = agent.update()

    # Always show progress every 500 steps
    if step - last_log >= 500 or step == TOTAL_STEPS - 1:
        elapsed = time.time() - t_start
        sps = (step + 1) / elapsed if elapsed > 0 else 0
        pct = 100 * (step + 1) / TOTAL_STEPS
        eta = (TOTAL_STEPS - step) / sps if sps > 0 else 0
        mem_size = len(agent.memory)
        status = f"[{step:>6}/{TOTAL_STEPS}] {pct:5.1f}% | {sps:4.0f} sps | ETA: {eta:4.0f}s | buf={mem_size:>5}"
        if step >= rb_cfg.min_replay_size and metrics:
            status += f" | loss={metrics['loss']:.3f} q={metrics['q_value']:.1f} eps={metrics['epsilon']:.3f}"
        else:
            status += f" | (warming up, eps={agent.epsilon:.3f})"
        print(status)
        last_log = step

    if terminated or truncated:
        state, info = env.reset()
        episode_num += 1

    if step > 0 and step % EVAL_FREQ == 0:
        agent.q_network.eval()
        result = evaluate(agent, env_cfg, EVAL_EPISODES)
        agent.q_network.train()
        mean = result["mean_reward"]
        elapsed = time.time() - t_start
        print(f"\n*** EVAL step {step:,} ({elapsed:.0f}s) ***")
        print(f"    Mean: {mean:.1f} +/- {result['std_reward']:.1f}"
              f"  Min: {result['min_reward']:.1f}  Max: {result['max_reward']:.1f}")
        if mean > best_mean_reward:
            best_mean_reward = mean
            os.makedirs("checkpoints/rainbow-improved", exist_ok=True)
            agent.save("checkpoints/rainbow-improved/best.pt")
            print(f"    *** NEW BEST ({best_mean_reward:.1f}) ***")
        print()

env.close()
elapsed_total = time.time() - t_start
print()
print("=" * 60)
print(f"DONE: {TOTAL_STEPS:,} steps in {elapsed_total:.0f}s ({TOTAL_STEPS/elapsed_total:.0f} sps)")
print(f"Episodes: {episode_num}  |  Best eval: {best_mean_reward:.1f}")
print("=" * 60)

print("\nFinal evaluation...")
agent.q_network.eval()
final = evaluate(agent, env_cfg, 20)
print(f"\nFinal (20 eps): mean={final['mean_reward']:.1f} +/- {final['std_reward']:.1f}"
      f"  min={final['min_reward']:.1f}  max={final['max_reward']:.1f}")
agent.q_network.train()
os.makedirs("checkpoints/rainbow-improved", exist_ok=True)
agent.save("checkpoints/rainbow-improved/final.pt")
print("Saved: checkpoints/rainbow-improved/final.pt")
