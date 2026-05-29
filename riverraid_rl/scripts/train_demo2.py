import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:\Vs Code\RELproject")

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.agents.rainbow import RainbowAgent

env_cfg = EnvConfig()
N = 5000

r_cfg = RainbowConfig()
r_cfg.min_replay_size = 200
r_cfg.buffer_capacity = 2000
r_cfg.target_update_freq = 200
r_cfg.train_freq = 1
r_cfg.batch_size = 8
r_cfg.hidden_dim = 128
r_cfg.num_atoms = 11
r_cfg.v_min = -5
r_cfg.v_max = 15
r_cfg.epsilon_decay_steps = 3000
r_cfg.learning_rate = 0.0005

print("="*55)
print("RAINBOW TRAINING DEMO")
print("="*55)

rb = RainbowAgent(env_cfg, r_cfg, 6, "cpu")
before = evaluate(rb, env_cfg, 5)
print(f"Before: {before['mean_reward']:.1f}")

env = make_riverraid_env(env_cfg)
state, info = env.reset()
t0 = time.time()
for step in range(N):
    action = rb.act(np.array(state), training=True)
    ns, r, term, trunc, info = env.step(action)
    rb.memory.push(state, action, r, ns, term or trunc)
    state = ns
    if step >= r_cfg.min_replay_size and step % r_cfg.train_freq == 0:
        rb.update()
    if step % 1000 == 0:
        print(f"  step {step:>4}")
    if term or trunc:
        state, info = env.reset()
env.close()
elapsed = time.time() - t0
print(f"Trained: {elapsed:.0f}s ({N/elapsed:.0f} steps/s)")

after = evaluate(rb, env_cfg, 10)
print(f"After:  {after['mean_reward']:.1f} +/- {after['std_reward']:.1f}")
print(f"Improvement: {after['mean_reward']-before['mean_reward']:+.1f}")
print("="*55)
