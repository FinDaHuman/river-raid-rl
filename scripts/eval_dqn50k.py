import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:\Vs Code\RELproject")

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig, DQNConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.agents.dqn import DQNAgent

env_cfg = EnvConfig()
dqn_cfg = DQNConfig()
dqn_cfg.hidden_dim = 256
dqn_cfg.min_replay_size = 0

dqn = DQNAgent(env_cfg, dqn_cfg, 6, "cpu")
dqn.load("checkpoints/dqn_50k.pt")
print(f"Loaded: steps={dqn.steps}, epsilon={dqn.epsilon:.4f}")

r = evaluate(dqn, env_cfg, 10)
print(f"DQN (24K steps): mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  min={r['min_reward']:.1f}  max={r['max_reward']:.1f}  len={r['mean_length']:.1f}")
