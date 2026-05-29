import sys, os
sys.path.insert(0, r"D:\Vs Code\RELproject")

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig, DQNConfig, RainbowConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.agents.random_agent import RandomAgent
from riverraid_rl.agents.rule_based import RuleBasedAgent
from riverraid_rl.agents.dqn import DQNAgent
from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.agents.icm_rainbow import ICMRainbowAgent
from riverraid_rl.agents.hierarchical import HierarchicalRiverRaidAgent

env_cfg = EnvConfig()

agents = [
    ("Random", RandomAgent(6)),
    ("Rule-Based", RuleBasedAgent()),
    ("DQN (untrained)", DQNAgent(env_cfg, DQNConfig(), 6, "cpu")),
    ("Rainbow (untrained)", RainbowAgent(env_cfg, RainbowConfig(), 6, "cpu")),
    ("ICM-Rainbow (untrained)", ICMRainbowAgent(env_cfg, RainbowConfig(), 6, "cpu")),
    ("Hierarchical (untrained)", HierarchicalRiverRaidAgent(env_cfg, RainbowConfig(), 6, "cpu")),
]

print("=" * 70)
print(f"{'Agent':30s} {'Mean Reward':>10s} {'Std':>8s} {'Length':>8s}")
print("=" * 70)
for name, agent in agents:
    r = evaluate(agent, env_cfg, 5)
    print(f"{name:30s} {r['mean_reward']:8.1f}  +/-{r['std_reward']:6.1f}  {r['mean_length']:6.1f}")
print("=" * 70)
