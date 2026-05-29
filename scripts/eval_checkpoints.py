import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:\Vs Code\RELproject")

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig, DQNConfig, RainbowConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.agents.random_agent import RandomAgent
from riverraid_rl.agents.rule_based import RuleBasedAgent
from riverraid_rl.agents.dqn import DQNAgent
from riverraid_rl.agents.rainbow import RainbowAgent

env_cfg = EnvConfig()
results = []

# 1. Random baseline
r = evaluate(RandomAgent(6), env_cfg, 10)
results.append(("Random (untrained)", r))
print(f"Random:       mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")

# 2. Rule-based baseline
r = evaluate(RuleBasedAgent(), env_cfg, 10)
results.append(("Rule-Based", r))
print(f"Rule-Based:   mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")

# 3. Rainbow-baseline (untrained / 0 steps)
try:
    rb = RainbowAgent(env_cfg, RainbowConfig(), 6, "cpu")
    rb.load("checkpoints/rainbow-baseline/final.pt")
    r = evaluate(rb, env_cfg, 10)
    results.append(("Rainbow (untrained)", r))
    print(f"Rainbow (0 st): mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")
except Exception as e:
    print(f"Rainbow baseline load failed: {e}")

# 4. DQN 50K
try:
    dqn_cfg = DQNConfig()
    dqn = DQNAgent(env_cfg, dqn_cfg, 6, "cpu")
    dqn.load("checkpoints/dqn_50k.pt")
    r = evaluate(dqn, env_cfg, 10)
    results.append(("DQN (50K steps)", r))
    print(f"DQN (50K):    mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")
except Exception as e:
    print(f"DQN 50K load failed: {e}")

# 5. ICM-Rainbow (0 steps)
try:
    from riverraid_rl.agents.icm_rainbow import ICMRainbowAgent
    icm = ICMRainbowAgent(env_cfg, RainbowConfig(), 6, "cpu", 0.01)
    icm.load("checkpoints/icm_rainbow-0m/final.pt")
    r = evaluate(icm, env_cfg, 10)
    results.append(("ICM-Rainbow (0 st)", r))
    print(f"ICM-Rainbow:  mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")
except Exception as e:
    print(f"ICM load failed: {e}")

# 6. Rainbow-0m-1779973383 (0 steps)
try:
    rb2 = RainbowAgent(env_cfg, RainbowConfig(), 6, "cpu")
    rb2.load("checkpoints/rainbow-0m-1779973383/final.pt")
    r = evaluate(rb2, env_cfg, 10)
    results.append(("Rainbow (0 st #2)", r))
    print(f"Rainbow #2:   mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")
except Exception as e:
    print(f"Rainbow #2 load failed: {e}")

print()
print("=" * 65)
print(f"{'Agent':30s} {'Mean Reward':>10s} {'Std':>8s} {'Length':>8s}")
print("=" * 65)
for name, r in sorted(results, key=lambda x: x[1]["mean_reward"], reverse=True):
    print(f"{name:30s} {r['mean_reward']:>8.1f}  +/-{r['std_reward']:>6.1f}  {r['mean_length']:>6.1f}")
print("=" * 65)
