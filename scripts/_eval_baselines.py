import sys, warnings, os, glob
warnings.filterwarnings("ignore")
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

env_cfg = EnvConfig()
results = []

# 1. Random baseline
r = evaluate(RandomAgent(6), env_cfg, 10)
results.append(("Random", r))
print(f"Random:       mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")

# 2. Rule-based baseline
r = evaluate(RuleBasedAgent(), env_cfg, 10)
results.append(("Rule-Based", r))
print(f"Rule-Based:   mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")

# 3. DQN 50K checkpoint
try:
    dqn = DQNAgent(env_cfg, DQNConfig(), 6, "cpu")
    dqn.load("checkpoints/dqn_50k.pt")
    r = evaluate(dqn, env_cfg, 10)
    results.append(("DQN (50K st)", r))
    print(f"DQN (50K st):  mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")
except Exception as e:
    print(f"DQN 50K load failed: {e}")

# 4. Rainbow baseline
try:
    rb = RainbowAgent(env_cfg, RainbowConfig(), 6, "cpu")
    rb.load("checkpoints/rainbow-baseline/final.pt")
    r = evaluate(rb, env_cfg, 10)
    results.append(("Rainbow (base)", r))
    print(f"Rainbow (base): mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")
except Exception as e:
    print(f"Rainbow baseline load failed: {e}")

# 5. All rainbow-0m checkpoints
checkpoint_dirs = sorted(glob.glob("checkpoints/rainbow-0m-*"))
for d in checkpoint_dirs:
    name = os.path.basename(d)
    for ckpt_file in ["best.pt", "final.pt"]:
        path = os.path.join(d, ckpt_file)
        if os.path.exists(path):
            try:
                rb2 = RainbowAgent(env_cfg, RainbowConfig(), 6, "cpu")
                rb2.load(path)
                r = evaluate(rb2, env_cfg, 5)
                results.append((f"Rainbow ({name[:20]})", r))
                print(f"Rainbow ({name}): mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")
                break
            except Exception as e2:
                print(f"  {path} load failed: {e2}")

# 6. ICM-Rainbow
try:
    icm = ICMRainbowAgent(env_cfg, RainbowConfig(), 6, "cpu", 0.01)
    icm.load("checkpoints/icm_rainbow-0m/final.pt")
    r = evaluate(icm, env_cfg, 5)
    results.append(("ICM-Rainbow", r))
    print(f"ICM-Rainbow:   mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")
except Exception as e:
    print(f"ICM load failed: {e}")

# 7. Rainbow CPU trained
try:
    rb_cpu = RainbowAgent(env_cfg, RainbowConfig(), 6, "cpu")
    rb_cpu.load("checkpoints/rainbow-cpu/final.pt")
    r = evaluate(rb_cpu, env_cfg, 10)
    results.append(("Rainbow-CPU", r))
    print(f"Rainbow-CPU:   mean={r['mean_reward']:.1f}  std={r['std_reward']:.1f}  len={r['mean_length']:.1f}")
except Exception as e:
    print(f"Rainbow-CPU load failed: {e}")

print()
print("=" * 65)
print(f"{'Agent':25s} {'Mean Reward':>10s} {'Std':>8s} {'Length':>8s}")
print("=" * 65)
for name, r in sorted(results, key=lambda x: x[1]["mean_reward"], reverse=True):
    print(f"{name:25s} {r['mean_reward']:>8.1f}  +/-{r['std_reward']:>6.1f}  {r['mean_length']:>6.1f}")
print("=" * 65)
