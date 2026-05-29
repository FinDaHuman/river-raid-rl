import sys, os, time, warnings
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
from riverraid_rl.agents.random_agent import RandomAgent
from riverraid_rl.agents.rule_based import RuleBasedAgent

env_cfg = EnvConfig()
N = 5000

dqn_cfg = DQNConfig()
dqn_cfg.min_replay_size = 200
dqn_cfg.buffer_capacity = 2000
dqn_cfg.target_update_freq = 200
dqn_cfg.train_freq = 1
dqn_cfg.batch_size = 8
dqn_cfg.hidden_dim = 128
dqn_cfg.epsilon_decay_steps = 3000
dqn_cfg.learning_rate = 0.0005

print("="*55)
print("RIVERRAID RL - QUICK DEMO")
print("="*55)

r = evaluate(RandomAgent(6), env_cfg, 10)
print(f"Random:     {r['mean_reward']:.1f} +/- {r['std_reward']:.1f}")
r = evaluate(RuleBasedAgent(), env_cfg, 10)
print(f"Rule-Based: {r['mean_reward']:.1f} +/- {r['std_reward']:.1f}")

dqn = DQNAgent(env_cfg, dqn_cfg, 6, "cpu")
before = evaluate(dqn, env_cfg, 5)
print(f"\nDQN before: {before['mean_reward']:.1f}")

env = make_riverraid_env(env_cfg)
state, info = env.reset()
t0 = time.time()
for step in range(N):
    action = dqn.act(np.array(state), training=True)
    ns, r, term, trunc, info = env.step(action)
    dqn.memory.push(state, action, r, ns, term or trunc)
    state = ns
    if step >= dqn_cfg.min_replay_size and step % dqn_cfg.train_freq == 0:
        dqn.update()
    if step % 1000 == 0:
        print(f"  step {step:>4} | eps={dqn.epsilon:.3f}")
    if term or trunc:
        state, info = env.reset()
env.close()
elapsed = time.time() - t0
print(f"Trained: {elapsed:.0f}s ({N/elapsed:.0f} steps/s)")

after = evaluate(dqn, env_cfg, 10)
print(f"\nDQN after:  {after['mean_reward']:.1f} +/- {after['std_reward']:.1f}")
print(f"Improvement: {after['mean_reward']-before['mean_reward']:+.1f}")
print("="*55)
