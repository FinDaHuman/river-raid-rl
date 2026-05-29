import sys, os, warnings
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
from riverraid_rl.agents.hierarchical import HierarchicalRiverRaidAgent

env_cfg = EnvConfig()

results = []

agents = [
    ("Random", RandomAgent(6)),
    ("Rule-Based", RuleBasedAgent()),
    ("DQN (5K)", DQNAgent(env_cfg, type('o',(),{'min_replay_size':200,'buffer_capacity':2000,'target_update_freq':200,'train_freq':1,'batch_size':8,'hidden_dim':128,'epsilon_decay_steps':3000,'learning_rate':0.0005,'max_grad_norm':10.0,'epsilon_start':1.0,'epsilon_end':0.01})(), 6, "cpu")),
    ("Rainbow (5K)", RainbowAgent(env_cfg, type('o',(),{'min_replay_size':200,'buffer_capacity':2000,'target_update_freq':200,'train_freq':1,'batch_size':8,'hidden_dim':128,'num_atoms':11,'v_min':-5,'v_max':15,'epsilon_decay_steps':3000,'learning_rate':0.0005,'max_grad_norm':10.0,'epsilon_start':1.0,'epsilon_end':0.01,'gamma':0.99,'alpha':0.6,'beta_start':0.4,'beta_frames':100000})(), 6, "cpu")),
]

dqn = DQNAgent(env_cfg, type('o',(),{'min_replay_size':200,'buffer_capacity':2000,'target_update_freq':200,'train_freq':1,'batch_size':8,'hidden_dim':128,'epsilon_decay_steps':3000,'learning_rate':0.0005,'max_grad_norm':10.0,'epsilon_start':1.0,'epsilon_end':0.01})(), 6, "cpu")
dqn.load("checkpoints/dqn_20k.pt")

rb = RainbowAgent(env_cfg, type('o',(),{'min_replay_size':200,'buffer_capacity':2000,'target_update_freq':200,'train_freq':1,'batch_size':8,'hidden_dim':128,'num_atoms':11,'v_min':-5,'v_max':15,'epsilon_decay_steps':3000,'learning_rate':0.0005,'max_grad_norm':10.0,'epsilon_start':1.0,'epsilon_end':0.01,'gamma':0.99,'alpha':0.6,'beta_start':0.4,'beta_frames':100000})(), 6, "cpu")
rb.load("checkpoints/dqn_20k.pt")

print("="*65)
print("    RIVERRAID RL - FINAL AGENT COMPARISON")
print("="*65)
print(f"{'Agent':30s} {'Reward':>8s} {'Std':>8s} {'Length':>8s}")
print("-"*65)

for name, agent in [
    ("Random (untrained)", RandomAgent(6)),
    ("Rule-Based (heuristic)", RuleBasedAgent()),
    ("DQN (5K steps)", dqn),
    ("Rainbow (5K steps)", rb),
]:
    r = evaluate(agent, env_cfg, 10)
    print(f"{name:30s} {r['mean_reward']:>7.1f} +/-{r['std_reward']:>6.1f} {r['mean_length']:>6.1f}")

print("="*65)
print(f"{'Agent':30s} {'Reward':>8s} {'Std':>8s} {'Length':>8s}")
print("-"*65)
# Quick re-evaluation for display
r1 = evaluate(RandomAgent(6), env_cfg, 10)
print(f"{'Random (untrained)':30s} {r1['mean_reward']:>7.1f} +/-{r1['std_reward']:>6.1f} {r1['mean_length']:>6.1f}")
r2 = evaluate(RuleBasedAgent(), env_cfg, 10)
print(f"{'Rule-Based (heuristic)':30s} {r2['mean_reward']:>7.1f} +/-{r2['std_reward']:>6.1f} {r2['mean_length']:>6.1f}")
r3 = evaluate(dqn, env_cfg, 10)
print(f"{'DQN (trained 20K)':30s} {r3['mean_reward']:>7.1f} +/-{r3['std_reward']:>6.1f} {r3['mean_length']:>6.1f}")
r4 = evaluate(rb, env_cfg, 10)
print(f"{'Rainbow (trained 20K)':30s} {r4['mean_reward']:>7.1f} +/-{r4['std_reward']:>6.1f} {r4['mean_length']:>6.1f}")
print("="*65)
