import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:\Vs Code\RELproject")

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig, DQNConfig
from riverraid_rl.agents.dqn import DQNAgent

env_cfg = EnvConfig()

dqn_cfg = DQNConfig()
dqn_cfg.min_replay_size = 200
dqn_cfg.buffer_capacity = 1000
dqn_cfg.target_update_freq = 100
dqn_cfg.train_freq = 1
dqn_cfg.batch_size = 4
dqn_cfg.hidden_dim = 64
dqn_cfg.epsilon_decay_steps = 2000
dqn_cfg.learning_rate = 0.0005

dqn = DQNAgent(env_cfg, dqn_cfg, 6, "cpu")
print(f"Params: {sum(p.numel() for p in dqn.q_network.parameters())}")

env = make_riverraid_env(env_cfg)
state, info = env.reset()

t0 = time.time()
for step in range(3000):
    action = dqn.act(np.array(state), training=True)
    ns, r, term, trunc, info = env.step(action)
    dqn.memory.push(state, action, r, ns, term or trunc)
    state = ns
    if step >= dqn_cfg.min_replay_size and step % dqn_cfg.train_freq == 0:
        dqn.update()
    if term or trunc:
        state, info = env.reset()

elapsed = time.time() - t0
print(f"3000 steps in {elapsed:.1f}s ({3000/elapsed:.0f} steps/s)")
env.close()
