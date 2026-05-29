import time
import warnings
import sys

warnings.filterwarnings("ignore")

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np

from riverraid_rl.agents.dqn import DQNAgent
from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.agents.random_agent import RandomAgent
from riverraid_rl.agents.rule_based import RuleBasedAgent
from riverraid_rl.config import EnvConfig, DQNConfig, RainbowConfig
from riverraid_rl.env import make_riverraid_env
from riverraid_rl.utils.evaluation import evaluate


def main():
    print("=" * 60)
    print("Riverraid RL - Training Demonstration")
    print("=" * 60)

    env_cfg = EnvConfig()

    # 1. Baselines
    print("\n=== 1. Baseline Evaluation ===")
    random_agent = RandomAgent(6)
    result = evaluate(random_agent, env_cfg, 5)
    print(f"  Random Agent:     {result['mean_reward']:.1f} +/- {result['std_reward']:.1f}")

    rule_agent = RuleBasedAgent()
    result = evaluate(rule_agent, env_cfg, 5)
    print(f"  Rule-Based Agent: {result['mean_reward']:.1f} +/- {result['std_reward']:.1f}")

    # 2. DQN Training (50K steps)
    print("\n=== 2. DQN Baseline Training (50K steps) ===")
    dqn_cfg = DQNConfig()
    dqn_cfg.min_replay_size = 2000
    dqn_cfg.buffer_capacity = 10000
    dqn_cfg.target_update_freq = 1000
    dqn_cfg.train_freq = 2
    dqn_cfg.batch_size = 16
    dqn_cfg.hidden_dim = 256
    dqn_cfg.epsilon_decay_steps = 30000
    dqn_cfg.learning_rate = 0.0005

    agent = DQNAgent(env_cfg, dqn_cfg, 6, "cpu")

    result_before = evaluate(agent, env_cfg, 5)
    print(f"  Before training: {result_before['mean_reward']:.1f} +/- {result_before['std_reward']:.1f}")

    env = make_riverraid_env(env_cfg)
    state, info = env.reset()
    start = time.time()
    episode_count = 0

    for step in range(50000):
        action = agent.act(np.array(state), training=True)
        ns, reward, term, trunc, info = env.step(action)
        agent.memory.push(state, action, reward, ns, term or trunc)
        state = ns
        if step >= dqn_cfg.min_replay_size and step % dqn_cfg.train_freq == 0:
            agent.update()
        if term or trunc:
            state, info = env.reset()
            episode_count += 1
        if step > 0 and step % 10000 == 0:
            print(f"  Step {step}: epsilon={agent.epsilon:.3f}")

    elapsed = time.time() - start
    print(f"  Training: {50000/elapsed:.1f} steps/s, {episode_count} episodes")
    env.close()

    result_after = evaluate(agent, env_cfg, 5)
    print(f"  After training:  {result_after['mean_reward']:.1f} +/- {result_after['std_reward']:.1f}")
    improvement = result_after["mean_reward"] - result_before["mean_reward"]
    print(f"  Improvement: {improvement:+.1f}")

    # 3. Save checkpoint
    agent.save("checkpoints/dqn-50k-demo.pt")
    print("\nCheckpoint saved to checkpoints/dqn-50k-demo.pt")

    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
