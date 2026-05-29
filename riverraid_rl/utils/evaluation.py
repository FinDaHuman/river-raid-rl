import numpy as np

from riverraid_rl.agents.base import BaseAgent
from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig


def evaluate(agent: BaseAgent, env_config: EnvConfig, num_episodes: int = 10, render: bool = False) -> dict:
    env = make_riverraid_env(env_config, clip_rewards=False)
    episode_rewards = []
    episode_lengths = []
    episode_fuels = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0.0
        episode_length = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(np.array(state), training=False)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_length += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

    env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "num_episodes": num_episodes,
    }
