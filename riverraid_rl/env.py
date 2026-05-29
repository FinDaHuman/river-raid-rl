import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

from riverraid_rl.config import EnvConfig


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        assert "FIRE" in action_meanings, f"No FIRE action found in {action_meanings}"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        fire_idx = self.env.unwrapped.get_action_meanings().index("FIRE")
        obs, _, terminated, truncated, info = self.env.step(fire_idx)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)


def make_riverraid_env(config: EnvConfig, clip_rewards: bool = True) -> gym.Env:
    import ale_py
    gym.register_envs(ale_py)

    env = gym.make(
        config.env_id,
        frameskip=1,
        repeat_action_probability=config.repeat_action_probability,
        full_action_space=False,
        max_episode_steps=config.max_episode_steps,
    )

    env = MaxAndSkipEnv(env, skip=config.frame_skip)
    env = EpisodicLifeEnv(env)
    action_meanings = env.unwrapped.get_action_meanings()
    if "FIRE" in action_meanings and action_meanings[1] == "FIRE":
        env = FireResetEnv(env)
    if config.grayscale:
        env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (config.screen_size, config.screen_size))
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStackObservation(env, stack_size=config.frame_stack)

    return env


def get_action_meanings():
    return [
        "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
        "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
        "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
        "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
    ]


def get_minimal_action_set():
    return [0, 1, 2, 3, 4, 5]
