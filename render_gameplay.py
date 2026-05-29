from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np

from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.env import (
    EpisodicLifeEnv,
    FireResetEnv,
    FrameStackObservation,
    GrayscaleObservation,
    MaxAndSkipEnv,
    ResizeObservation,
)


def make_env(config: EnvConfig):
    gym.register_envs(ale_py)
    env = gym.make(
        config.env_id,
        render_mode="rgb_array",
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
    env = FrameStackObservation(env, stack_size=config.frame_stack)
    return env


def main():
    checkpoint_path = Path("checkpoints/rainbow-200k/best.pt")
    if not checkpoint_path.exists():
        alt = list(Path("checkpoints").rglob("best.pt"))
        if alt:
            checkpoint_path = alt[0]
        else:
            print("No checkpoint found. Train first: python train_riverraid.py --steps 200000")
            return

    env_config = EnvConfig()
    rainbow_config = RainbowConfig()
    rainbow_config.hidden_dim = 256
    rainbow_config.num_atoms = 51
    rainbow_config.v_min = -5
    rainbow_config.v_max = 50
    rainbow_config.min_replay_size = 5000
    rainbow_config.buffer_capacity = 200000

    agent = RainbowAgent(env_config, rainbow_config, num_actions=6, device="cpu")
    agent.load(str(checkpoint_path))
    agent.q_network.eval()
    print(f"Loaded: {checkpoint_path}")

    raw_env = make_env(env_config)
    fps = 30
    out_path = Path("media/riverraid-best.mp4")
    out_path.parent.mkdir(exist_ok=True)
    writer = None
    total_reward = 0
    state, info = raw_env.reset()
    step = 0
    terminated = truncated = False

    while not (terminated or truncated):
        rgb = raw_env.render()
        if writer is None:
            h, w = rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        action = agent.act(np.array(state), training=False)
        state, reward, terminated, truncated, info = raw_env.step(action)
        total_reward += reward
        step += 1

    raw_env.close()
    if writer:
        writer.release()

    print(f"Episode: score={total_reward}, steps={step}")
    print(f"Video: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
