from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EnvConfig:
    env_id: str = "ALE/Riverraid-v5"
    frame_stack: int = 4
    frame_skip: int = 4
    screen_size: int = 84
    grayscale: bool = True
    noop_max: int = 30
    repeat_action_probability: float = 0.0
    max_episode_steps: int = 108000


@dataclass
class DQNConfig:
    learning_rate: float = 0.00025
    batch_size: int = 32
    buffer_capacity: int = 100000
    min_replay_size: int = 50000
    target_update_freq: int = 10000
    train_freq: int = 4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 250000
    max_grad_norm: float = 10.0
    hidden_dim: int = 512


@dataclass
class RainbowConfig(DQNConfig):
    v_min: float = -10.0
    v_max: float = 10.0
    num_atoms: int = 51
    n_step: int = 3
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100000


@dataclass
class TrainingConfig:
    total_timesteps: int = 10000000
    eval_freq: int = 250000
    eval_episodes: int = 10
    log_freq: int = 1000
    video_freq: int = 50000
    save_freq: int = 250000
    seed: int = 42
    num_envs: int = 1
    device: str = "cpu"
    project_name: str = "riverraid-rl"
    run_name: str = "rainbow-baseline"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    rainbow: RainbowConfig = field(default_factory=RainbowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
