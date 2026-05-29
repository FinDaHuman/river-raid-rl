import gymnasium as gym
import numpy as np


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, difficulty_levels: int = 5):
        super().__init__(env)
        self.difficulty_levels = difficulty_levels
        self.current_difficulty = 0
        self.episode_count = 0
        self.success_threshold = 1000

    def reset(self, **kwargs):
        self.episode_count += 1
        difficulty = self._compute_difficulty()
        self.current_difficulty = difficulty
        seed = kwargs.pop("seed", None)
        if seed is not None:
            kwargs["seed"] = seed + difficulty * 1000
        return self.env.reset(**kwargs)

    def _compute_difficulty(self) -> int:
        if self.episode_count < 100:
            return 0
        elif self.episode_count < 500:
            return 1
        elif self.episode_count < 2000:
            return 2
        elif self.episode_count < 5000:
            return 3
        else:
            return 4

    def get_difficulty(self) -> int:
        return self.current_difficulty
