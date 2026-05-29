import random

import numpy as np

from riverraid_rl.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, num_actions: int, seed: int = 42):
        self.num_actions = num_actions
        self.rng = random.Random(seed)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        return self.rng.randint(0, self.num_actions - 1)

    def update(self, *args, **kwargs):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
