from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> int:
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
