import numpy as np

from riverraid_rl.agents.base import BaseAgent


class RuleBasedAgent(BaseAgent):
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.vertical_direction = 0

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if state.ndim >= 3:
            frame = state[-1]
        else:
            frame = state
        h, w = frame.shape
        player_x = self._find_player(frame)

        if player_x is None:
            return 1

        road_center = self._find_road_center(frame)

        if road_center is None:
            return 1

        direction_bias = 0
        if abs(player_x - road_center) > 3:
            if player_x < road_center:
                direction_bias = 3
            else:
                direction_bias = 4

        self.vertical_direction = (self.vertical_direction + 1) % 4
        if self.vertical_direction < 3:
            vertical_action = 2
        else:
            vertical_action = 5

        if direction_bias == 0:
            return vertical_action
        return direction_bias

    def _find_player(self, frame: np.ndarray) -> int:
        bottom_third = frame[2 * frame.shape[0] // 3 :, :]
        for col in range(bottom_third.shape[1]):
            column = bottom_third[:, col]
            if column.min() < 50:
                return col
        return None

    def _find_road_center(self, frame: np.ndarray) -> int:
        mid_row = frame.shape[0] // 2
        row_data = frame[mid_row, :]
        white_cols = np.where(row_data > 200)[0]
        if len(white_cols) == 0:
            return None
        return int(white_cols.mean())

    def update(self, *args, **kwargs):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
