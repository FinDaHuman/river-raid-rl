import gymnasium as gym
import numpy as np


class FuelTrackingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.fuel_ram_addr = 0x7E
        self.prev_fuel = 0.0
        self.fuel_history = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_fuel = self._read_fuel()
        self.fuel_history = [self.prev_fuel]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_fuel = self._read_fuel()
        fuel_change = current_fuel - self.prev_fuel
        self.fuel_history.append(current_fuel)
        info["fuel_level"] = current_fuel
        info["fuel_change"] = fuel_change
        fuel_bonus = 0.0
        if fuel_change > 0:
            fuel_bonus = 1.0
        if fuel_change < -1:
            fuel_bonus = -0.5
        self.prev_fuel = current_fuel
        return obs, reward + fuel_bonus, terminated, truncated, info

    def _read_fuel(self) -> float:
        ram = self.env.unwrapped.ale.getRAM()
        fuel_raw = ram[self.fuel_ram_addr]
        return fuel_raw / 255.0

    def get_fuel_level(self) -> float:
        return self._read_fuel()

    def get_fuel_history(self) -> list:
        return self.fuel_history


class HierarchicalController:
    def __init__(self, num_sub_policies: int = 2):
        self.num_sub_policies = num_sub_policies
        self.current_goal = 0
        self.goal_duration = 0
        self.max_goal_duration = 50

    def select_goal(self, fuel_level: float, frame: int) -> int:
        if self.goal_duration >= self.max_goal_duration:
            self.goal_duration = 0
            if fuel_level < 0.3:
                self.current_goal = 1
            elif fuel_level > 0.7:
                self.current_goal = 0
        self.goal_duration += 1
        return self.current_goal

    def reset(self):
        self.current_goal = 0
        self.goal_duration = 0
