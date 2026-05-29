import random
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Tuple] = []
        self.priorities: List[float] = []
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ):
        max_priority = max(self.priorities, default=1.0)
        transition = (state, action, reward, next_state, terminated)

        if self.size < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        if self.size < batch_size:
            batch_size = self.size

        priorities = np.array(self.priorities[: self.size])
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6

    def __len__(self) -> int:
        return self.size


class NStepReplayBuffer:
    def __init__(self, capacity: int, n_step: int, gamma: float):
        self.n_step_buffer = deque(maxlen=n_step)
        self.replay_buffer = ReplayBuffer(capacity)
        self.n_step = n_step
        self.gamma = gamma

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ):
        self.n_step_buffer.append((state, action, reward, next_state, terminated))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, terminated = self._get_n_step_info()
            self.replay_buffer.push(state, action, reward, next_state, terminated)

    def _get_n_step_info(self) -> Tuple:
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        reward = 0.0
        terminated = False

        for i, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * r
            if d:
                next_state = ns
                terminated = True
                break
            next_state = ns

        return state, action, reward, next_state, terminated

    def sample(self, batch_size: int) -> Tuple:
        return self.replay_buffer.sample(batch_size)

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def flush(self):
        while len(self.n_step_buffer) > 1:
            self.n_step_buffer.popleft()
            if len(self.n_step_buffer) >= 1:
                state, action, reward, next_state, terminated = self._get_n_step_info()
                self.replay_buffer.push(state, action, reward, next_state, terminated)


class NStepPrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        n_step: int,
        gamma: float,
        alpha: float = 0.6,
    ):
        self.prioritized = PrioritizedReplayBuffer(capacity, alpha)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ):
        self.n_step_buffer.append((state, action, reward, next_state, terminated))
        if len(self.n_step_buffer) == self.n_step:
            n_state, n_action, n_reward, n_next_state, n_done = self._compute_n_step()
            self.prioritized.push(n_state, n_action, n_reward, n_next_state, n_done)

    def _compute_n_step(self) -> Tuple:
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        reward = 0.0
        next_state = None
        terminated = False

        for i, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * r
            next_state = ns
            if d:
                terminated = True
                break

        return state, action, reward, next_state, terminated

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        return self.prioritized.sample(batch_size, beta)

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        self.prioritized.update_priorities(indices, priorities)

    def __len__(self) -> int:
        return len(self.prioritized)
