import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


def _copy_frame(frame: np.ndarray) -> np.ndarray:
    return np.asarray(frame, dtype=np.uint8).copy()


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
        self.buffer.append((_copy_frame(state), action, np.float32(reward), _copy_frame(next_state), bool(terminated)))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.uint8, copy=False),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.uint8, copy=False),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, frame_shape: tuple = None):
        self.capacity = capacity
        self.alpha = alpha
        self.frame_shape = frame_shape
        if frame_shape is not None:
            self.states = np.empty((capacity, *frame_shape), dtype=np.uint8)
            self.next_states = np.empty((capacity, *frame_shape), dtype=np.uint8)
        else:
            self.states = None
            self.next_states = None
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
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
        if self.states is None and self.frame_shape is None:
            self.frame_shape = state.shape
            self.states = np.empty((self.capacity, *self.frame_shape), dtype=np.uint8)
            self.next_states = np.empty((self.capacity, *self.frame_shape), dtype=np.uint8)

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = terminated

        max_priority = float(self.priorities[: self.size].max()) if self.size else 1.0
        self._set_priority(self.position, max_priority)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        if self.size < batch_size:
            batch_size = self.size

        total_priority = float(self.tree[1])
        if total_priority <= 0:
            indices = np.random.randint(0, self.size, size=batch_size)
            probs = np.full(batch_size, 1.0 / self.size, dtype=np.float32)
        else:
            segment = total_priority / batch_size
            samples = (np.arange(batch_size) + np.random.random(batch_size)) * segment
            indices = np.array([self._retrieve(sample) for sample in samples], dtype=np.int64)
            probs = self.tree[indices + self.capacity] / total_priority
            probs = np.maximum(probs, 1e-12)

        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices].astype(np.float32),
            indices.astype(np.int64),
            weights,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self._set_priority(int(idx), float(priority) + 1e-6)

    def _set_priority(self, idx: int, priority: float):
        priority = max(priority, 1e-6)
        self.priorities[idx] = priority

        tree_idx = idx + self.capacity
        scaled_priority = priority ** self.alpha
        change = scaled_priority - self.tree[tree_idx]
        while tree_idx >= 1:
            self.tree[tree_idx] += change
            tree_idx //= 2

    def _retrieve(self, sample: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if sample <= self.tree[left]:
                idx = left
            else:
                sample -= self.tree[left]
                idx = left + 1
        return min(idx - self.capacity, self.size - 1)

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
        frame_shape: tuple = None,
    ):
        self.prioritized = PrioritizedReplayBuffer(capacity, alpha, frame_shape)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step_buffers: Dict[int, Deque[Tuple]] = {}
        self.n_step = n_step
        self.gamma = gamma

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        env_id: Optional[int] = None,
    ):
        buffer = self._buffer_for(env_id)
        buffer.append((state, action, reward, next_state, terminated))

        if len(buffer) == self.n_step:
            self._push_computed(buffer)

        if terminated:
            if len(buffer) < self.n_step:
                self._push_computed(buffer)
            while len(buffer) > 1:
                buffer.popleft()
                self._push_computed(buffer)
            buffer.clear()

    def _buffer_for(self, env_id: Optional[int]) -> Deque[Tuple]:
        if env_id is None:
            return self.n_step_buffer
        if env_id not in self.n_step_buffers:
            self.n_step_buffers[env_id] = deque(maxlen=self.n_step)
        return self.n_step_buffers[env_id]

    def _push_computed(self, buffer: Deque[Tuple]):
        if not buffer:
            return
        n_state, n_action, n_reward, n_next_state, n_done = self._compute_n_step(buffer)
        self.prioritized.push(n_state, n_action, n_reward, n_next_state, n_done)

    def _compute_n_step(self, buffer: Optional[Deque[Tuple]] = None) -> Tuple:
        buffer = buffer or self.n_step_buffer
        state = buffer[0][0]
        action = buffer[0][1]
        reward = 0.0
        next_state = None
        terminated = False

        for i, (_, _, r, ns, d) in enumerate(buffer):
            reward += (self.gamma ** i) * r
            next_state = ns
            if d:
                terminated = True
                break

        return state, action, np.float32(reward), next_state, terminated

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        return self.prioritized.sample(batch_size, beta)

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        self.prioritized.update_priorities(indices, priorities)

    def __len__(self) -> int:
        return len(self.prioritized)
