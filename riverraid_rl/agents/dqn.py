import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from riverraid_rl.agents.base import BaseAgent
from riverraid_rl.config import DQNConfig, EnvConfig
from riverraid_rl.memory.replay import ReplayBuffer
from riverraid_rl.models.cnn import DQNCNN


class DQNAgent(BaseAgent):
    def __init__(
        self,
        env_config: EnvConfig,
        dqn_config: DQNConfig,
        num_actions: int,
        device: str = "cpu",
    ):
        self.config = dqn_config
        self.num_actions = num_actions
        self.device = torch.device(device)

        input_shape = (env_config.frame_stack, env_config.screen_size, env_config.screen_size)
        self.q_network = DQNCNN(input_shape, num_actions, dqn_config.hidden_dim).to(self.device)
        self.target_network = DQNCNN(input_shape, num_actions, dqn_config.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=dqn_config.learning_rate, eps=1.5e-4)
        self.memory = ReplayBuffer(dqn_config.buffer_capacity)

        self.steps = 0
        self.epsilon = dqn_config.epsilon_start

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return q_values.argmax(dim=1).item()

    def update(self) -> Optional[dict]:
        if len(self.memory) < self.config.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states).gather(1, actions).squeeze()

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + self.config.gamma * next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.steps += 1
        self._update_epsilon()

        if self.steps % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": loss.item(), "q_value": current_q.mean().item(), "epsilon": self.epsilon}

    def _update_epsilon(self):
        decay = self.config.epsilon_decay_steps
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - (self.config.epsilon_start - self.config.epsilon_end) * self.steps / decay,
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
        self.epsilon = checkpoint["epsilon"]
