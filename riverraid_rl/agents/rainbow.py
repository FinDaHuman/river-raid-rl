import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from riverraid_rl.agents.base import BaseAgent
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.memory.replay import NStepPrioritizedReplayBuffer
from riverraid_rl.models.cnn import CategoricalDuelingDQN


class RainbowAgent(BaseAgent):
    def __init__(
        self,
        env_config: EnvConfig,
        rainbow_config: RainbowConfig,
        num_actions: int,
        device: str = "cpu",
    ):
        self.config = rainbow_config
        self.num_actions = num_actions
        self.device = torch.device(device)

        input_shape = (env_config.frame_stack, env_config.screen_size, env_config.screen_size)
        self.q_network = CategoricalDuelingDQN(
            input_shape, num_actions, rainbow_config.num_atoms, rainbow_config.hidden_dim
        ).to(self.device)
        self.target_network = CategoricalDuelingDQN(
            input_shape, num_actions, rainbow_config.num_atoms, rainbow_config.hidden_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=rainbow_config.learning_rate, eps=1.5e-4)
        self.memory = NStepPrioritizedReplayBuffer(
            rainbow_config.buffer_capacity,
            rainbow_config.n_step,
            rainbow_config.gamma,
            rainbow_config.alpha,
        )

        self.atoms = torch.linspace(rainbow_config.v_min, rainbow_config.v_max, rainbow_config.num_atoms).to(self.device)
        self.delta_z = (rainbow_config.v_max - rainbow_config.v_min) / (rainbow_config.num_atoms - 1)
        self.n_step = rainbow_config.n_step

        self.steps = 0
        self.beta = rainbow_config.beta_start
        self.epsilon = rainbow_config.epsilon_start

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.q_network(state_t)
            q_values = (dist * self.atoms).sum(dim=2)
        return q_values.argmax(dim=1).item()

    def _project_distribution(
        self, next_dist: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        batch_size = next_dist.size(0)
        atoms = self.atoms.unsqueeze(0).expand(batch_size, -1)
        gamma_n = self.config.gamma ** self.n_step
        target_z = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma_n * atoms
        target_z = target_z.clamp(self.config.v_min, self.config.v_max)

        b = (target_z - self.config.v_min) / self.delta_z
        lower = b.floor().long().clamp(0, self.config.num_atoms - 1)
        upper = b.ceil().long().clamp(0, self.config.num_atoms - 1)

        same_atom = lower == upper
        lower_weight = torch.where(same_atom, torch.ones_like(b), upper.float() - b).clamp(0, 1)
        upper_weight = torch.where(same_atom, torch.zeros_like(b), b - lower.float()).clamp(0, 1)

        projected = torch.zeros(batch_size, self.config.num_atoms, device=next_dist.device)
        projected.scatter_add_(1, lower, next_dist * lower_weight)
        projected.scatter_add_(1, upper, next_dist * upper_weight)

        return projected

    def update(self) -> Optional[dict]:
        if len(self.memory) < self.config.min_replay_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(
            self.config.batch_size, self.beta
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        current_dist = self.q_network(states)
        current_dist = current_dist.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.config.num_atoms)).squeeze(1)

        with torch.no_grad():
            next_dist = self.q_network(next_states)
            next_q = (next_dist * self.atoms).sum(dim=2)
            next_actions = next_q.argmax(dim=1)

            next_dist_target = self.target_network(next_states)
            next_dist_target = next_dist_target.gather(
                1, next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.config.num_atoms)
            ).squeeze(1)

            target_dist = self._project_distribution(next_dist_target, rewards, dones)

        loss = -(target_dist * current_dist.log()).sum(dim=1)
        loss = (loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            td_error = (target_dist - current_dist).abs().sum(dim=1).cpu().numpy()
        self.memory.update_priorities(indices, td_error)

        self.steps += 1
        self._update_epsilon()
        self._update_beta()

        if self.steps % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        with torch.no_grad():
            q_val = (current_dist * self.atoms).sum(dim=1).mean().item()

        return {
            "loss": loss.item(),
            "q_value": q_val,
            "epsilon": self.epsilon,
            "beta": self.beta,
        }

    def _update_epsilon(self):
        decay = self.config.epsilon_decay_steps
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - (self.config.epsilon_start - self.config.epsilon_end) * self.steps / decay,
        )

    def _update_beta(self):
        self.beta = min(
            1.0,
            self.config.beta_start + (1.0 - self.config.beta_start) * self.steps / self.config.beta_frames,
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
                "beta": self.beta,
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
        self.beta = checkpoint["beta"]
