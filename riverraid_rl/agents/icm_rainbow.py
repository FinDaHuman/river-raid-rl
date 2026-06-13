import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from riverraid_rl.agents.base import BaseAgent
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.memory.replay import PrioritizedReplayBuffer
from riverraid_rl.models.cnn import CategoricalDuelingDQN
from riverraid_rl.models.icm import IntrinsicCuriosityModule


class ICMRainbowAgent(BaseAgent):
    def __init__(
        self,
        env_config: EnvConfig,
        rainbow_config: RainbowConfig,
        num_actions: int,
        device: str = "cpu",
        eta: float = 0.01,
    ):
        self.config = rainbow_config
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.eta = eta

        input_shape = (env_config.frame_stack, env_config.screen_size, env_config.screen_size)
        self.q_network = CategoricalDuelingDQN(
            input_shape, num_actions, rainbow_config.num_atoms, rainbow_config.hidden_dim
        ).to(self.device)
        self.target_network = CategoricalDuelingDQN(
            input_shape, num_actions, rainbow_config.num_atoms, rainbow_config.hidden_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.icm = IntrinsicCuriosityModule(
            input_shape, num_actions, feature_dim=288, hidden_dim=256
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.q_network.parameters()) + list(self.icm.parameters()),
            lr=rainbow_config.learning_rate,
            eps=1.5e-4,
        )
        self.memory = PrioritizedReplayBuffer(rainbow_config.buffer_capacity, rainbow_config.alpha)

        self.atoms = torch.linspace(rainbow_config.v_min, rainbow_config.v_max, rainbow_config.num_atoms).to(self.device)
        self.delta_z = (rainbow_config.v_max - rainbow_config.v_min) / (rainbow_config.num_atoms - 1)
        self.n_step = rainbow_config.n_step

        self.steps = 0
        self.beta = rainbow_config.beta_start

    def _make_onehot(self, action: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action, num_classes=self.num_actions).float()

    def act(self, state: np.ndarray, training: bool = True) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.q_network(state_t)
            q_values = (dist * self.atoms).sum(dim=2)
        return q_values.argmax(dim=1).item()

    def _project_distribution(self, next_dist, rewards, dones):
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

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        actions_onehot = self._make_onehot(actions_t.squeeze(1))

        icm_loss, intrinsic_rewards, icm_metrics = self.icm(states_t, next_states_t, actions_onehot)

        combined_rewards = rewards_t + self.eta * intrinsic_rewards.detach()

        current_dist = self.q_network(states_t)
        current_dist = current_dist.gather(
            1, actions_t.unsqueeze(-1).expand(-1, -1, self.config.num_atoms)
        ).squeeze(1)

        with torch.no_grad():
            next_dist = self.q_network(next_states_t)
            next_q = (next_dist * self.atoms).sum(dim=2)
            next_actions = next_q.argmax(dim=1)

            next_dist_target = self.target_network(next_states_t)
            next_dist_target = next_dist_target.gather(
                1, next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.config.num_atoms)
            ).squeeze(1)

            target_dist = self._project_distribution(next_dist_target, combined_rewards, dones_t)

        rl_loss = -(target_dist * current_dist.log()).sum(dim=1)
        rl_loss = (rl_loss * weights_t).mean()

        total_loss = rl_loss + icm_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.icm.parameters()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        with torch.no_grad():
            td_error = (target_dist - current_dist).abs().sum(dim=1).cpu().numpy()
        self.memory.update_priorities(indices, td_error)

        self.steps += 1
        self._update_beta()

        if self.steps % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        with torch.no_grad():
            q_val = (current_dist * self.atoms).sum(dim=1).mean().item()

        metrics = {
            "loss": total_loss.item(),
            "rl_loss": rl_loss.item(),
            "icm_loss": icm_loss.item(),
            "q_value": q_val,
            "intrinsic_reward": intrinsic_rewards.mean().item(),
            "beta": self.beta,
        }
        metrics.update(icm_metrics)
        return metrics

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
                "icm": self.icm.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "beta": self.beta,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.icm.load_state_dict(checkpoint["icm"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
        self.beta = checkpoint["beta"]
