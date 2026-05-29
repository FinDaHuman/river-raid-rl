import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from riverraid_rl.agents.base import BaseAgent
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.env_hierarchical import FuelTrackingWrapper, HierarchicalController
from riverraid_rl.memory.replay import PrioritizedReplayBuffer, NStepReplayBuffer
from riverraid_rl.models.cnn import CategoricalDuelingDQN
from riverraid_rl.models.attention import AttentionDQN
from riverraid_rl.models.icm import IntrinsicCuriosityModule


class HierarchicalRiverRaidAgent(BaseAgent):
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

        self.nav_network = AttentionDQN(input_shape, num_actions, rainbow_config.hidden_dim).to(self.device)
        self.fuel_network = AttentionDQN(input_shape, num_actions, rainbow_config.hidden_dim).to(self.device)
        self.meta_network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        ).to(self.device)

        self.target_nav = AttentionDQN(input_shape, num_actions, rainbow_config.hidden_dim).to(self.device)
        self.target_fuel = AttentionDQN(input_shape, num_actions, rainbow_config.hidden_dim).to(self.device)
        self.target_nav.load_state_dict(self.nav_network.state_dict())
        self.target_fuel.load_state_dict(self.fuel_network.state_dict())
        self.target_nav.eval()
        self.target_fuel.eval()

        self.icm = IntrinsicCuriosityModule(input_shape, num_actions, feature_dim=288, hidden_dim=256).to(self.device)

        self.optimizer = optim.Adam(
            list(self.nav_network.parameters())
            + list(self.fuel_network.parameters())
            + list(self.meta_network.parameters())
            + list(self.icm.parameters()),
            lr=rainbow_config.learning_rate,
            eps=1.5e-4,
        )

        self.memory = PrioritizedReplayBuffer(rainbow_config.buffer_capacity, rainbow_config.alpha)
        self.meta_memory = PrioritizedReplayBuffer(50000, rainbow_config.alpha)

        self.controller = HierarchicalController()
        self.fuel_levels = []
        self.positions = []
        self.steps = 0
        self.beta = rainbow_config.beta_start

    def act(self, state: np.ndarray, training: bool = True) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal = self.controller.current_goal

        if goal == 0:
            network = self.nav_network
        else:
            network = self.fuel_network

        with torch.no_grad():
            q_values = network(state_t)
        return q_values.argmax(dim=1).item()

    def update_meta(
        self, fuel_level: float, enemy_density: float, score: float
    ) -> int:
        features = np.array([
            fuel_level,
            enemy_density,
            score / 1000.0,
            self.controller.goal_duration / 50.0,
            self.steps / 10000.0,
            np.sin(self.steps * 0.01),
            np.cos(self.steps * 0.01),
            self.controller.current_goal,
        ], dtype=np.float32)
        features_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.meta_network(features_t)
        return logits.argmax(dim=1).item()

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

        actions_onehot = F.one_hot(actions_t.squeeze(1), num_classes=self.num_actions).float()
        icm_loss, intrinsic_rewards, icm_metrics = self.icm(states_t, next_states_t, actions_onehot)
        combined_rewards = rewards_t + self.eta * intrinsic_rewards.detach()

        nav_q = self.nav_network(states_t)
        fuel_q = self.fuel_network(states_t)
        current_q = torch.where(
            self.controller.current_goal == 0, nav_q, fuel_q
        ).gather(1, actions_t).squeeze()

        with torch.no_grad():
            nav_next = self.target_nav(next_states_t)
            fuel_next = self.target_fuel(next_states_t)
            next_q = torch.where(
                self.controller.current_goal == 0, nav_next, fuel_next
            ).max(dim=1)[0]
            target_q = combined_rewards + self.config.gamma * next_q * (1 - dones_t)

        loss = F.mse_loss(current_q, target_q) + icm_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.nav_network.parameters())
            + list(self.fuel_network.parameters())
            + list(self.icm.parameters()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        with torch.no_grad():
            td_error = (target_q - current_q).abs().cpu().numpy()
        self.memory.update_priorities(indices, td_error)

        self.steps += 1
        self._update_beta()

        if self.steps % self.config.target_update_freq == 0:
            self.target_nav.load_state_dict(self.nav_network.state_dict())
            self.target_fuel.load_state_dict(self.fuel_network.state_dict())

        return {
            "loss": loss.item(),
            "q_value": current_q.mean().item(),
            "icm_loss": icm_loss.item(),
            "goal": self.controller.current_goal,
            "beta": self.beta,
        }

    def _update_beta(self):
        self.beta = min(
            1.0,
            self.config.beta_start + (1.0 - self.config.beta_start) * self.steps / self.config.beta_frames,
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "nav_network": self.nav_network.state_dict(),
            "fuel_network": self.fuel_network.state_dict(),
            "meta_network": self.meta_network.state_dict(),
            "icm": self.icm.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "beta": self.beta,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.nav_network.load_state_dict(checkpoint["nav_network"])
        self.fuel_network.load_state_dict(checkpoint["fuel_network"])
        self.meta_network.load_state_dict(checkpoint["meta_network"])
        self.icm.load_state_dict(checkpoint["icm"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
        self.beta = checkpoint["beta"]
