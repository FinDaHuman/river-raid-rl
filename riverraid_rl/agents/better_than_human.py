"""Human-target Rainbow agent for River Raid.

This module defines a stronger candidate agent than the baseline Rainbow file,
but it does not guarantee human-level play without enough training data and
evaluation. It is structured for low-end PCs: compact model defaults, uint8
replay compatibility, NoisyNet exploration, prioritized n-step replay, and
teacher warm-start configuration hooks.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from riverraid_rl.agents.base import BaseAgent
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.memory.replay import NStepPrioritizedReplayBuffer
from riverraid_rl.models.noisy import NoisyLinear
from riverraid_rl.models.cnn import CategoricalDuelingDQNAttention
from riverraid_rl.models.attention import SpatialAttention, ChannelAttention


@dataclass
class BetterThanHumanRainbowConfig(RainbowConfig):
    """Training knobs for the human-target agent.

    The defaults are deliberately CPU-friendly. Human-level or better results
    require much larger training budgets than this repository has run so far.
    """

    hidden_dim: int = 256
    batch_size: int = 32
    buffer_capacity: int = 100_000
    min_replay_size: int = 5_000
    target_update_freq: int = 2_500
    train_freq: int = 4
    learning_rate: float = 0.0001
    gamma: float = 0.997
    v_min: float = -10.0
    v_max: float = 100.0
    num_atoms: int = 51
    n_step: int = 5
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 250_000
    max_grad_norm: float = 10.0
    use_noisy_nets: bool = True
    use_attention: bool = False
    teacher_warm_start_steps: int = 20_000
    human_target_score: int = 13_513


class NoisyCategoricalDuelingDQN(nn.Module):
    """Dueling C51 network with NoisyLinear exploration heads."""

    def __init__(
        self,
        input_shape: tuple,
        num_actions: int,
        num_atoms: int = 51,
        hidden_dim: int = 256,
    ):
        super().__init__()
        c, h, w = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = NoisyLinear(conv_out_size, hidden_dim)
        self.fc_val = NoisyLinear(conv_out_size, hidden_dim)
        self.advantage = NoisyLinear(hidden_dim, num_actions * num_atoms)
        self.value = NoisyLinear(hidden_dim, num_atoms)
        self._init_conv_weights()

    def _init_conv_weights(self):
        for module in self.conv.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _get_conv_out(self, shape: tuple) -> int:
        with torch.no_grad():
            output = self.conv(torch.zeros(1, *shape))
        return int(np.prod(output.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        advantage = F.relu(self.fc_adv(x))
        value = F.relu(self.fc_val(x))
        advantage = self.advantage(advantage).view(-1, self.num_actions, self.num_atoms)
        value = self.value(value).view(-1, 1, self.num_atoms)

        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=2).clamp_min(1e-6)


class NoisyCategoricalDuelingDQNAttention(NoisyCategoricalDuelingDQN):
    """Noisy C51 Dueling DQN with Spatial + Channel Attention."""

    def __init__(self, input_shape, num_actions, num_atoms=51, hidden_dim=256):
        super().__init__(input_shape, num_actions, num_atoms, hidden_dim)
        self.attn1 = SpatialAttention(32)
        self.attn2 = ChannelAttention(64)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv[0](x))
        x = self.attn1(x)
        x = F.relu(self.conv[2](x))
        x = self.attn2(x)
        x = F.relu(self.conv[4](x))
        x = x.view(x.size(0), -1)
        advantage = F.relu(self.fc_adv(x))
        value = F.relu(self.fc_val(x))
        advantage = self.advantage(advantage).view(-1, self.num_actions, self.num_atoms)
        value = self.value(value).view(-1, 1, self.num_atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=2).clamp_min(1e-6)


class BetterThanHumanRainbowAgent(BaseAgent):
    """Noisy Rainbow C51 candidate for high-score River Raid training."""

    def __init__(
        self,
        env_config: EnvConfig,
        config: BetterThanHumanRainbowConfig,
        num_actions: int,
        device: str = "cpu",
        total_steps: Optional[int] = None,
    ):
        self.config = config
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.steps = 0
        self.beta = config.beta_start

        input_shape = (env_config.frame_stack, env_config.screen_size, env_config.screen_size)
        if config.use_attention:
            network_cls = NoisyCategoricalDuelingDQNAttention
        else:
            network_cls = NoisyCategoricalDuelingDQN
        self.q_network = network_cls(
            input_shape, num_actions, config.num_atoms, config.hidden_dim
        ).to(self.device)
        self.target_network = network_cls(
            input_shape, num_actions, config.num_atoms, config.hidden_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate, eps=1.5e-4)
        if total_steps:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps // config.train_freq
            )
        else:
            self.scheduler = None
        self.memory = NStepPrioritizedReplayBuffer(
            config.buffer_capacity,
            config.n_step,
            config.gamma,
            config.alpha,
            frame_shape=input_shape,
        )
        self.atoms = torch.linspace(config.v_min, config.v_max, config.num_atoms, device=self.device)
        self.delta_z = (config.v_max - config.v_min) / (config.num_atoms - 1)
        self.n_step = config.n_step

        if self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

    def act(self, state: np.ndarray, training: bool = True) -> int:
        self.q_network.train(training)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.q_network(state_t)
            q_values = (dist * self.atoms).sum(dim=2)
        return int(q_values.argmax(dim=1).item())

    def act_batch(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        self.q_network.train(training)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist = self.q_network(states_t)
            q_values = (dist * self.atoms).sum(dim=2)
        return q_values.argmax(dim=1).cpu().numpy()

    def update(self) -> Optional[dict]:
        if len(self.memory) < self.config.min_replay_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(
            self.config.batch_size, self.beta
        )

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        with torch.amp.autocast(self.device.type, enabled=self.scaler is not None):
            current_dist = self.q_network(states_t)
            current_dist = current_dist.gather(
                1, actions_t.unsqueeze(-1).expand(-1, -1, self.config.num_atoms)
            ).squeeze(1)

            with torch.no_grad():
                next_online_dist = self.q_network(next_states_t)
                next_q = (next_online_dist * self.atoms).sum(dim=2)
                next_actions = next_q.argmax(dim=1)

                next_target_dist = self.target_network(next_states_t)
                next_target_dist = next_target_dist.gather(
                    1, next_actions.view(-1, 1, 1).expand(-1, -1, self.config.num_atoms)
                ).squeeze(1)
                target_dist = self._project_distribution(next_target_dist, rewards_t, dones_t)

            loss_per_sample = -(target_dist * current_dist.log()).sum(dim=1)
            loss = (loss_per_sample * weights_t).mean()

        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        with torch.no_grad():
            td_error = (target_dist - current_dist).abs().sum(dim=1).cpu().numpy()
            q_value = (current_dist * self.atoms).sum(dim=1).mean().item()
        self.memory.update_priorities(indices, td_error)

        self.steps += 1
        self._update_beta()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.steps % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        current_lr = self.optimizer.param_groups[0]["lr"]
        return {"loss": float(loss.item()), "q_value": q_value, "beta": self.beta, "lr": current_lr}

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
        return projected.clamp_min(1e-6)

    def _update_beta(self):
        self.beta = min(
            1.0,
            self.config.beta_start + (1.0 - self.config.beta_start) * self.steps / self.config.beta_frames,
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "beta": self.beta,
            "config": self.config.__dict__,
        }
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        torch.save(data, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = int(checkpoint["steps"])
        self.beta = float(checkpoint["beta"])
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])


def build_low_end_human_target_config() -> BetterThanHumanRainbowConfig:
    """Return the recommended config for slow CPU experimentation."""

    return BetterThanHumanRainbowConfig(
        hidden_dim=256,
        batch_size=32,
        buffer_capacity=100_000,
        min_replay_size=5_000,
        train_freq=4,
        target_update_freq=2_500,
        learning_rate=0.0001,
        gamma=0.997,
        v_min=-10.0,
        v_max=100.0,
        num_atoms=51,
        n_step=5,
        beta_frames=250_000,
        teacher_warm_start_steps=20_000,
    )
