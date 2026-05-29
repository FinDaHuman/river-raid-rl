import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.conv(x))
        return x * attn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x)
        return x * scale


class AttentionDQN(nn.Module):
    def __init__(self, input_shape: tuple, num_actions: int, hidden_dim: int = 512):
        super().__init__()
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.attn1 = SpatialAttention(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.attn2 = ChannelAttention(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            o = self.conv1(dummy)
            o = self.conv2(self.attn1(o))
            o = self.conv3(self.attn2(o))
            conv_out = int(np.prod(o.size()))

        self.fc1 = nn.Linear(conv_out, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = self.attn1(x)
        x = F.relu(self.conv2(x))
        x = self.attn2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AuxiliaryDQN(nn.Module):
    def __init__(self, input_shape: tuple, num_actions: int, hidden_dim: int = 512):
        super().__init__()
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            o = self.conv1(dummy)
            o = self.conv2(o)
            o = self.conv3(o)
            self.conv_out = int(np.prod(o.size()))

        self.shared = nn.Linear(self.conv_out, hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.fuel_head = nn.Linear(hidden_dim, 1)
        self.enemy_density_head = nn.Linear(hidden_dim, 10)
        self.position_head = nn.Linear(hidden_dim, 160)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        shared = F.relu(self.shared(x))

        q_values = self.policy_head(shared)
        value = self.value_head(shared)
        fuel_pred = torch.sigmoid(self.fuel_head(shared))
        enemy_density = F.softmax(self.enemy_density_head(shared), dim=1)
        position_pred = torch.sigmoid(self.position_head(shared))

        return q_values, {
            "value": value,
            "fuel_pred": fuel_pred,
            "enemy_density": enemy_density,
            "position_pred": position_pred,
        }
