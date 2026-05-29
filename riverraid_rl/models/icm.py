import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InverseModel(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, next_state], dim=1)
        return self.net(x)


class ForwardModel(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class FeatureEncoder(nn.Module):
    def __init__(self, input_shape: tuple, feature_dim: int = 288):
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
            conv_out = int(np.prod(o.size()))

        self.fc = nn.Linear(conv_out, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class IntrinsicCuriosityModule(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        num_actions: int,
        feature_dim: int = 288,
        hidden_dim: int = 256,
        forward_scale: float = 1.0,
        inverse_scale: float = 1.0,
    ):
        super().__init__()
        self.forward_scale = forward_scale
        self.inverse_scale = inverse_scale
        self.feature_dim = feature_dim

        self.feature_encoder = FeatureEncoder(input_shape, feature_dim)
        self.inverse_model = InverseModel(feature_dim, num_actions, hidden_dim)
        self.forward_model = ForwardModel(feature_dim, num_actions, hidden_dim)

    def forward(
        self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor
    ) -> tuple:
        phi_state = self.feature_encoder(state)
        phi_next_state = self.feature_encoder(next_state)

        pred_action = self.inverse_model(phi_state, phi_next_state)
        pred_next_feature = self.forward_model(phi_state, action)

        inverse_loss = F.cross_entropy(pred_action, action.argmax(dim=1))
        forward_loss = F.mse_loss(pred_next_feature, phi_next_state.detach())

        intrinsic_reward = self.forward_scale * (
            (pred_next_feature - phi_next_state.detach()).norm(dim=1, p=2)
        )

        loss = self.inverse_scale * inverse_loss + self.forward_scale * forward_loss
        return loss, intrinsic_reward, {
            "icm/inverse_loss": inverse_loss.item(),
            "icm/forward_loss": forward_loss.item(),
            "icm/intrinsic_reward": intrinsic_reward.mean().item(),
        }
