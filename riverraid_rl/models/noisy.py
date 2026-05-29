import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma, self.std_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.std_init / math.sqrt(self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
            bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyDuelingDQN(nn.Module):
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
            conv_out = int(torch.prod(torch.tensor(o.size())).item())

        self.fc_adv = NoisyLinear(conv_out, hidden_dim)
        self.fc_val = NoisyLinear(conv_out, hidden_dim)
        self.advantage = NoisyLinear(hidden_dim, num_actions)
        self.value = NoisyLinear(hidden_dim, 1)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv = F.relu(self.fc_adv(x))
        val = F.relu(self.fc_val(x))
        adv = self.advantage(adv)
        val = self.value(val)
        return val + adv - adv.mean(dim=1, keepdim=True)
