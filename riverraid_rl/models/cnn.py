import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNCNN(nn.Module):
    def __init__(self, input_shape: tuple, num_actions: int, hidden_dim: int = 512):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape: tuple, num_actions: int, hidden_dim: int = 512):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Linear(conv_out_size, hidden_dim)
        self.fc_val = nn.Linear(conv_out_size, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

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


class CategoricalDuelingDQN(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        num_actions: int,
        num_atoms: int = 51,
        hidden_dim: int = 512,
    ):
        super().__init__()
        c, h, w = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Linear(conv_out_size, hidden_dim)
        self.fc_val = nn.Linear(conv_out_size, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, num_actions * num_atoms)
        self.value = nn.Linear(hidden_dim, num_atoms)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv = F.relu(self.fc_adv(x))
        val = F.relu(self.fc_val(x))
        adv = self.advantage(adv).view(-1, self.num_actions, self.num_atoms)
        val = self.value(val).view(-1, 1, self.num_atoms)
        logits = val + adv - adv.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=2)


class CategoricalDuelingDQNAttention(CategoricalDuelingDQN):
    def __init__(self, input_shape: tuple, num_actions: int, num_atoms: int = 51, hidden_dim: int = 512):
        super().__init__(input_shape, num_actions, num_atoms, hidden_dim)
        from riverraid_rl.models.attention import SpatialAttention, ChannelAttention
        self.attn1 = SpatialAttention(32)
        self.attn2 = ChannelAttention(64)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = self.attn1(x)
        x = F.relu(self.conv2(x))
        x = self.attn2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv = F.relu(self.fc_adv(x))
        val = F.relu(self.fc_val(x))
        adv = self.advantage(adv).view(-1, self.num_actions, self.num_atoms)
        val = self.value(val).view(-1, 1, self.num_atoms)
        logits = val + adv - adv.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=2)
