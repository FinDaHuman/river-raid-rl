import numpy as np
import torch

from riverraid_rl.agents.better_than_human import (
    BetterThanHumanRainbowAgent,
    BetterThanHumanRainbowConfig,
    NoisyCategoricalDuelingDQN,
    build_low_end_human_target_config,
)
from riverraid_rl.config import EnvConfig


def test_low_end_human_target_config_is_cpu_reasonable():
    cfg = build_low_end_human_target_config()

    assert cfg.hidden_dim <= 256
    assert cfg.batch_size <= 32
    assert cfg.buffer_capacity <= 100_000
    assert cfg.num_atoms == 51
    assert cfg.use_noisy_nets is True
    assert cfg.teacher_warm_start_steps > 0


def test_noisy_categorical_dueling_network_outputs_valid_distributions():
    model = NoisyCategoricalDuelingDQN((4, 84, 84), num_actions=6, num_atoms=51, hidden_dim=64)
    states = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)

    dist = model(states)

    assert dist.shape == (2, 6, 51)
    assert torch.allclose(dist.sum(dim=2), torch.ones(2, 6), atol=1e-5)
    assert torch.isfinite(dist).all()


def test_better_than_human_agent_can_act_and_update_from_replay():
    env_cfg = EnvConfig()
    cfg = BetterThanHumanRainbowConfig(
        hidden_dim=64,
        batch_size=4,
        buffer_capacity=32,
        min_replay_size=4,
        target_update_freq=10,
        num_atoms=21,
    )
    agent = BetterThanHumanRainbowAgent(env_cfg, cfg, num_actions=6, device="cpu")

    for i in range(8):
        state = np.full((4, 84, 84), i, dtype=np.uint8)
        next_state = np.full((4, 84, 84), i + 1, dtype=np.uint8)
        agent.memory.push(state, i % 6, float(i % 2), next_state, False)

    action = agent.act(np.zeros((4, 84, 84), dtype=np.uint8), training=True)
    metrics = agent.update()

    assert 0 <= action < 6
    assert metrics is not None
    assert "loss" in metrics
    assert "q_value" in metrics
    assert "beta" in metrics
