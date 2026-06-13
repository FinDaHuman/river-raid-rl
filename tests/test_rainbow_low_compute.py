import numpy as np
import torch

from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.memory.replay import NStepPrioritizedReplayBuffer, PrioritizedReplayBuffer


def make_state(value: int = 0) -> np.ndarray:
    return np.full((4, 84, 84), value, dtype=np.uint8)


def test_c51_projection_preserves_probability_mass_on_exact_atoms():
    env_cfg = EnvConfig()
    cfg = RainbowConfig(num_atoms=3, v_min=0.0, v_max=2.0, hidden_dim=8)
    agent = RainbowAgent(env_cfg, cfg, num_actions=2, device="cpu")

    next_dist = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.8, 0.1],
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor([0.0, 1.0], dtype=torch.float32)
    dones = torch.tensor([1.0, 1.0], dtype=torch.float32)

    projected = agent._project_distribution(next_dist, rewards, dones)

    assert torch.allclose(projected.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(projected[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(projected[1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_prioritized_replay_stores_frames_as_uint8_and_samples():
    replay = PrioritizedReplayBuffer(capacity=8, alpha=0.6)

    for i in range(6):
        replay.push(make_state(i), i % 2, float(i), make_state(i + 1), False)

    states, actions, rewards, next_states, dones, indices, weights = replay.sample(4, beta=0.4)

    assert states.dtype == np.uint8
    assert next_states.dtype == np.uint8
    assert actions.dtype == np.int64
    assert rewards.dtype == np.float32
    assert dones.dtype == np.float32
    assert len(indices) == 4
    assert weights.shape == (4,)


def test_n_step_replay_keeps_vector_env_trajectories_separate():
    replay = NStepPrioritizedReplayBuffer(capacity=16, n_step=3, gamma=0.9, alpha=0.6)

    for step in range(3):
        replay.push(make_state(10 + step), 0, 1.0, make_state(11 + step), False, env_id=0)
        replay.push(make_state(100 + step), 1, 2.0, make_state(101 + step), False, env_id=1)

    p = replay.prioritized
    assert p.size == 2

    assert int(p.states[0][0, 0, 0]) == 10
    assert int(p.next_states[0][0, 0, 0]) == 13
    assert p.rewards[0] == np.float32(1.0 + 0.9 + 0.9**2)

    assert int(p.states[1][0, 0, 0]) == 100
    assert int(p.next_states[1][0, 0, 0]) == 103
    assert p.rewards[1] == np.float32(2.0 + 2.0 * 0.9 + 2.0 * 0.9**2)


def test_n_step_replay_flushes_terminal_partial_sequences():
    replay = NStepPrioritizedReplayBuffer(capacity=16, n_step=3, gamma=0.9, alpha=0.6)

    replay.push(make_state(1), 0, 1.0, make_state(2), False)
    replay.push(make_state(2), 1, 2.0, make_state(3), True)

    p = replay.prioritized
    assert p.size == 2
    assert p.rewards[0] == np.float32(1.0 + 2.0 * 0.9)
    assert bool(p.dones[0]) is True
    assert p.rewards[1] == np.float32(2.0)
    assert bool(p.dones[1]) is True
