"""Test that top‑level imports from ``riverraid_rl`` resolve correctly.

Note: ``riverraid_rl.env`` depends on a specific gymnasium version
(``FrameStackObservation`` moved across releases) so it is tested separately
in ``test_env.py`` if the environment is installed.
"""

def test_import_config():
    from riverraid_rl.config import EnvConfig, DQNConfig, RainbowConfig, TrainingConfig
    assert EnvConfig().frame_stack == 4
    assert DQNConfig().batch_size == 32
    assert RainbowConfig().num_atoms == 51


def test_import_models():
    from riverraid_rl.models.cnn import DQNCNN, DuelingDQN, CategoricalDuelingDQN
    assert DQNCNN is not None


def test_import_memory():
    from riverraid_rl.memory.replay import ReplayBuffer, PrioritizedReplayBuffer
    assert ReplayBuffer is not None
