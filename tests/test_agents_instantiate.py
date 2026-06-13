"""Test that every agent class can be instantiated and can act on dummy input."""

import numpy as np
import pytest

from riverraid_rl.config import EnvConfig, DQNConfig, RainbowConfig
from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.agents.dqn import DQNAgent
from riverraid_rl.agents.random_agent import RandomAgent
from riverraid_rl.agents.rule_based import RuleBasedAgent

DUMMY_STATE = np.zeros((4, 84, 84), dtype=np.uint8)


@pytest.fixture
def env_cfg():
    return EnvConfig()


def _check_action(agent, name):
    action = agent.act(DUMMY_STATE, training=False)
    assert 0 <= action <= 5, f"{name}: action {action} out of range"


def test_random_agent_instantiate():
    agent = RandomAgent(num_actions=6, seed=42)
    _check_action(agent, "RandomAgent")


def test_rule_based_agent_instantiate():
    agent = RuleBasedAgent(seed=42)
    _check_action(agent, "RuleBasedAgent")


def test_dqn_agent_instantiate(env_cfg):
    cfg = DQNConfig()
    agent = DQNAgent(env_cfg, cfg, num_actions=6, device="cpu")
    _check_action(agent, "DQNAgent")


def test_rainbow_agent_instantiate(env_cfg):
    cfg = RainbowConfig()
    agent = RainbowAgent(env_cfg, cfg, num_actions=6, device="cpu")
    _check_action(agent, "RainbowAgent")
