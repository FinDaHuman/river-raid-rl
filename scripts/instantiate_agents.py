"""Sanity‑check that every agent class can be instantiated and called.

No environment or training data required – this script imports each agent,
builds a minimal config, instantiates it, and calls ``act()`` on dummy input.

Usage:
    python scripts/instantiate_agents.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

# ── Import every available agent ──────────────────────────────────────────
from riverraid_rl.config import EnvConfig, DQNConfig, RainbowConfig
from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.agents.dqn import DQNAgent
from riverraid_rl.agents.random_agent import RandomAgent
from riverraid_rl.agents.rule_based import RuleBasedAgent

# ── Dummy observation matching the environment shape ──────────────────────
DUMMY_STATE = np.zeros((4, 84, 84), dtype=np.uint8)


def try_agent(name: str, agent):
    """Try ``agent.act()`` and report success or failure."""
    try:
        action = agent.act(DUMMY_STATE, training=False)
        assert 0 <= action <= 5, f"Action {action} out of range [0,5]"
        print(f"  [OK] {name:<30} -> action={action}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name:<30} -> {e}")
        return False


def main():
    device = "cpu"
    env_cfg = EnvConfig()
    num_actions = 6

    print("Instantiating agents...\n")

    # 1. RandomAgent (no config required)
    ra = RandomAgent(num_actions, seed=42)
    try_agent("RandomAgent", ra)

    # 2. RuleBasedAgent (no config required)
    rba = RuleBasedAgent(seed=42)
    try_agent("RuleBasedAgent", rba)

    # 3. DQNAgent (vanilla DQN)
    dqn_cfg = DQNConfig()
    dqn = DQNAgent(env_cfg, dqn_cfg, num_actions, device)
    try_agent("DQNAgent", dqn)

    # 4. RainbowAgent (full Rainbow)
    rb_cfg = RainbowConfig()
    rb = RainbowAgent(env_cfg, rb_cfg, num_actions, device)
    try_agent("RainbowAgent", rb)

    print("\nDone. All agents that passed are ready for training.")


if __name__ == "__main__":
    main()
