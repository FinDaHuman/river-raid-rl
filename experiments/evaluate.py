"""Utility to evaluate a saved checkpoint.

Usage:
    python experiments/evaluate.py --checkpoint path/to/model.pt --episodes 20
"""

import argparse
import os
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import torch

from riverraid_rl import make_riverraid_env, EnvConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.config import RainbowConfig

def load_agent(checkpoint_path: str, env_cfg: EnvConfig, device: str = "cpu") -> RainbowAgent:
    """Load a checkpoint into a RainbowAgent.

    The checkpoint does not store the full config, so we instantiate a
    ``RainbowConfig`` with the default hyper‑parameters used for training and
    then overwrite the model weights via ``agent.load``.
    """
    cfg = RainbowConfig()  # defaults match the training YAML
    agent = RainbowAgent(env_cfg, cfg, num_actions=6, device=device)
    agent.load(checkpoint_path)
    return agent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved River Raid checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    env_cfg = EnvConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = load_agent(args.checkpoint, env_cfg, device)
    result = evaluate(agent, env_cfg, num_episodes=args.episodes)
    print("Evaluation result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
