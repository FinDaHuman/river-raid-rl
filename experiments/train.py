"""Unified training script for the River Raid RL monorepo.

The script reads a YAML configuration that specifies:
- ``env``: environment hyper‑parameters (EnvConfig fields)
- ``agent``: which agent to instantiate (e.g. ``rainbow``, ``dqn``, ``double_dqn`` ...)
- ``training``: training loop parameters (total_timesteps, eval_freq, etc.)

It builds the environment, creates the requested agent, runs the standard training loop
(including evaluation, logging and checkpointing) and finally prints a concise report.

The implementation deliberately mirrors the original ``train_riverraid.py`` but is
compact, well‑documented and hardware‑aware – it will automatically fall back to the
CPU if a GPU is not available.  All heavy‑weight imports are placed inside the
``main`` function so that the module can be imported without side‑effects (useful
for unit tests).
"""

import argparse
import os
import time
from pathlib import Path

import yaml

# Import from the thin wrapper package we created under ``src``.
# ``src`` is added to ``PYTHONPATH`` by the ``setup.cfg`` / ``pyproject`` build system,
# but for interactive use we prepend the repository root.
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from riverraid_rl import (
    make_riverraid_env,
    EnvConfig,
    DQNConfig,
    RainbowConfig,
    TrainingConfig,
    Config,
)
from riverraid_rl.agents.base import BaseAgent
from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.agents.dqn import DQNAgent
# Additional agents can be imported here when they are implemented in the repo.

from riverraid_rl.utils.logger import Logger
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.utils.progress import ProgressTracker, load_progress


def build_agent(agent_name: str, env_cfg: EnvConfig, cfg, device: str, num_actions: int) -> BaseAgent:
    """Factory that instantiates the requested agent.

    Parameters
    ----------
    agent_name:
        Identifier of the agent type (e.g. ``rainbow`` or ``dqn``).
    env_cfg:
        The environment configuration (used for network input shape).
    cfg:
        The full configuration object (contains sub‑configs for each agent).
    device:
        ``"cpu"`` or ``"cuda"`` – automatically chosen if CUDA is available.
    num_actions:
        Number of discrete actions in the River Raid environment.
    """
    if agent_name == "rainbow":
        return RainbowAgent(env_cfg, cfg.rainbow, num_actions, device)
    elif agent_name == "dqn":
        return DQNAgent(env_cfg, cfg.dqn, num_actions, device)
    else:
        raise ValueError(f"Unsupported agent '{agent_name}'. Available: rainbow, dqn")


def main():
    parser = argparse.ArgumentParser(description="Train a River Raid RL agent")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("-o", "--output", default="experiments/run", help="Directory to store logs and checkpoints")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    # Convert dictionaries to dataclasses for type safety.
    env_cfg = EnvConfig(**cfg_dict.get("env", {}))
    # The config file may contain a single ``agent`` block with its specific fields.
    # We construct the corresponding dataclass on the fly.
    agent_cfg_name = cfg_dict["agent"]["type"]
    if agent_cfg_name == "rainbow":
        agent_cfg = RainbowConfig(**cfg_dict["agent"].get("params", {}))
    elif agent_cfg_name == "dqn":
        agent_cfg = DQNConfig(**cfg_dict["agent"].get("params", {}))
    else:
        raise ValueError(f"Unknown agent type {agent_cfg_name}")

    training_cfg = TrainingConfig(**cfg_dict.get("training", {}))
    cfg = Config(env=env_cfg, dqn=cfg_dict.get("dqn", {}), rainbow=cfg_dict.get("rainbow", {}), training=training_cfg)

    # ---------------------------------------------------------------------
    # Device selection – respect low‑end hardware.
    # ---------------------------------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ---------------------------------------------------------------------
    # Environment and agent construction
    # ---------------------------------------------------------------------
    env = make_riverraid_env(env_cfg, clip_rewards=True)
    num_actions = 6  # River Raid discrete action space (fixed for this game)
    agent = build_agent(agent_cfg_name, env_cfg, cfg, device, num_actions)

    # ---------------------------------------------------------------------
    # Logging utilities
    # ---------------------------------------------------------------------
    logger = Logger("logs", training_cfg.run_name)
    tracker = ProgressTracker(os.path.join(args.output, "progress"))

    # ---------------------------------------------------------------------
    # Training loop (mirrors original implementation but is concise)
    # ---------------------------------------------------------------------
    state, _ = env.reset()
    episode_reward = 0.0
    best_mean = float("-inf")
    start = time.time()

    for step in range(training_cfg.total_timesteps):
        action = agent.act(state, training=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, terminated or truncated)
        state = next_state
        episode_reward += reward

        if len(agent.memory) >= getattr(agent.config, "min_replay_size", 0):
            metrics = agent.update()
            if metrics and step % training_cfg.log_freq == 0:
                logger.log(step, metrics)

        if terminated or truncated:
            state, _ = env.reset()
            episode_reward = 0.0

        if step > 0 and step % training_cfg.eval_freq == 0:
            eval_res = evaluate(agent, env_cfg, training_cfg.eval_episodes)
            logger.log(step, {"eval/mean": eval_res["mean_reward"], "eval/std": eval_res["std_reward"]})
            mean = eval_res["mean_reward"]
            if mean > best_mean:
                best_mean = mean
                checkpoint_path = os.path.join(args.output, f"best_{step}.pt")
                agent.save(checkpoint_path)
                print(f"New best checkpoint saved: {checkpoint_path}")
            tracker.print_eval(step, step, eval_res, time.time() - start, best_mean)

        if step > 0 and step % training_cfg.save_freq == 0:
            agent.save(os.path.join(args.output, f"ckpt_{step}.pt"))

    env.close()
    final_path = os.path.join(args.output, "final.pt")
    agent.save(final_path)
    logger.save()
    print(f"Training finished. Final checkpoint: {final_path}")

if __name__ == "__main__":
    main()
