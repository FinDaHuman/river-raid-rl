"""Plot training curves from logged metrics.

Reads the ``metrics.json`` files produced by ``Logger`` and the ``progress.json``
files produced by ``ProgressTracker``, then generates publication‑quality figures.

Usage:
    python results/plot_training.py                     # all runs in logs/
    python results/plot_training.py --paths logs/run1 logs/run2  # selected runs
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "results" / "figures"


def load_log_metrics(path: Path):
    """Load a ``metrics.json`` and extract loss / q_value / eval curves."""
    with open(path / "metrics.json") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    steps = np.array(data.get("steps", []), dtype=int)

    # Extract per‑step training metrics (if logged)
    loss = np.array(metrics.get("loss", []))
    q_value = np.array(metrics.get("q_value", []))
    epsilon = np.array(metrics.get("epsilon", []))

    # Extract evaluation metrics (have prefix "eval/")
    eval_steps = []
    eval_means = []
    eval_stds = []
    for i, s in enumerate(steps):
        key = f"eval/mean_reward at step {s}"
        if key in metrics:
            eval_steps.append(s)
            eval_means.append(metrics[key])
            eval_stds.append(metrics.get(f"eval/std_reward at step {s}", 0))

    return {
        "run_name": path.name,
        "steps": steps,
        "loss": loss,
        "q_value": q_value,
        "epsilon": epsilon,
        "eval_steps": np.array(eval_steps),
        "eval_means": np.array(eval_means),
        "eval_stds": np.array(eval_stds),
    }


def load_progress(path: Path):
    """Load a ``progress.json`` checkpoint evaluation history."""
    progress_file = path / "progress.json"
    if not progress_file.exists():
        return None
    with open(progress_file) as f:
        data = json.load(f)
    return data


def plot_eval_curves(runs_data, output_path: Path):
    """Plot evaluation mean reward vs. training steps for multiple runs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for rd in runs_data:
        if len(rd["eval_steps"]) == 0:
            continue
        ax.errorbar(
            rd["eval_steps"], rd["eval_means"], yerr=rd["eval_stds"],
            marker="o", capsize=3, label=rd["run_name"], alpha=0.8,
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward (eval, 10 episodes)")
    ax.set_title("River Raid – Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "eval_curves.png", dpi=150)
    print(f"Saved: {output_path / 'eval_curves.png'}")
    plt.close(fig)


def plot_loss_curves(runs_data, output_path: Path):
    """Plot smoothed training loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for rd in runs_data:
        if len(rd["loss"]) == 0:
            continue
        # Simple moving average for readability
        win = max(1, len(rd["loss"]) // 50)
        kernel = np.ones(win) / win
        smooth = np.convolve(rd["loss"], kernel, mode="valid")
        steps = rd["steps"][: len(smooth)]
        ax.plot(steps, smooth, label=rd["run_name"], alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.set_title("River Raid – Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "loss_curves.png", dpi=150)
    print(f"Saved: {output_path / 'loss_curves.png'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from logged metrics")
    parser.add_argument("--paths", nargs="*", default=None, help="Paths to run log directories")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.paths:
        log_dirs = [Path(p) for p in args.paths]
    else:
        log_dirs = sorted(LOGS_DIR.iterdir()) if LOGS_DIR.exists() else []

    if not log_dirs:
        print("No log directories found. Use --paths to specify.")
        sys.exit(1)

    runs_data = []
    for d in log_dirs:
        if not d.is_dir():
            continue
        try:
            runs_data.append(load_log_metrics(d))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Skipping {d.name}: {e}")

    if not runs_data:
        print("No valid metrics found.")
        sys.exit(1)

    plot_eval_curves(runs_data, OUTPUT_DIR)
    plot_loss_curves(runs_data, OUTPUT_DIR)


if __name__ == "__main__":
    main()
