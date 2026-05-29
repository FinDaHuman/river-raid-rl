import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional


class ProgressTracker:
    THEORETICAL_MAX = 1_000_000
    HUMAN_EXPERT = 13_513
    RAINBOW_SOTA = 20_675
    DQN_BENCHMARK = 8_311

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.history: List[Dict] = []
        self.start_time = time.time()
        os.makedirs(save_dir, exist_ok=True)

    def record_eval(self, step: int, frames: int, episode: int,
                    mean_reward: float, std_reward: float,
                    min_reward: float, max_reward: float,
                    mean_length: float, elapsed: float,
                    extra: Optional[Dict] = None):
        pct_max = 100.0 * mean_reward / self.THEORETICAL_MAX
        pct_human = 100.0 * mean_reward / self.HUMAN_EXPERT
        pct_sota = 100.0 * mean_reward / self.RAINBOW_SOTA

        entry = {
            "step": step,
            "frames": frames,
            "episode": episode,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "mean_length": mean_length,
            "pct_theoretical_max": pct_max,
            "pct_human_expert": pct_human,
            "pct_rainbow_sota": pct_sota,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            entry.update(extra)
        self.history.append(entry)
        self._save()

    def report(self, mean_reward: float, best_ever: float) -> str:
        lines = []
        lines.append("")
        lines.append("=" * 75)
        lines.append(f"  {'Progress Report':^69s}")
        lines.append("=" * 75)
        lines.append(f"  {'Metric':30s} {'Current':>14s} {'Best Ever':>14s} {'Target':>14s}")
        lines.append("-" * 75)
        lines.append(f"  {'Raw Score':30s} {mean_reward:>14.1f} {best_ever:>14.1f} {self.THEORETICAL_MAX:>14,}")
        lines.append(f"  {'% of Theoretical Max (1M)':30s} {100*mean_reward/self.THEORETICAL_MAX:>13.4f}% {100*best_ever/self.THEORETICAL_MAX:>13.4f}% {'100.0000%':>14s}")
        lines.append(f"  {'% of Human Expert (13.5K)':30s} {100*mean_reward/self.HUMAN_EXPERT:>13.2f}% {100*best_ever/self.HUMAN_EXPERT:>13.2f}% {'100.00%':>14s}")
        lines.append(f"  {'% of Rainbow SOTA (20.7K)':30s} {100*mean_reward/self.RAINBOW_SOTA:>13.2f}% {100*best_ever/self.RAINBOW_SOTA:>13.2f}% {'100.00%':>14s}")
        lines.append(f"  {'% of DQN Nature (8.3K)':30s} {100*mean_reward/self.DQN_BENCHMARK:>13.2f}% {100*best_ever/self.DQN_BENCHMARK:>13.2f}% {'100.00%':>14s}")
        lines.append("=" * 75)
        return "\n".join(lines)

    def print_eval(self, step: int, frames: int, result: Dict, elapsed: float, best: float):
        mean = result["mean_reward"]
        self.record_eval(
            step=step, frames=frames, episode=result.get("num_episodes", 0),
            mean_reward=mean, std_reward=result["std_reward"],
            min_reward=result["min_reward"], max_reward=result["max_reward"],
            mean_length=result["mean_length"], elapsed=elapsed,
        )
        print(self.report(mean, max(mean, best)))

    def _save(self):
        path = os.path.join(self.save_dir, "progress.json")
        with open(path, "w") as f:
            json.dump({
                "history": self.history,
                "best_mean_reward": max(h["mean_reward"] for h in self.history) if self.history else None,
                "total_elapsed": time.time() - self.start_time,
                "targets": {
                    "theoretical_max": self.THEORETICAL_MAX,
                    "human_expert": self.HUMAN_EXPERT,
                    "rainbow_sota": self.RAINBOW_SOTA,
                    "dqn_benchmark": self.DQN_BENCHMARK,
                },
            }, f, indent=2)

    @staticmethod
    def print_summary_header():
        print()
        print("=" * 110)
        hdr = (f"  {'Step':>8s} {'Frames':>10s} {'Mean':>8s} {'Std':>6s} "
               f"{'%Max':>9s} {'%Human':>9s} {'%SOTA':>9s} {'Length':>7s} {'Best':>8s} {'Time':>8s}")
        print(hdr)
        print("=" * 110)
        return hdr

    @staticmethod
    def format_eval_row(step, frames, mean, std, pct_max, pct_human, pct_sota, length, best, elapsed):
        return (f"  {step:>8,} {frames:>10,} {mean:>8.1f} {std:>6.1f} "
                f"{pct_max:>8.4f}% {pct_human:>8.2f}% {pct_sota:>8.2f}% "
                f"{length:>7.1f} {best:>8.1f} {elapsed:>7.0f}s")


def load_progress(save_dir: str) -> Optional[dict]:
    path = os.path.join(save_dir, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
