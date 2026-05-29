import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List


class Logger:
    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics: Dict[str, List] = defaultdict(list)
        self.steps: List[int] = []
        self.start_time = datetime.now()

    def log(self, step: int, metrics: Dict[str, float]):
        self.steps.append(step)
        for key, value in metrics.items():
            self.metrics[key].append(value)

        elapsed = (datetime.now() - self.start_time).total_seconds()
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"[Step {step}] [{elapsed:.0f}s] {metrics_str}")

    def save(self):
        data = {
            "steps": self.steps,
            "metrics": dict(self.metrics),
            "start_time": self.start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }
        path = os.path.join(self.log_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Metrics saved to {path}")

    def get_log_dir(self):
        return self.log_dir
