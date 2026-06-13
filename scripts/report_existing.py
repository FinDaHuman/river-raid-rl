"""Report existing checkpoint performance from progress.json files.

Scans all checkpoint directories for ``progress.json``, extracts the best eval
metrics, and prints a compact table.  No environment or GPU needed.
"""

import json
from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def main():
    rows = []
    for entry in sorted(CHECKPOINTS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        progress_file = entry / "progress.json"
        if not progress_file.exists():
            continue
        with open(progress_file) as f:
            data = json.load(f)
        best = data.get("best_mean_reward", "?")
        targets = data.get("targets", {})
        histories = data.get("history", [])
        last = histories[-1] if histories else {}
        rows.append({
            "run": entry.name,
            "best": best,
            "steps": last.get("step", "?"),
            "mean": last.get("mean_reward", "?"),
            "pct_human": last.get("pct_human_expert", "?"),
            "pct_sota": last.get("pct_rainbow_sota", "?"),
            "elapsed": last.get("elapsed_seconds", "?"),
        })

    if not rows:
        print("No progress.json files found under checkpoints/")
        return

    header = f"{'Run':<30} {'Best':>8} {'Steps':>8} {'Mean':>8} {'%Human':>8} {'%SOTA':>8} {'Time(s)':>8}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in rows:
        pct_h = r['pct_human']
        pct_s = r['pct_sota']
        h_str = pct_h if isinstance(pct_h, str) else f"{pct_h:.2f}"
        s_str = pct_s if isinstance(pct_s, str) else f"{pct_s:.2f}"
        t_str = r['elapsed'] if isinstance(r['elapsed'], str) else f"{r['elapsed']:.0f}"
        print(
            f"{r['run']:<30} {r['best']:>8} {r['steps']:>8} {r['mean']:>8} "
            f"{h_str:>8} {s_str:>8} {t_str:>8}"
        )
    print("=" * len(header))


if __name__ == "__main__":
    main()
