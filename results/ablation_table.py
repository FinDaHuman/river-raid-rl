"""Generate an ablation‑study comparison table from checkpoint progress files.

Scans each folder under ``checkpoints/`` for ``progress.json``, extracts the best
and final mean reward, and prints a formatted Markdown table suitable for pasting
into the README or a report.

Usage:
    python results/ablation_table.py
"""

import json
from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def extract_run(entry: Path):
    if not entry.is_dir():
        return None
    prog = entry / "progress.json"
    if not prog.exists():
        return None
    with open(prog) as f:
        data = json.load(f)

    best = data.get("best_mean_reward", None)
    total_elapsed = data.get("total_elapsed", None)
    targets = data.get("targets", {})
    histories = data.get("history", [])

    first = histories[0] if histories else {}
    last = histories[-1] if histories else {}

    return {
        "name": entry.name,
        "best": best,
        "best_at_step": last.get("step", "?"),
        "final_mean": last.get("mean_reward", "?"),
        "final_std": last.get("std_reward", "?"),
        "final_pct_human": last.get("pct_human_expert", "?"),
        "final_pct_sota": last.get("pct_rainbow_sota", "?"),
        "total_steps": last.get("step", "?"),
        "elapsed_s": total_elapsed or last.get("elapsed_seconds", "?"),
        "targets": targets,
    }


def main():
    rows = []
    for entry in sorted(CHECKPOINTS_DIR.iterdir()):
        r = extract_run(entry)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No runs found with progress.json")
        return

    print("## Ablation Study – Results\n")
    header = (
        f"| {'Run':<30} | {'Steps':>8} | {'Best':>8} | {'Final Mean':>10} "
        f"| {'%Human':>7} | {'%SOTA':>7} | {'Time (s)':>8} |"
    )
    sep = "|" + "|".join(["-" * (len(c.strip()) + 2 if "Run" in c else 12) for c in header.split("|")[1:-1]]) + "|"
    print(header)
    print(sep)
    for r in rows:
        best_str = f"{r['best']:.1f}" if isinstance(r['best'], (int, float)) else "N/A"
        mean_str = f"{r['final_mean']:.1f}" if isinstance(r['final_mean'], (int, float)) else "?"
        human_str = f"{r['final_pct_human']:.2f}" if isinstance(r['final_pct_human'], (int, float)) else "?"
        sota_str = f"{r['final_pct_sota']:.2f}" if isinstance(r['final_pct_sota'], (int, float)) else "?"
        time_str = f"{r['elapsed_s']:.0f}" if isinstance(r['elapsed_s'], (int, float)) else "?"
        print(
            f"| {r['name']:<30} | {r['total_steps']:>8} | {best_str:>8} | {mean_str:>10} "
            f"| {human_str:>7} | {sota_str:>7} | {time_str:>8} |"
        )
    print()

    # Print benchmark targets for reference
    if rows and rows[0]["targets"]:
        t = rows[0]["targets"]
        print("### Benchmark Targets\n")
        for k, v in t.items():
            print(f"- **{k.replace('_', ' ').title()}**: {v:,}")
        print()


if __name__ == "__main__":
    main()
