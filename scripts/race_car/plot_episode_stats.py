"""Plot per-episode training metrics from a race_car SAC run.

Reads ``train_episode_log.csv`` (written by the SAC trainer when the
``RaceCarEpisodeStats`` wrapper is active) and plots:

- ``lap_time`` per finished episode (with rolling median)
- ``success`` and ``violation`` rates per finished episode (rolling means)
- ``lap_return`` per finished episode (with rolling median)

Each row in the CSV is one finished episode (no step-level smoothing).

Usage::

    python scripts/race_car/plot_episode_stats.py output/race_car_sac_fop_*/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(csv_path: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[float]] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for k, v in row.items():
                columns.setdefault(k, []).append(float(v) if v != "" else float("nan"))
    return {k: np.asarray(vals, dtype=np.float64) for k, vals in columns.items()}


def _rolling(x: np.ndarray, w: int, op: str = "mean") -> np.ndarray:
    if len(x) == 0 or w <= 1:
        return x
    w = min(w, len(x))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        chunk = x[lo : i + 1]
        out[i] = np.nanmean(chunk) if op == "mean" else np.nanmedian(chunk)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Trainer output directory.")
    parser.add_argument("--window", type=int, default=20, help="Rolling window in episodes.")
    parser.add_argument("--out", type=Path, default=None, help="Optional PNG path.")
    args = parser.parse_args()

    csv_path = args.run_dir / "train_episode_log.csv"
    if not csv_path.is_file():
        raise SystemExit(f"No train_episode_log.csv at {csv_path}")
    cols = _load(csv_path)

    n = len(cols["timestamp"])
    ep = np.arange(1, n + 1)
    w = args.window

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(ep, cols["lap_time"], ".", ms=3, alpha=0.4, label="lap_time")
    axes[0].plot(ep, _rolling(cols["lap_time"], w, "median"), "-", label=f"rolling median (w={w})")
    axes[0].set_ylabel("lap time [s]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(ep, _rolling(cols["success"], w, "mean"), label="success rate")
    axes[1].plot(ep, _rolling(cols["violation"], w, "mean"), label="violation rate")
    axes[1].set_ylabel(f"rate (rolling, w={w})")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(ep, cols["lap_return"], ".", ms=3, alpha=0.4, label="lap_return")
    axes[2].plot(
        ep, _rolling(cols["lap_return"], w, "median"), "-", label=f"rolling median (w={w})"
    )
    axes[2].set_ylabel("lap return")
    axes[2].set_xlabel("training episode")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.suptitle(f"race_car training episodes — {args.run_dir.name}")
    fig.tight_layout()

    if args.out is not None:
        fig.savefig(args.out, dpi=120)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
