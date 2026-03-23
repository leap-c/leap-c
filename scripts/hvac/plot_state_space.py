#!/usr/bin/env python3
"""Print and plot the continuous and discrete state-space matrices for the HVAC model.

Prints Ac, Bc, Ec, Ad, Bd, Ed to stdout and saves a figure with colour-mapped
matrix plots to --output (default: outputs/state_space_matrices.pdf).

Usage
-----
  python scripts/hvac/plot_state_space.py
  python scripts/hvac/plot_state_space.py --dt 900 --output /tmp/matrices.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from leap_c.examples.hvac.dynamics import (
    HydronicDynamicsParameters,
    transcribe_continuous_state_space,
    transcribe_discrete_state_space,
)

_STATE_LABELS = ["Ti", "Th", "Te"]
_INPUT_LABELS = ["qh"]
_DIST_LABELS = ["Ta", "Is"]


def _print_matrix(name: str, M: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    col_w = max(12, max(len(c) for c in col_labels) + 2)
    row_w = max(len(r) for r in row_labels) + 1
    header = " " * (row_w + 2) + "".join(f"{c:>{col_w}}" for c in col_labels)
    print(f"\n{name}  ({M.shape[0]}×{M.shape[1]})")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, row_label in enumerate(row_labels):
        vals = "".join(f"{M[i, j]:>{col_w}.4e}" for j in range(M.shape[1]))
        print(f"  {row_label:<{row_w}}{vals}")


def _plot_matrix(
    ax: plt.Axes, M: np.ndarray, title: str, row_labels: list[str], col_labels: list[str]
) -> None:
    vmax = np.max(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    # Annotate each cell with its value
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.2e}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print and plot HVAC state-space matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dt", type=float, default=900.0, help="Sampling period in seconds (15 min = 900 s)."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/state_space_matrices.pdf"))
    args = parser.parse_args()

    params = HydronicDynamicsParameters()
    Ac, Bc, Ec = transcribe_continuous_state_space(params)
    Ad, Bd, Ed = transcribe_discrete_state_space(dt=args.dt, params=params)

    # ── stdout ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  HVAC state-space matrices  (dt = {args.dt:.0f} s)")
    print("=" * 60)
    print("\nState  x = [Ti, Th, Te]  (K)")
    print("Input  u = [qh]          (W)")
    print("Dist.  d = [Ta, Is]      (K, W/m²)")

    _print_matrix("Ac  [1/s]", Ac, _STATE_LABELS, _STATE_LABELS)
    _print_matrix("Bc  [K/(W·s)]", Bc, _STATE_LABELS, _INPUT_LABELS)
    _print_matrix("Ec  [K/(K·s) | K/(W/m²·s)]", Ec, _STATE_LABELS, _DIST_LABELS)

    print()
    _print_matrix("Ad  [–]", Ad, _STATE_LABELS, _STATE_LABELS)
    _print_matrix("Bd  [K/W]", Bd, _STATE_LABELS, _INPUT_LABELS)
    _print_matrix("Ed  [K/K | K/(W/m²)]", Ed, _STATE_LABELS, _DIST_LABELS)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"HVAC state-space matrices  (dt = {args.dt:.0f} s, nominal parameters)",
        fontsize=12,
        fontweight="bold",
    )

    specs = [
        # row 0 – continuous
        (axes[0, 0], Ac, "Ac  [1/s]", _STATE_LABELS, _STATE_LABELS),
        (axes[0, 1], Bc, "Bc  [K/(W·s)]", _STATE_LABELS, _INPUT_LABELS),
        (axes[0, 2], Ec, "Ec  [K/(K·s), K/(W/m²·s)]", _STATE_LABELS, _DIST_LABELS),
        # row 1 – discrete
        (axes[1, 0], Ad, "Ad  [–]", _STATE_LABELS, _STATE_LABELS),
        (axes[1, 1], Bd, "Bd  [K/W]", _STATE_LABELS, _INPUT_LABELS),
        (axes[1, 2], Ed, "Ed  [K/K, K/(W/m²)]", _STATE_LABELS, _DIST_LABELS),
    ]
    for ax, M, title, rl, cl in specs:
        _plot_matrix(ax, M, title, rl, cl)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nFigure saved to: {args.output}")


if __name__ == "__main__":
    main()
