"""Plot timeseries data from a HVAC validation run.

Usage:
    python scripts/hvac/plotting/plot_validation_run.py <path/to/val_timeseries_step0.csv>

The CSV is produced by run_baseline.py when --env hvac is used. The plot is saved
as a PNG alongside the CSV file.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _add_failure_patches(axes, t, solver_status):
    """Shade every contiguous region where solver_status != 0 across all axes."""
    failure = (solver_status != 0).to_numpy()
    # Find start/end indices of contiguous failure blocks
    edges = np.diff(failure.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    for ax in axes:
        for s, e in zip(starts, ends):
            ax.axvspan(
                t.iloc[s],
                t.iloc[min(e, len(t) - 1)],
                color="red",
                alpha=0.12,
                linewidth=0,
                zorder=0,
            )


def plot_validation_run(csv_path: Path, output_path: Path | None = None) -> Path:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    t = df["datetime"]

    fig, axes = plt.subplots(5, 1, figsize=(14, 24), sharex=True)
    fig.suptitle(f"HVAC Validation Run\n{csv_path.name}", fontsize=12)

    # --- Panel 1: Ti (indoor) + comfort bounds ---
    ax = axes[0]
    ax.plot(t, df["Ti_C"], label="Ti (indoor)", color="tab:blue", linewidth=1.5)
    ax.plot(
        t,
        df["ambient_temp_C"],
        label="T_amb",
        color="tab:gray",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
    )
    ax.fill_between(
        t, df["T_lb_C"], df["T_ub_C"], alpha=0.12, color="tab:blue", label="Comfort band"
    )
    ax.step(t, df["T_lb_C"], color="tab:blue", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.step(t, df["T_ub_C"], color="tab:blue", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Ti (°C)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Th (radiator) ---
    ax = axes[1]
    ax.plot(t, df["Th_C"], label="Th (radiator)", color="tab:red", linewidth=1.2)
    ax.set_ylabel("Th (°C)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Te (envelope) ---
    ax = axes[2]
    ax.plot(t, df["Te_C"], label="Te (envelope)", color="tab:green", linewidth=1.2)
    ax.set_ylabel("Te (°C)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Price + money spent ---
    ax = axes[3]
    ax2 = ax.twinx()
    ax.step(t, df["price"], color="tab:purple", linewidth=1.2, where="post", label="Price")
    ax2.step(
        t,
        df["action_W"],
        color="tab:red",
        linewidth=1,
        where="post",
        label="Heating power (W)",
    )
    ax.set_ylabel("Price (€/kWh)", color="tab:purple")
    ax2.set_ylabel("Heating power (W)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 6: Solar irradiance ---
    ax = axes[4]
    ax.fill_between(t, df["solar_W_m2"], alpha=0.5, color="gold")
    ax.step(t, df["solar_W_m2"], color="goldenrod", linewidth=1, where="post")
    ax.set_ylabel("Solar irradiance\n(W/m²)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # --- shared x-axis formatting ---
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=0, ha="center")

    # --- summary stats in figure ---
    total_cost = df["money_spent"].sum()
    total_energy = df["energy_kwh"].sum()
    success_rate = df["success"].mean() * 100
    total_viol = df["constraint_violation_K"].sum()
    summary = (
        f"Total cost: {total_cost:.3f} €  |  "
        f"Total energy: {total_energy:.2f} kWh  |  "
        f"Success rate: {success_rate:.1f}%  |  "
        f"Total violation: {total_viol:.1f} K·steps"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=9, color="dimgray")

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    out = output_path or csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out}")
    return out


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot HVAC validation timeseries from a CSV file.")
    parser.add_argument("csv", type=Path, help="Path to val_timeseries_step*.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: same dir as CSV, same stem + .png)",
    )
    args = parser.parse_args()
    plot_validation_run(args.csv, args.output)
