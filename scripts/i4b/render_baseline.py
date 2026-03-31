r"""Interactive rendering of i4b baseline MPC trajectories.

Loads the CSV timeseries and the NPZ MPC trajectory file produced by
``run_baseline.py`` and renders an interactive two-panel figure.  A slider
lets you scrub through every closed-loop timestep and see the MPC prediction
that was active at that step overlaid on the actual trajectory.

Layout
------
Top panel    : Room temperature — actual past (solid), MPC predicted state
               trajectory over the horizon (dashed), comfort band (shaded).
Middle panel : Heat-pump supply temperature — actual past (solid), MPC planned
               control trajectory over the horizon (dashed).
Bottom panel : All other building states from the MPC trajectory (T_wall,
               T_hp_ret, …) at the selected timestep.

Usage
-----
    python scripts/i4b/render_baseline.py --run-dir <output/…>
    python scripts/i4b/render_baseline.py \\
        --csv val_timeseries_step0.csv \\
        --npz val_mpc_trajectories_step0.npz
    python scripts/i4b/render_baseline.py --run-dir <dir> --save animation.gif
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

# State index mapping for the 4R3C model (default).  Override via --state-names.
_DEFAULT_STATE_NAMES = ["T_room", "T_wall", "T_hp_ret"]


def _find_latest_run(output_root: Path = Path("output")) -> Path:
    """Return the most recently modified run directory under ``output_root``."""
    candidates = sorted(
        (p for p in output_root.rglob("val_mpc_trajectories_step0.npz")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No val_mpc_trajectories_step0.npz found under {output_root}")
    return candidates[-1].parent


def load_data(
    run_dir: Path | None,
    csv_path: Path | None,
    npz_path: Path | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """Load CSV timeseries and NPZ MPC trajectories.

    Returns:
        df:          Timeseries dataframe, one row per closed-loop step.
        x:           MPC state trajectories, shape (T, N+1, nx).
        u:           MPC control trajectories, shape (T, N, nu).
        state_names: Name of each state dimension, length nx.
        du_dp:       Sensitivity dT_HP[k]/dQdot_gains[j], shape (T, N, N+1),
                     or None if not present in the NPZ.
    """
    if run_dir is not None:
        npz_candidates = sorted(run_dir.glob("val_mpc_trajectories_step*.npz"))
        csv_candidates = sorted(run_dir.glob("val_timeseries_step*.csv"))
        if not npz_candidates or not csv_candidates:
            raise FileNotFoundError(f"Missing files in {run_dir}")
        npz_path = npz_candidates[-1]
        csv_path = csv_candidates[-1]

    df = pd.read_csv(csv_path)
    data = np.load(npz_path, allow_pickle=True)
    state_names = list(data["state_names"]) if "state_names" in data else _DEFAULT_STATE_NAMES
    du_dp = data["du_dp"] if "du_dp" in data else None
    return df, data["x"], data["u"], state_names, du_dp


def render(
    df: pd.DataFrame,
    x: np.ndarray,
    u: np.ndarray,
    state_names: list[str],
    delta_t: float,
    save_path: Path | None,
    du_dp: np.ndarray | None = None,
) -> None:
    """Build the interactive figure.

    Args:
        df:          Timeseries dataframe.
        x:           State trajectories, shape (T, N+1, nx).
        u:           Control trajectories, shape (T, N, nu).
        state_names: Names for each state dimension.
        delta_t:     Sampling interval in seconds (used for the time axis).
        save_path:   If given, save an animation to this path instead of showing.
        du_dp:       Sensitivity dT_HP[k]/dQdot_gains[j], shape (T, N, N+1), or None.
    """
    T, N_plus1, nx = x.shape
    N = N_plus1 - 1
    steps = df["step"].to_numpy()
    dt_h = delta_t / 3600.0  # hours per step

    has_sens = du_dp is not None

    # Build a shared time axis in hours for the closed-loop data.
    t_cl = steps * dt_h

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_extra_states = nx - 1  # states other than T_room (index 0)
    n_ts_rows = 2 + max(n_extra_states, 0)  # time-series rows
    n_rows = n_ts_rows + (1 if has_sens else 0)
    # Sensitivity panel is taller than time-series panels to give the matrix room.
    height_ratios = [1] * n_ts_rows + ([1.6] if has_sens else [])
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(12, 3 * n_ts_rows + (4 if has_sens else 0)),
        sharex=False,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    ax_room = axes[0, 0]
    ax_ctrl = axes[1, 0]
    ax_extra = [axes[i + 2, 0] for i in range(n_extra_states)]
    ax_sens = axes[n_ts_rows, 0] if has_sens else None

    plt.subplots_adjust(bottom=0.12, hspace=0.45)

    # ── Static closed-loop background ─────────────────────────────────────────
    ax_room.fill_between(
        t_cl,
        df["T_set_lower"],
        df["T_set_upper"],
        alpha=0.10,
        color="green",
        label="comfort band",
    )
    ax_room.plot(t_cl, df["T_room"], color="tab:blue", lw=1.2, label="T_room actual")
    ax_room.step(t_cl, df["T_set_lower"], color="k", ls="--", lw=0.8, label="T_set_lower")
    ax_room.step(t_cl, df["T_set_upper"], color="k", ls=":", lw=0.8, label="T_set_upper")
    ax_room.set_ylabel("Temperature [°C]")
    ax_room.set_title("Room temperature — actual & MPC prediction")
    ax_room.legend(fontsize=7, loc="upper right")
    ax_room.grid(True, alpha=0.4)

    ax_ctrl.plot(t_cl, df["T_hp_sup"], color="tab:orange", lw=1.2, label="T_hp_sup actual")
    ax_ctrl.set_ylabel("T_HP supply [°C]")
    ax_ctrl.set_title("Heat-pump supply — actual & MPC plan")
    ax_ctrl.legend(fontsize=7, loc="upper right")
    ax_ctrl.grid(True, alpha=0.4)

    for i, ax in enumerate(ax_extra):
        state_idx = i + 1
        label = state_names[state_idx] if state_idx < len(state_names) else f"x[{state_idx}]"
        if label in df.columns:
            ax.plot(t_cl, df[label], lw=1.2, label=f"{label} actual")
        ax.set_ylabel(f"{label} [°C]")
        ax.set_title(f"{label} — actual & MPC prediction")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.4)

        # Fix y-limits from full data range (closed-loop + all MPC predictions).
        all_vals = [x[:, :, state_idx].ravel()]
        if label in df.columns:
            all_vals.append(df[label].to_numpy())
        vmin = min(v.min() for v in all_vals)
        vmax = max(v.max() for v in all_vals)
        margin = max((vmax - vmin) * 0.05, 0.5)
        ax.set_ylim(vmin - margin, vmax + margin)

    # ── Sensitivity matrix panel (static layout, dynamic data) ───────────────
    im_sens = None
    if has_sens:
        # Fixed symmetric colour scale across all timesteps for stable comparison.
        vmax = float(np.abs(du_dp).max())
        if vmax == 0.0:
            vmax = 1.0
        im_sens = ax_sens.imshow(
            du_dp[0],
            aspect="auto",
            origin="upper",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax_sens.set_xlabel("Qdot_gains stage  j")
        ax_sens.set_ylabel("T_HP stage  k")
        ax_sens.set_title(
            r"$\partial T_\mathrm{HP}[k]\,/\,\partial\dot{Q}_\mathrm{gains}[j]$"
            "  [degC/W]  — sensitivity matrix at current step"
        )
        # Tick every stage
        ax_sens.set_xticks(range(N + 1))
        ax_sens.set_yticks(range(N))
        ax_sens.set_xticklabels(range(N + 1), fontsize=7)
        ax_sens.set_yticklabels(range(N), fontsize=7)
        fig.colorbar(im_sens, ax=ax_sens, label="degC/W", shrink=0.8, pad=0.02)

    # ── Dynamic MPC prediction elements (updated by slider) ───────────────────
    horizon_t = np.arange(N_plus1) * dt_h  # relative time offsets [h]

    # Vertical cursor line in each axis
    (cursor_room,) = ax_room.plot([], [], color="gray", lw=0.8, ls="-", alpha=0.6)
    (cursor_ctrl,) = ax_ctrl.plot([], [], color="gray", lw=0.8, ls="-", alpha=0.6)
    cursor_extras = [ax.plot([], [], color="gray", lw=0.8, ls="-", alpha=0.6)[0] for ax in ax_extra]

    # MPC predicted T_room
    (pred_room,) = ax_room.plot(
        [],
        [],
        color="tab:blue",
        lw=1.5,
        ls="--",
        marker=".",
        markersize=3,
        label="MPC pred T_room",
        zorder=5,
    )
    ax_room.legend(fontsize=7, loc="upper right")

    # MPC planned T_HP (step-wise for each horizon interval)
    (pred_ctrl,) = ax_ctrl.plot(
        [],
        [],
        color="tab:orange",
        lw=1.5,
        ls="--",
        marker=".",
        markersize=3,
        label="MPC plan T_HP",
        zorder=5,
    )
    ax_ctrl.legend(fontsize=7, loc="upper right")

    # Extra states
    pred_extras = [
        ax.plot([], [], lw=1.5, ls="--", marker=".", markersize=3, zorder=5)[0] for ax in ax_extra
    ]

    def _update(t_idx: int) -> None:
        t0 = t_cl[t_idx]
        h_t = t0 + horizon_t  # absolute horizon time [h]

        # Room temperature prediction (state index 0)
        pred_room.set_data(h_t, x[t_idx, :, 0])

        # Control trajectory (nu channels; show first)
        pred_ctrl.set_data(t0 + np.arange(N) * dt_h, u[t_idx, :, 0])

        # Extra states
        for i, (line, ax) in enumerate(zip(pred_extras, ax_extra)):
            state_idx = i + 1
            line.set_data(h_t, x[t_idx, :, state_idx])
            ax.relim()
            ax.autoscale_view(scaley=True, scalex=False)

        # Cursor lines
        y_room = ax_room.get_ylim()
        y_ctrl = ax_ctrl.get_ylim()
        cursor_room.set_data([t0, t0], y_room)
        cursor_ctrl.set_data([t0, t0], y_ctrl)
        for cur, ax in zip(cursor_extras, ax_extra):
            cur.set_data([t0, t0], ax.get_ylim())

        # Extend x-axis to show the full horizon if it goes beyond current data
        x_max = max(t_cl[-1], h_t[-1]) + dt_h
        x_min = t_cl[0] - dt_h
        for ax in [ax_room, ax_ctrl] + ax_extra:
            ax.set_xlim(x_min, x_max)

        # Update sensitivity matrix
        if im_sens is not None:
            im_sens.set_data(du_dp[t_idx])

        fig.canvas.draw_idle()

    # ── Slider ────────────────────────────────────────────────────────────────
    ax_slider = fig.add_axes([0.12, 0.03, 0.76, 0.025])
    slider = Slider(ax_slider, "Step", 0, T - 1, valinit=0, valstep=1)

    def _on_slide(val: float) -> None:
        _update(int(val))

    slider.on_changed(_on_slide)
    _update(0)

    # ── Save or show ──────────────────────────────────────────────────────────
    if save_path is not None:
        from matplotlib.animation import FuncAnimation, PillowWriter

        def _animate(frame: int):
            slider.set_val(frame)

        anim = FuncAnimation(fig, _animate, frames=T, interval=100)
        suffix = save_path.suffix.lower()
        if suffix == ".gif":
            anim.save(save_path, writer=PillowWriter(fps=10))
        else:
            anim.save(save_path, fps=10)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Render i4b baseline MPC trajectories.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output directory from run_baseline.py.  Picks the latest NPZ/CSV if omitted.",
    )
    src.add_argument(
        "--csv",
        type=Path,
        default=None,
        dest="csv_path",
        help="Path to val_timeseries_step*.csv (requires --npz).",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        dest="npz_path",
        help="Path to val_mpc_trajectories_step*.npz.",
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=900.0,
        help="Sampling interval [s].",
    )
    parser.add_argument(
        "--state-names",
        nargs="+",
        default=None,
        help="Override state dimension names (default: read from NPZ).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save animation to FILE (.gif or .mp4) instead of showing interactively.",
    )

    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None and args.csv_path is None:
        run_dir = _find_latest_run()
        print(f"Auto-detected run directory: {run_dir}")

    df, x, u, state_names, du_dp = load_data(run_dir, args.csv_path, args.npz_path)
    if args.state_names is not None:
        state_names = args.state_names
    if du_dp is not None:
        print(f"Loaded sensitivity du_dp, shape {du_dp.shape}")
    render(df, x, u, state_names, args.delta_t, args.save, du_dp)
