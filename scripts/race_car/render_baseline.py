"""Interactive rendering of race-car baseline channel logs.

Loads ``val_log_step*.npz`` + ``val_log_step*.json`` produced by
``run_baseline.py`` and builds an interactive figure. The subplots adapt to
whichever channels are present: add or remove channels in ``channels.py`` and
re-run — this script requires no changes.

Panel rules (one subplot per "panel" group):
- ``line`` panel (any channel with scalar/sequence data) — static closed-loop
  history + dashed overlay driven by the slider.
- ``matrix`` panel (channel with a matrix field) — imshow of the frame at the
  selected step, fixed symmetric colour scale.

The x-axis is shown in seconds: ``delta_t_s`` from the channel log header is
pre-scaled by 3600 before passing to :func:`render`, which internally divides
by 3600 to compute ``dt_h``. This yields the natural time unit for race-car
closed-loop runs.

Usage
-----
    python scripts/race_car/render_baseline.py --run-dir <output/...>
    python scripts/race_car/render_baseline.py --run-dir <dir> --save lap.gif
"""

from __future__ import annotations

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from leap_c.examples.race_car.bicycle_model import (
    DEFAULT_TRACK_FILE,
    N_MAX_DEFAULT,
    frenet_to_cartesian,
    get_track,
)

Updater = Callable[[int], None]


@dataclass
class Panel:
    name: str
    kind: str  # "line" | "matrix"
    channels: list[dict]  # channel metadata dicts


def _find_latest_run(output_root: Path = Path("output")) -> Path:
    candidates = sorted(
        (p for p in output_root.rglob("val_log_step*.npz")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No val_log_step*.npz under {output_root}")
    return candidates[-1].parent


def load(run_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    npz_candidates = sorted(run_dir.glob("val_log_step*.npz"))
    json_candidates = sorted(run_dir.glob("val_log_step*.json"))
    if not npz_candidates or not json_candidates:
        raise FileNotFoundError(f"Missing val_log files in {run_dir}")
    arrays = dict(np.load(npz_candidates[-1], allow_pickle=False))
    metadata = json.loads(json_candidates[-1].read_text())
    return arrays, metadata


def _group_panels(channel_meta: list[dict]) -> list[Panel]:
    order: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for ch in channel_meta:
        if not ch["kinds"]:
            continue  # channel produced no data this run
        p = ch["panel"]
        if p not in grouped:
            grouped[p] = []
            order.append(p)
        grouped[p].append(ch)

    panels: list[Panel] = []
    for name in order:
        chans = grouped[name]
        kind = "matrix" if any("matrix" in c["kinds"] for c in chans) else "line"
        panels.append(Panel(name=name, kind=kind, channels=chans))
    return panels


def _build_line_panel(
    ax,
    panel: Panel,
    arrays: dict[str, np.ndarray],
    t_cl: np.ndarray,
    dt_h: float,
) -> Updater | None:
    """Draw statics and return an updater closure for slider-driven overlays."""
    ylabels = [c["ylabel"] for c in panel.channels if c["ylabel"]]

    for ch in panel.channels:
        name = ch["name"]
        if "scalar" in ch["kinds"]:
            key = ch["keys"]["scalar"]
            ax.plot(t_cl, arrays[key], lw=1.2, label=name)
        if "scalars_dict" in ch["kinds"]:
            prefix = f"{name}."
            for k in sorted(key for key in arrays if key.startswith(prefix)):
                ax.plot(t_cl, arrays[k], lw=0.9, label=k[len(prefix) :])

    # Dynamic overlays for sequence channels
    overlay_lines: list[tuple[dict, Any]] = []
    for ch in panel.channels:
        if "sequence" not in ch["kinds"]:
            continue
        (line,) = ax.plot(
            [], [], lw=1.5, ls="--", marker=".", markersize=3, label=f"{ch['name']} pred"
        )
        overlay_lines.append((ch, line))

    # ── Fix y-limits to cover every value the slider can expose ──
    # Gather every scalar / sequence array drawn on this panel (including the
    # scalars_dict fan-outs) and compute a stable y-range with a small margin.
    pools: list[np.ndarray] = []
    for ch in panel.channels:
        if "scalar" in ch["kinds"]:
            pools.append(arrays[ch["keys"]["scalar"]])
        if "sequence" in ch["kinds"]:
            pools.append(arrays[ch["keys"]["sequence"]].ravel())
        if "scalars_dict" in ch["kinds"]:
            prefix = f"{ch['name']}."
            pools.extend(arrays[k] for k in arrays if k.startswith(prefix))
    if pools:
        all_vals = np.concatenate([p.ravel() for p in pools])
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size:
            vmin, vmax = float(finite.min()), float(finite.max())
            margin = max((vmax - vmin) * 0.05, 0.5 if vmax - vmin < 1e-9 else 0.0)
            ax.set_ylim(vmin - margin, vmax + margin)
            ax.set_autoscaley_on(False)

    # Cursor — spans the fixed y-range.
    y_lo, y_hi = ax.get_ylim()
    (cursor,) = ax.plot([y_lo, y_hi], [y_lo, y_hi], color="gray", lw=0.8, alpha=0.6)

    ax.set_title(panel.name)
    if ylabels:
        ax.set_ylabel(ylabels[0])
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.4)

    if not overlay_lines:

        def _update_scalar_only(t_idx: int) -> None:
            t0 = t_cl[t_idx]
            cursor.set_data([t0, t0], [y_lo, y_hi])

        return _update_scalar_only

    def _update(t_idx: int) -> None:
        t0 = t_cl[t_idx]
        for ch, line in overlay_lines:
            arr = arrays[ch["keys"]["sequence"]][t_idx]  # (K,)
            K = arr.shape[0]
            horizon_t = t0 + np.arange(K) * dt_h
            line.set_data(horizon_t, arr)
        cursor.set_data([t0, t0], [y_lo, y_hi])

    return _update


def _build_matrix_panel(
    ax,
    panel: Panel,
    arrays: dict[str, np.ndarray],
) -> Updater:
    # Use the first matrix-kind channel in the panel. Overlapping a matrix
    # with other kinds in the same panel is not supported.
    ch = next(c for c in panel.channels if "matrix" in c["kinds"])
    stacked = arrays[ch["keys"]["matrix"]]  # (T, H, W)
    vmax = float(np.abs(stacked).max()) or 1.0
    im = ax.imshow(
        stacked[0],
        aspect="auto",
        origin="upper",
        cmap=ch.get("cmap", "RdBu_r"),
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(ch.get("ylabel") or ch["name"])
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    def _update(t_idx: int) -> None:
        im.set_data(stacked[t_idx])

    return _update


def _build_track_panel(
    ax,
    arrays: dict[str, np.ndarray],
    track_file: Path,
    n_max: float,
    color_plan_velocity: bool,
    fig,
) -> Updater | None:
    """Top-down Cartesian view: track corridor + velocity-colored trail + MPC plan.

    Returns None if the required Frenet arrays (s, n, v) are missing so the
    caller can fall back to the legacy layout.
    """
    if not {"s", "n", "v"}.issubset(arrays):
        return None

    sref, xref, yref, psiref, _ = get_track(track_file)
    pathlength = float(sref[-1])

    nx = -np.sin(psiref)
    ny = np.cos(psiref)
    inner_x = xref + n_max * nx
    inner_y = yref + n_max * ny
    outer_x = xref - n_max * nx
    outer_y = yref - n_max * ny
    ax.plot(xref, yref, "k--", lw=0.4, label="centerline")
    ax.plot(inner_x, inner_y, "k-", lw=0.6)
    ax.plot(outer_x, outer_y, "k-", lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Race car - Cartesian view")

    for i in range(int(pathlength) + 1):
        k = int(np.argmin(np.abs(sref - i)))
        s_k = float(sref[k])
        lx, ly, _ = frenet_to_cartesian(s_k, n_max + 0.06, sref, xref, yref, psiref)
        tin_x, tin_y, _ = frenet_to_cartesian(s_k, n_max, sref, xref, yref, psiref)
        tout_x, tout_y, _ = frenet_to_cartesian(s_k, -n_max, sref, xref, yref, psiref)
        ax.plot(
            [np.asarray(tin_x).item(), np.asarray(tout_x).item()],
            [np.asarray(tin_y).item(), np.asarray(tout_y).item()],
            "k-",
            lw=0.3,
            alpha=0.5,
        )
        ax.text(
            np.asarray(lx).item(),
            np.asarray(ly).item(),
            f"{i}m",
            fontsize=6,
            ha="center",
            va="center",
            alpha=0.7,
        )

    traj_scatter = ax.scatter(
        [],
        [],
        c=[],
        cmap="rainbow",
        s=4,
        edgecolor="none",
        vmin=0.0,
        vmax=2.0,
        label="trajectory",
    )
    cbar = fig.colorbar(traj_scatter, ax=ax, fraction=0.035)
    cbar.set_label("v [m/s]")

    have_plan = {"s_hrz", "n_hrz"}.issubset(arrays)
    use_plan_scatter = color_plan_velocity and have_plan and "v_hrz" in arrays
    plan_scatter = None
    plan_line = None
    if have_plan:
        if use_plan_scatter:
            plan_scatter = ax.scatter(
                [],
                [],
                c=[],
                cmap="rainbow",
                s=12,
                edgecolor="none",
                vmin=0.0,
                vmax=2.0,
                label="MPC plan",
            )
        else:
            (plan_line,) = ax.plot(
                [], [], "g--", lw=1.0, marker=".", markersize=3, label="MPC plan"
            )

    (car_dot,) = ax.plot([], [], "ro", ms=8, label="car")
    ax.legend(loc="upper right", fontsize=8)

    def _update(t_idx: int) -> None:
        s_past = arrays["s"][: t_idx + 1]
        n_past = arrays["n"][: t_idx + 1]
        x_past, y_past, _ = frenet_to_cartesian(s_past, n_past, sref, xref, yref, psiref)
        traj_scatter.set_offsets(np.c_[np.atleast_1d(x_past), np.atleast_1d(y_past)])
        traj_scatter.set_array(np.asarray(arrays["v"][: t_idx + 1]))

        cx, cy, _ = frenet_to_cartesian(
            float(arrays["s"][t_idx]),
            float(arrays["n"][t_idx]),
            sref,
            xref,
            yref,
            psiref,
        )
        car_dot.set_data([np.asarray(cx).item()], [np.asarray(cy).item()])

        if have_plan:
            s_h = arrays["s_hrz"][t_idx]
            n_h = arrays["n_hrz"][t_idx]
            px, py, _ = frenet_to_cartesian(s_h, n_h, sref, xref, yref, psiref)
            if use_plan_scatter:
                plan_scatter.set_offsets(np.c_[np.atleast_1d(px), np.atleast_1d(py)])
                plan_scatter.set_array(np.asarray(arrays["v_hrz"][t_idx]))
            else:
                plan_line.set_data(px, py)

    return _update


def _decide_right_cols(flag: str, n_panels: int) -> int:
    if flag == "auto":
        return 1 if n_panels <= 6 else 2
    return int(flag)


def render(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    save_path: Path | None,
    *,
    show_track: bool = True,
    color_plan_velocity: bool = True,
    track_file: Path = DEFAULT_TRACK_FILE,
    n_max: float = N_MAX_DEFAULT,
    right_cols_flag: str = "auto",
) -> None:
    header = metadata["header"]
    dt_h = float(header["delta_t_s"]) / 3600.0
    N = int(header["N_horizon"])

    panels = _group_panels(metadata["channels"])
    if not panels:
        raise RuntimeError("No channels with data to render.")

    # Determine number of closed-loop steps from the first scalar array found.
    T = _infer_T(arrays, metadata)
    # Cursor at slider s sits at s*dt — the pre-step time of the s+1-th MPC
    # tick. Scalar[s] and plan.x[0] from tick s+1 are the same physical value
    # by construction (both are the MPC's x0).
    t_cl = np.arange(T) * dt_h

    have_track_data = {"s", "n", "v"}.issubset(arrays)
    track_enabled = show_track and have_track_data
    right_cols = _decide_right_cols(right_cols_flag, len(panels))

    if track_enabled:
        R = ceil(len(panels) / right_cols)
        fig_w = 7.0 + 5.0 * right_cols
        fig_h = min(14.0, max(6.0, 1.8 * R + 1.0))
        fig = plt.figure(figsize=(fig_w, fig_h))
        outer = fig.add_gridspec(
            R,
            1 + right_cols,
            width_ratios=[2.0] + [1.0] * right_cols,
            hspace=0.55,
            wspace=0.30,
        )
        track_ax = fig.add_subplot(outer[:, 0])
        right_axes = []
        for i in range(len(panels)):
            r, c = divmod(i, right_cols)
            right_axes.append(fig.add_subplot(outer[r, 1 + c]))
    else:
        height_ratios = [1.6 if p.kind == "matrix" else 1.0 for p in panels]
        n_line = sum(1 for p in panels if p.kind == "line")
        n_mat = sum(1 for p in panels if p.kind == "matrix")
        figsize = (12, max(3.0, 2.2 * n_line + 3.5 * n_mat + 1.0))
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"height_ratios": height_ratios},
        )
        plt.subplots_adjust(hspace=0.55)
        right_axes = list(axes[:, 0])
        track_ax = None

    plt.subplots_adjust(bottom=0.08)

    updaters: list[Updater] = []
    if track_ax is not None:
        u = _build_track_panel(track_ax, arrays, track_file, n_max, color_plan_velocity, fig)
        if u is not None:
            updaters.append(u)

    for panel, ax in zip(panels, right_axes):
        if panel.kind == "matrix":
            updaters.append(_build_matrix_panel(ax, panel, arrays))
        else:
            u = _build_line_panel(ax, panel, arrays, t_cl, dt_h)
            if u is not None:
                updaters.append(u)

    # Horizon extends x-axis of line panels; match across them.
    x_min = t_cl[0] - dt_h
    x_max = t_cl[-1] + N * dt_h
    for panel, ax in zip(panels, right_axes):
        if panel.kind == "line":
            ax.set_xlim(x_min, x_max)

    ax_slider = fig.add_axes([0.12, 0.02, 0.76, 0.02])
    slider = Slider(ax_slider, "Step", 0, T - 1, valinit=0, valstep=1)

    def _on_slide(val: float) -> None:
        t_idx = int(val)
        for u in updaters:
            u(t_idx)
        fig.canvas.draw_idle()

    slider.on_changed(_on_slide)
    _on_slide(0)

    if save_path is not None:
        from matplotlib.animation import FuncAnimation, PillowWriter

        anim = FuncAnimation(fig, lambda f: slider.set_val(f), frames=T, interval=100)
        if save_path.suffix.lower() == ".gif":
            anim.save(save_path, writer=PillowWriter(fps=10))
        else:
            anim.save(save_path, fps=10)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()


def _infer_T(arrays: dict[str, np.ndarray], metadata: dict) -> int:
    for ch in metadata["channels"]:
        for kind in ("scalar", "sequence", "matrix"):
            if kind in ch["kinds"]:
                return int(arrays[ch["keys"][kind]].shape[0])
    raise RuntimeError("Could not infer T from any channel.")


def main() -> None:
    parser = ArgumentParser(
        description="Render race_car baseline channel log.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=None, metavar="FILE")
    parser.add_argument(
        "--track",
        action=BooleanOptionalAction,
        default=True,
        help="Show the Cartesian track subplot on the left.",
    )
    parser.add_argument(
        "--color-plan-velocity",
        action=BooleanOptionalAction,
        default=True,
        help="Color the MPC plan by predicted velocity (shares colorbar with the past trail).",
    )
    parser.add_argument(
        "--track-file",
        type=Path,
        default=DEFAULT_TRACK_FILE,
        help="Track reference file (columns: s, x, y, psi, kappa).",
    )
    parser.add_argument(
        "--n-max",
        type=float,
        default=N_MAX_DEFAULT,
        help="Track corridor half-width [m] used for drawing the boundary.",
    )
    parser.add_argument(
        "--right-cols",
        choices=["auto", "1", "2"],
        default="auto",
        help="Channel-panel columns on the right (auto = 1 if <=6 panels else 2).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _find_latest_run()
    print(f"Rendering from: {run_dir}")
    arrays, metadata = load(run_dir)

    # Make the x-axis display seconds: render() computes dt_h = delta_t_s / 3600.
    metadata["header"]["delta_t_s"] = float(metadata["header"]["delta_t_s"]) * 3600.0
    render(
        arrays,
        metadata,
        args.save,
        show_track=args.track,
        color_plan_velocity=args.color_plan_velocity,
        track_file=args.track_file,
        n_max=args.n_max,
        right_cols_flag=args.right_cols,
    )


if __name__ == "__main__":
    main()
