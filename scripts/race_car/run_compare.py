"""Side-by-side benchmark: RaceCarPlanner vs MpccPlanner on the same RaceCarEnv.

Builds two independent envs with the same seed and runs each planner closed-loop
for one lap. Logs per-step state, action, solve time, and solver status to a
joint NPZ; emits a comparison overview figure (Cartesian track plot + lap
metrics bar chart).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from leap_c.examples.race_car.bicycle_model import frenet_to_cartesian, get_track
from leap_c.examples.race_car.env import RaceCarEnv, RaceCarEnvConfig
from leap_c.examples.race_car.mpcc_planner import MpccPlanner, MpccPlannerConfig
from leap_c.examples.race_car.planner import RaceCarPlanner, RaceCarPlannerConfig


@dataclass
class RunResult:
    label: str
    states: np.ndarray  # (T+1, 6)
    actions: np.ndarray  # (T, 2)
    solve_times: np.ndarray  # (T,)
    statuses: np.ndarray  # (T,)
    dt: float
    success: bool


def _run(
    label: str,
    planner_factory: Callable[[], object],
    env_cfg: RaceCarEnvConfig,
    seed: int,
    max_steps: int,
) -> RunResult:
    env = RaceCarEnv(cfg=env_cfg)
    planner = planner_factory()

    obs_np, _ = env.reset(seed=seed)
    ctx = None
    states: list[np.ndarray] = [obs_np.copy()]
    actions: list[np.ndarray] = []
    solve_times: list[float] = []
    statuses: list[int] = []

    success = False
    for step in range(max_steps):
        obs_t = torch.tensor(obs_np, dtype=torch.float64).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            ctx, _u0, _x_plan, u_plan, _value = planner(obs_t, ctx=ctx)  # type: ignore[operator]
        solve_times.append(time.perf_counter() - t0)
        statuses.append(int(getattr(ctx, "status", np.array([0])).item()))
        action = u_plan[0, 0, :].cpu().numpy().astype(np.float64)
        actions.append(action.copy())

        obs_np, _r, term, trunc, info = env.step(action)
        states.append(obs_np.copy())
        if term or trunc:
            success = term and not trunc
            print(f"[{label}] step {step + 1}: term={term}, trunc={trunc}, info={info}")
            break

    return RunResult(
        label=label,
        states=np.asarray(states),
        actions=np.asarray(actions),
        solve_times=np.asarray(solve_times),
        statuses=np.asarray(statuses),
        dt=env_cfg.dt,
        success=success,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("output/race_car_compare"))
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mpcc-frame",
        choices=("cartesian", "frenet"),
        default="cartesian",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = RaceCarEnvConfig(max_steps=args.max_steps)

    runs = [
        _run(
            "race_car",
            lambda: RaceCarPlanner(cfg=RaceCarPlannerConfig()),
            env_cfg,
            args.seed,
            args.max_steps,
        ),
        _run(
            f"race_car_mpcc_{args.mpcc_frame}",
            lambda: MpccPlanner(cfg=MpccPlannerConfig(frame=args.mpcc_frame)),
            env_cfg,
            args.seed,
            args.max_steps,
        ),
    ]

    print("\n=== Summary ===")
    print(f"{'planner':30s} {'lap time [s]':>12s} {'|n|_max [cm]':>14s} {'solve [ms]':>12s}")
    for r in runs:
        n_steps = r.actions.shape[0]
        lap_time = n_steps * r.dt
        n_max_cm = float(np.abs(r.states[:, 1]).max()) * 100.0
        solve_ms = float(np.mean(r.solve_times)) * 1e3
        print(f"{r.label:30s} {lap_time:12.2f} {n_max_cm:14.2f} {solve_ms:12.2f}")

    np.savez(
        args.output_dir / "compare.npz",
        **{f"{r.label}_states": r.states for r in runs},
        **{f"{r.label}_actions": r.actions for r in runs},
        **{f"{r.label}_solve_times_s": r.solve_times for r in runs},
        **{f"{r.label}_statuses": r.statuses for r in runs},
    )

    sref, xref, yref, psiref, _ = get_track(env_cfg.track_file)
    nx_, ny_ = -np.sin(psiref), np.cos(psiref)
    inner = np.stack([xref + env_cfg.n_max * nx_, yref + env_cfg.n_max * ny_], axis=1)
    outer = np.stack([xref - env_cfg.n_max * nx_, yref - env_cfg.n_max * ny_], axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(xref, yref, "k--", lw=0.4, label="centerline")
    ax.plot(*inner.T, "k-", lw=0.6)
    ax.plot(*outer.T, "k-", lw=0.6)
    colors = ("tab:blue", "tab:red")
    for r, c in zip(runs, colors):
        x_c, y_c, _ = frenet_to_cartesian(r.states[:, 0], r.states[:, 1], sref, xref, yref, psiref)
        ax.plot(x_c, y_c, "-", color=c, lw=1.2, label=r.label)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Closed-loop trajectories")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_dir / "compare_trajectories.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [r.label for r in runs]
    lap_times = [r.actions.shape[0] * r.dt for r in runs]
    n_maxes = [float(np.abs(r.states[:, 1]).max()) * 100.0 for r in runs]
    solves = [float(np.mean(r.solve_times)) * 1e3 for r in runs]
    axes[0].bar(labels, lap_times, color=colors)
    axes[0].set_ylabel("lap time [s]")
    axes[1].bar(labels, n_maxes, color=colors)
    axes[1].set_ylabel("|n|_max [cm]")
    axes[2].bar(labels, solves, color=colors)
    axes[2].set_ylabel("mean solve [ms]")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(alpha=0.3)
    fig.suptitle("Planner comparison")
    fig.tight_layout()
    fig.savefig(args.output_dir / "compare_metrics.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
