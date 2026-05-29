"""Closed-loop smoke test: RaceCarPlanner driving RaceCarEnv for one lap.

Equivalent to the acados ``main.py`` for the race_cars example. Saves a Cartesian trajectory
plot and a 6-state / 2-control time series so the result can be visually compared to the
acados baseline figure.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from leap_c.examples.race_car.bicycle_model import frenet_to_cartesian, get_track
from leap_c.examples.race_car.env import RaceCarEnv, RaceCarEnvConfig
from leap_c.examples.race_car.planner import RaceCarPlanner, RaceCarPlannerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("output/race_car_closed_loop"))
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sref-lookahead", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = RaceCarEnvConfig(max_steps=args.max_steps)
    env = RaceCarEnv(cfg=env_cfg)
    planner_cfg = RaceCarPlannerConfig(sref_lookahead=args.sref_lookahead)
    planner = RaceCarPlanner(cfg=planner_cfg)

    obs_np, _ = env.reset(seed=args.seed)
    ctx = None
    states: list[np.ndarray] = [obs_np.copy()]
    actions: list[np.ndarray] = []
    solve_times: list[float] = []
    statuses: list[int] = []

    print(
        f"Starting closed loop. dt={env_cfg.dt}s, N={planner_cfg.N_horizon}, "
        f"lookahead={planner_cfg.sref_lookahead}m"
    )
    for step in range(args.max_steps):
        obs_t = torch.tensor(obs_np, dtype=torch.float64).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            ctx, u_traj_pred_init, x_traj, u_traj, _value = planner(obs_t, ctx=ctx)
        solve_times.append(time.perf_counter() - t0)
        statuses.append(int(getattr(ctx, "status", np.array([0])).item()))
        action = u_traj[0, 0, :].cpu().numpy().astype(np.float64)
        actions.append(action.copy())

        obs_np, _r, term, trunc, info = env.step(action)
        states.append(obs_np.copy())

        if step % 50 == 0:
            print(
                f"  step {step:4d}  s={obs_np[0]:+.3f}m  n={obs_np[1] * 100:+5.1f}cm  "
                f"v={obs_np[3]:.3f}m/s  solve={solve_times[-1] * 1e3:.1f}ms"
            )
        if term or trunc:
            kind = "terminated (lap done)" if term else "truncated"
            print(f"  -> {kind} at step {step + 1}; info={info}")
            break

    states_arr = np.asarray(states)
    actions_arr = np.asarray(actions)
    n_steps = actions_arr.shape[0]
    t_arr = np.arange(n_steps + 1) * env_cfg.dt

    print(f"\nLap summary: {n_steps} steps, {n_steps * env_cfg.dt:.2f} s simulated.")
    print(f"  s range: [{states_arr[:, 0].min():.3f}, {states_arr[:, 0].max():.3f}] m")
    print(
        f"  |n| max: {np.abs(states_arr[:, 1]).max() * 100:.2f} cm "
        f"(limit {env_cfg.n_max * 100:.0f} cm)"
    )
    print(f"  v mean / max: {states_arr[:, 3].mean():.3f} / {states_arr[:, 3].max():.3f} m/s")
    print(
        f"  solve time mean / max: {np.mean(solve_times) * 1e3:.2f} / "
        f"{np.max(solve_times) * 1e3:.2f} ms"
    )
    nz = sum(1 for s in statuses if s != 0)
    print(f"  non-zero solver statuses: {nz} / {len(statuses)}")

    np.savez(
        args.output_dir / "trajectory.npz",
        states=states_arr,
        actions=actions_arr,
        t=t_arr,
        solve_times_s=np.asarray(solve_times),
        statuses=np.asarray(statuses),
    )

    # ---------- Cartesian view of the lap on the track ----------
    sref, xref, yref, psiref, _ = get_track(env_cfg.track_file)
    nx_, ny_ = -np.sin(psiref), np.cos(psiref)
    inner = np.stack([xref + env_cfg.n_max * nx_, yref + env_cfg.n_max * ny_], axis=1)
    outer = np.stack([xref - env_cfg.n_max * nx_, yref - env_cfg.n_max * ny_], axis=1)
    x_traj_cart, y_traj_cart, _ = frenet_to_cartesian(
        states_arr[:, 0], states_arr[:, 1], sref, xref, yref, psiref
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(xref, yref, "k--", lw=0.4, label="centerline")
    ax.plot(*inner.T, "k-", lw=0.6)
    ax.plot(*outer.T, "k-", lw=0.6)
    sc = ax.scatter(x_traj_cart, y_traj_cart, c=states_arr[:, 3], s=4, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="v [m/s]")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Closed-loop lap ({n_steps} steps, {n_steps * env_cfg.dt:.2f}s)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "trajectory_cartesian.png", dpi=150)
    plt.close(fig)

    # ---------- Time series of states and controls ----------
    state_names = ["s [m]", "n [m]", "alpha [rad]", "v [m/s]", "D [-]", "delta [rad]"]
    action_names = ["derD [1/s]", "derDelta [rad/s]"]
    fig, axes = plt.subplots(8, 1, figsize=(9, 12), sharex=True)
    for i, name in enumerate(state_names):
        axes[i].plot(t_arr, states_arr[:, i], lw=1.2)
        axes[i].set_ylabel(name)
        axes[i].grid(alpha=0.3)
    for j, name in enumerate(action_names):
        axes[6 + j].step(t_arr[:-1], actions_arr[:, j], where="post", lw=1.2)
        axes[6 + j].set_ylabel(name)
        axes[6 + j].grid(alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Race-car closed-loop states / controls")
    fig.tight_layout()
    fig.savefig(args.output_dir / "trajectory_timeseries.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
