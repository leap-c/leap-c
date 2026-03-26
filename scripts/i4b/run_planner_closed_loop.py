"""Closed-loop MPC simulation with I4bPlanner and I4bEnv.

Run from the repo root:
    python scripts/i4b/run_planner_closed_loop.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

from leap_c.examples.i4b.env import I4bEnv, I4bEnvConfig
from leap_c.examples.i4b.planner import I4bPlanner, I4bPlannerConfig

# ── i4b path setup ────────────────────────────────────────────────────────────
_I4B_ROOT = Path(__file__).resolve().parents[2] / "external" / "i4b"
if str(_I4B_ROOT) not in sys.path:
    sys.path.insert(0, str(_I4B_ROOT))
os.chdir(_I4B_ROOT)

from src.gym_interface import BUILDING_NAMES2CLASS  # noqa: E402
from src.models.model_hvac import Heatpump_AW  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────
BUILDING = "sfh_2016_now_0_soc"
METHOD = "4R3C"
MDOT_HP = 0.25
DELTA_T = 900  # 15-min steps
N_HORIZON = 12  # 3-hour horizon (shorter for a quick demo)
DAYS = 1

# ── Shared models (same objects for env and planner) ───────────────────────────
hp_model = Heatpump_AW(mdot_HP=MDOT_HP)

env_cfg = I4bEnvConfig(
    building_params=BUILDING_NAMES2CLASS[BUILDING],
    hp_model=hp_model,
    method=METHOD,
    mdot_hp=MDOT_HP,
    delta_t=DELTA_T,
    days=DAYS,
    T_set_lower=20.0,
    T_set_upper=26.0,
)
env = I4bEnv(cfg=env_cfg)

planner_cfg = I4bPlannerConfig(
    N_horizon=N_HORIZON,
    delta_t=float(DELTA_T),
    ws=1e-1,
)
planner = I4bPlanner(
    building_model=env.bldg_model,
    hp_model=hp_model,
    cfg=planner_cfg,
)

# ── Closed-loop rollout ────────────────────────────────────────────────────────
obs_np, _ = env.reset(seed=0)
ctx = None

T_room_log = []
T_HP_log = []
reward_log = []

steps_per_day = 24 * int(3600 / DELTA_T)
max_steps = DAYS * steps_per_day

print(f"Running {DAYS} day(s) = {max_steps} steps with N_horizon={N_HORIZON} ...")

for step in range(max_steps):
    obs_t = torch.tensor(obs_np, dtype=torch.float64).unsqueeze(0)  # (1, obs_dim)

    with torch.no_grad():
        ctx, u0_norm, x_traj, u_traj, value = planner(obs_t, ctx=ctx)

    action_np = u0_norm.squeeze(0).cpu().numpy()  # (1,)
    obs_np, reward, terminated, truncated, info = env.step(action_np)

    T_room_log.append(info["T_room"])
    T_HP_log.append(info["T_hp_sup"])
    reward_log.append(reward)

    if step % 16 == 0:
        print(
            f"  step {step:4d} | T_room={info['T_room']:.2f} degC | "
            f"T_HP={info['T_hp_sup']:.1f} degC | E_el={info['E_el_kWh'] * 1000:.1f} Wh"
        )

    if terminated or truncated:
        break

print(f"\nTotal reward (negative energy): {sum(reward_log):.4f} kWh")

# ── Plot ───────────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    t = np.arange(len(T_room_log)) * DELTA_T / 3600

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax0 = axes[0]
    ax0.plot(t, T_room_log, label="T_room")
    ax0.axhline(env_cfg.T_set_lower, color="k", linestyle="--", linewidth=0.8, label="T_set_lower")
    ax0.axhline(env_cfg.T_set_upper, color="r", linestyle="--", linewidth=0.8, label="T_set_upper")
    ax0.set_ylabel("Temperature [degC]")
    ax0.legend(fontsize=8)
    ax0.set_title(f"I4b MPC closed-loop | {METHOD} | N_horizon={N_HORIZON}")
    ax0.grid(True)

    ax1 = axes[1]
    ax1.step(t, T_HP_log, where="post", label="T_HP supply")
    ax1.set_ylabel("T_HP [degC]")
    ax1.set_xlabel("Time [h]")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    plt.tight_layout()
    out_path = Path("outputs/i4b_planner_closed_loop.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")
    plt.show()
except ImportError:
    print("matplotlib not available, skipping plot.")
