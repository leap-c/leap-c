#!/usr/bin/env python3
r"""Sensitivity analysis for all three HVAC OCP parameterisations.

Produces five diagnostic figures per solver variant, saved under
  <output-dir>/<variant>/
  heatmap_du0_dp.pdf   – ∂u₀/∂p across scenario × parameter
  heatmap_dvalue_dp.pdf – ∂J/∂p across scenario × parameter
  state_sweep.pdf      – policy & gradients along a Ti sweep (interior → saturated)
  param_sweep.pdf      – u₀, J, and analytical vs FD ∂u₀/∂pᵢ for each parameter
  fd_check.pdf         – bar chart: analytic vs FD at the default operating point

Solver variants
---------------
  resistance             – original parameterisation (Rhi, Rie, Rea in K/W)
  conductance            – ghi = 1/Rhi, gie = 1/Rie, gea = 1/Rea  [W/K]
  normalized             – all learnable parameters mapped to [−1, 1]
  normalized_conductance – conductance space, then normalised to [−1, 1]
  parameter_linear       – compound params (ahi=1/CiRhi, …) linear in dynamics
  discrete_matrix        – 18 normalised elements of discrete Ad, Bd, Ed

Usage
-----
  python scripts/hvac/analyze_sensitivities.py \\
      --output-dir outputs/sensitivity \\
      --n-horizon 24 \\
      --reuse-code-dir /tmp/acados_hvac
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from leap_c.examples.hvac.acados_ocp import (
    export_parametric_ocp,
    export_parametric_ocp_conductance,
    export_parametric_ocp_discrete_matrix,
    export_parametric_ocp_normalized,
    export_parametric_ocp_normalized_conductance,
    export_parametric_ocp_parameter_linear,
    make_default_hvac_params,
    make_default_hvac_params_conductance,
    make_default_hvac_params_discrete_matrix,
    make_default_hvac_params_normalized,
    make_default_hvac_params_normalized_conductance,
    make_default_hvac_params_parameter_linear,
)
from leap_c.examples.hvac.dynamics import HydronicParameters
from leap_c.examples.hvac.utils import set_temperature_limits
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

# ── default operating conditions ─────────────────────────────────────────────
_QH_DEFAULT = 40  # quarter-hour of day (10:00) → occupied, strict comfort
_TH_DEFAULT = 303.0  # K, radiator temperature
_TE_DEFAULT = 283.0  # K, envelope temperature
_AMBIENT_K = 278.15  # K, 5 °C outdoor
_SOLAR = 50.0  # W/m²
_PRICE = 0.15  # €/kWh

# state component labels
_X_NAMES = ["Ti", "Th", "Te", "qh_prev"]

# sensitivity name → internal acados field name
_SENS_MAP: dict[str, str] = {
    "du0_dp": "du0_dp_global",
    "dx_dp": "dx_dp_global",
    "du_dp": "du_dp_global",
    "dvalue_dp": "dvalue_dp_global",
    "dvalue_daction": "dvalue_du0",
    "du0_dx0": "du0_dx0",
    "dvalue_dx0": "dvalue_dx0",
}


# ── solver wrapper ────────────────────────────────────────────────────────────


class _SolverWrapper:
    """Wraps AcadosDiffMpcTorch + AcadosParameterManager to mimic HvacPlanner.

    Exposes the same interface used by all analysis functions:
      wrapper(obs, param=param)       → (ctx, u0_W, x, u, value)
      wrapper.sensitivity(ctx, name)  → np.ndarray
      wrapper.param_manager           → AcadosParameterManager
    """

    def __init__(
        self,
        diff_mpc: AcadosDiffMpcTorch,
        param_manager: AcadosParameterManager,
        N_horizon: int,
    ) -> None:
        self.diff_mpc = diff_mpc
        self.param_manager = param_manager
        self._N = N_horizon

    def __call__(
        self,
        obs: dict,
        action: torch.Tensor | None = None,
        param: torch.Tensor | None = None,
        ctx=None,
    ):
        """Forward pass replicating the key steps of HvacPlanner.forward."""
        pm = self.param_manager
        quarter_hours = obs["time"]["quarter_hour"]  # (B, 1)
        batch_size = quarter_hours.shape[0]
        device = quarter_hours.device

        # Default learnable params
        if param is None and pm.learnable_parameters.shape[0] > 0:
            flat = torch.from_numpy(pm.learnable_parameters_default.cat.full().flatten()).to(device)
            param = flat.unsqueeze(0).expand(batch_size, -1)

        # qh_prev = 0 (no warm-start context for analysis)
        qh = torch.zeros((batch_size, 1), dtype=torch.float64, device=device)

        # Per-stage comfort bounds
        lb, ub = set_temperature_limits(
            np.array(
                [
                    np.arange(
                        float(quarter_hours[i, 0].cpu()),
                        float(quarter_hours[i, 0].cpu()) + self._N + 1,
                    )
                    % 96
                    for i in range(batch_size)
                ]
            )
        )
        self.diff_mpc.set_constraint_bounds(
            lb[:batch_size, 1:, np.newaxis],
            ub[:batch_size, 1:, np.newaxis],
            list(range(1, self._N + 1)),
        )

        # Non-learnable stagewise params (temperature, solar, price from forecast)
        overwrites: dict = {}
        n_stages = self._N + 1
        for key in ["temperature", "solar", "price"]:
            if key in pm.parameters:
                overwrites[key] = (
                    obs["forecast"][key][:, :n_stages]
                    .reshape(batch_size, -1, 1)
                    .detach()
                    .cpu()
                    .numpy()
                )

        p_stagewise = pm.combine_non_learnable_parameter_values(batch_size=batch_size, **overwrites)

        # Augmented state [Ti, Th, Te, qh_prev]
        state = torch.cat([obs["state"], qh], dim=1)

        diff_ctx, _, x, u, value = self.diff_mpc(state, action, param, p_stagewise, ctx=ctx)

        u0 = u[:, 0, :] * 1000.0  # kW → W, matching HvacPlanner convention
        return diff_ctx, u0, x, u, value

    def sensitivity(self, ctx, name: str) -> np.ndarray:
        if name not in _SENS_MAP:
            raise ValueError(f"Unknown sensitivity name '{name}'. Available: {list(_SENS_MAP)}")
        return self.diff_mpc.sensitivity(ctx, _SENS_MAP[name])

    def eval(self) -> "_SolverWrapper":
        return self


# ── shared helpers ────────────────────────────────────────────────────────────


def _comfort_bounds(quarter_hour: int) -> tuple[float, float]:
    lb, ub = set_temperature_limits(np.array([[quarter_hour]]))
    return float(lb[0, 0]), float(ub[0, 0])


def _make_obs(
    Ti: np.ndarray,
    quarter_hour: int | np.ndarray = _QH_DEFAULT,
    Th: float = _TH_DEFAULT,
    Te: float = _TE_DEFAULT,
    ambient_K: float = _AMBIENT_K,
    solar: float = _SOLAR,
    price: float = _PRICE,
    N_forecast: int = 96,
    device: str = "cpu",
) -> dict:
    """Build a batch observation dict from a 1-D array of Ti values (Kelvin)."""
    B = len(Ti)
    if isinstance(quarter_hour, (int, float)):
        qh_arr = np.full((B, 1), float(quarter_hour))
    else:
        qh_arr = np.asarray(quarter_hour, dtype=float).reshape(B, 1)

    return {
        "time": {
            "quarter_hour": torch.tensor(qh_arr, dtype=torch.float64, device=device),
        },
        "state": torch.tensor(
            np.stack([Ti, np.full(B, Th), np.full(B, Te)], axis=1),
            dtype=torch.float64,
            device=device,
        ),
        "forecast": {
            "temperature": torch.full(
                (B, N_forecast), ambient_K, dtype=torch.float64, device=device
            ),
            "solar": torch.full((B, N_forecast), solar, dtype=torch.float64, device=device),
            "price": torch.full((B, N_forecast), price, dtype=torch.float64, device=device),
        },
    }


def _default_param(solver: _SolverWrapper, B: int, device: str = "cpu") -> torch.Tensor:
    flat = solver.param_manager.learnable_parameters_default.cat.full().flatten()
    return torch.tensor(np.tile(flat, (B, 1)), dtype=torch.float64, device=device)


def _param_info(solver: _SolverWrapper) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    pm = solver.param_manager
    return (
        list(pm.learnable_parameters.keys()),
        pm.learnable_parameters_default.cat.full().flatten(),
        pm.learnable_parameters_lb.cat.full().flatten(),
        pm.learnable_parameters_ub.cat.full().flatten(),
    )


@torch.no_grad()
def _forward(solver: _SolverWrapper, obs: dict, param: torch.Tensor | None = None) -> tuple:
    """Forward pass. Returns (ctx, u0_W_np, value_np)."""
    ctx, u0, _x, _u, value = solver(obs, param=param)
    return ctx, u0.cpu().numpy(), value.cpu().numpy()


def _sens(solver: _SolverWrapper, ctx, name: str) -> np.ndarray:
    return solver.sensitivity(ctx, name)


# ── scenario collection ───────────────────────────────────────────────────────


def _build_scenarios(
    solver: _SolverWrapper, N_forecast: int, device: str
) -> list[tuple[str, dict, torch.Tensor]]:
    keys, defaults, lbs, ubs = _param_info(solver)
    lb_K, ub_K = _comfort_bounds(_QH_DEFAULT)
    Ti_mid = (lb_K + ub_K) / 2.0
    scenarios: list[tuple[str, dict, torch.Tensor]] = []

    def _add(label: str, Ti_K: float, param_idx: int | None = None, param_val: float | None = None):
        obs = _make_obs(np.array([Ti_K]), N_forecast=N_forecast, device=device)
        p = _default_param(solver, 1, device)
        if param_idx is not None and param_val is not None:
            p[0, param_idx] = param_val
        scenarios.append((label, obs, p))

    _add("interior", Ti_mid)
    _add("Ti_at_lb", lb_K)
    _add("Ti_2K_below_lb", lb_K - 2.0)
    _add("Ti_at_ub", ub_K)
    _add("Ti_2K_above_ub", ub_K + 2.0)
    _add("Ti_cold (qh→max)", lb_K - 10.0)
    _add("Ti_hot  (qh→0)", ub_K + 10.0)

    for i, (k, lb, ub) in enumerate(zip(keys, lbs, ubs)):
        if np.isfinite(lb):
            _add(f"{k} @ lb", Ti_mid, i, float(lb))
        if np.isfinite(ub):
            _add(f"{k} @ ub", Ti_mid, i, float(ub))

    return scenarios


# ── Figure 1 & 2 – scenario heatmaps ─────────────────────────────────────────


def plot_heatmaps(solver: _SolverWrapper, scenarios, out_dir: Path, device: str) -> None:
    keys, *_ = _param_info(solver)
    n_scen, n_par = len(scenarios), len(keys)

    du0_mat = np.full((n_scen, n_par), np.nan)
    dval_mat = np.full((n_scen, n_par), np.nan)
    row_labels, statuses = [], []

    for i, (label, obs, param) in enumerate(scenarios):
        row_labels.append(label)
        try:
            ctx, u0, val = _forward(solver, obs, param)
            st = int(ctx.status.flat[0])
            statuses.append(st)
            if st == 0:
                du0 = _sens(solver, ctx, "du0_dp")
                dv = _sens(solver, ctx, "dvalue_dp")
                du0_mat[i] = du0[0].reshape(-1, n_par)[0]
                dval_mat[i] = dv[0].reshape(-1, n_par)[0]
            else:
                print(f"  [WARN] '{label}' solver status={st}")
        except Exception as exc:
            statuses.append(-1)
            print(f"  [ERROR] '{label}': {exc}")

    print(f"  Solver statuses: {dict(zip(row_labels, statuses))}")

    for mat, fname, title in [
        (du0_mat, "heatmap_du0_dp.pdf", "∂u₀/∂p  (policy sensitivity)"),
        (dval_mat, "heatmap_dvalue_dp.pdf", "∂J/∂p   (value sensitivity)"),
    ]:
        vmax = np.nanmax(np.abs(mat)) or 1.0
        fig, ax = plt.subplots(figsize=(max(8, n_par * 1.1), max(5, n_scen * 0.45)))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_par))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_scen))
        ax.set_yticklabels(row_labels, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / fname)
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Figure 3 – Ti state sweep ─────────────────────────────────────────────────


def plot_state_sweep(solver: _SolverWrapper, N_forecast: int, out_dir: Path, device: str) -> None:
    lb_K, ub_K = _comfort_bounds(_QH_DEFAULT)
    Ti_arr = np.linspace(lb_K - 8.0, ub_K + 8.0, 40)
    obs = _make_obs(Ti_arr, N_forecast=N_forecast, device=device)
    param = _default_param(solver, len(Ti_arr), device)

    ctx, u0_W, val = _forward(solver, obs, param)
    u0_kW = u0_W.flatten() / 1000.0
    val_flat = val.flatten()

    du0_dx0 = _sens(solver, ctx, "du0_dx0")  # (B, nu, 4)
    dv_dx0 = _sens(solver, ctx, "dvalue_dx0")  # (B,  1, 4)

    Ti_C = Ti_arr - 273.15
    lb_C, ub_C = lb_K - 273.15, ub_K - 273.15

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    fig.suptitle("Sensitivity sweep over initial indoor temperature  Ti₀")

    def _shade(ax):
        ax.axvspan(lb_C, ub_C, color="lime", alpha=0.15, label="comfort zone")
        ax.axvline(lb_C, color="green", lw=0.9, ls="--")
        ax.axvline(ub_C, color="green", lw=0.9, ls="--")
        ax.axhline(0, color="grey", lw=0.5)

    ax = axes[0, 0]
    ax.plot(Ti_C, u0_kW, "b-o", ms=3)
    _shade(ax)
    ax.set_ylabel("u₀  [kW]")
    ax.set_title("First control action  u₀")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(Ti_C, val_flat, "r-o", ms=3)
    _shade(ax)
    ax.set_ylabel("J")
    ax.set_title("Optimal cost  J")

    ax = axes[1, 0]
    for k, name in enumerate(_X_NAMES):
        arr = du0_dx0[:, 0, k] if du0_dx0.ndim == 3 else du0_dx0[:, k]
        ax.plot(Ti_C, arr, "-o", ms=3, label=f"∂u₀/∂{name}")
    _shade(ax)
    ax.set_ylabel("∂u₀/∂x₀")
    ax.set_xlabel("Ti₀  [°C]")
    ax.set_title("Policy gradient w.r.t. initial state")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for k, name in enumerate(_X_NAMES):
        arr = dv_dx0[:, 0, k] if dv_dx0.ndim == 3 else dv_dx0[:, k]
        ax.plot(Ti_C, arr, "-o", ms=3, label=f"∂J/∂{name}")
    _shade(ax)
    ax.set_ylabel("∂J/∂x₀")
    ax.set_xlabel("Ti₀  [°C]")
    ax.set_title("Value gradient w.r.t. initial state")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "state_sweep.pdf")
    plt.close(fig)
    print("  Saved state_sweep.pdf")


# ── Figure 4 – per-parameter sweep ────────────────────────────────────────────


def plot_param_sweep(
    solver: _SolverWrapper, N_forecast: int, out_dir: Path, device: str, n_pts: int = 20
) -> None:
    keys, defaults, lbs, ubs = _param_info(solver)
    lb_K, ub_K = _comfort_bounds(_QH_DEFAULT)
    Ti_mid = (lb_K + ub_K) / 2.0
    n_par = len(keys)

    fig, axes = plt.subplots(n_par, 3, figsize=(13, 3.2 * n_par), squeeze=False)
    fig.suptitle("Per-parameter sweep:  u₀, J, and ∂u₀/∂pᵢ  (analytic vs FD)", y=1.002)

    for i, (k, lb, ub, default) in enumerate(zip(keys, lbs, ubs, defaults)):
        ax0, ax1, ax2 = axes[i]
        if not (np.isfinite(lb) and np.isfinite(ub)):
            for ax in (ax0, ax1, ax2):
                ax.text(
                    0.5,
                    0.5,
                    f"{k}\n(unbounded – skipped)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
            continue

        param_vals = np.linspace(lb, ub, n_pts)
        obs = _make_obs(np.full(n_pts, Ti_mid), N_forecast=N_forecast, device=device)
        param = _default_param(solver, n_pts, device)
        param[:, i] = torch.tensor(param_vals, dtype=torch.float64, device=device)

        ctx, u0_W, val = _forward(solver, obs, param)
        u0_kW = u0_W.flatten() / 1000.0
        val_flat = val.flatten()

        du0_dp_full = _sens(solver, ctx, "du0_dp")
        du0_dpi = du0_dp_full[:, 0, i] if du0_dp_full.ndim == 3 else du0_dp_full[:, i]
        fd_dpi = np.gradient(u0_W.flatten(), param_vals)

        ax0.plot(param_vals, u0_kW, "b-o", ms=3)
        ax0.axvline(default, color="k", lw=0.8, ls=":", label="default")
        ax0.set_ylabel("u₀  [kW]")
        ax0.set_title(k, fontsize=10, fontweight="bold")
        ax0.legend(fontsize=7)

        ax1.plot(param_vals, val_flat, "r-o", ms=3)
        ax1.axvline(default, color="k", lw=0.8, ls=":")
        ax1.set_ylabel("J")

        ax2.plot(param_vals, du0_dpi, "g-o", ms=3, label="analytic ∂u₀/∂p")
        ax2.plot(param_vals, fd_dpi, "k--x", ms=4, lw=1, label="FD ∂u₀/∂p")
        ax2.axvline(default, color="k", lw=0.8, ls=":")
        ax2.axhline(0, color="grey", lw=0.4)
        ax2.set_ylabel("∂u₀/∂p  [W/unit]")
        ax2.legend(fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("parameter value")

    fig.tight_layout()
    fig.savefig(out_dir / "param_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved param_sweep.pdf")


# ── Figure 5 – FD vs analytic check at default point ─────────────────────────


def plot_fd_check(
    solver: _SolverWrapper, N_forecast: int, out_dir: Path, device: str, eps_rel: float = 1e-4
) -> None:
    keys, defaults, *_ = _param_info(solver)
    n_par = len(keys)
    lb_K, ub_K = _comfort_bounds(_QH_DEFAULT)
    Ti_mid = (lb_K + ub_K) / 2.0
    obs = _make_obs(np.array([Ti_mid]), N_forecast=N_forecast, device=device)

    p0 = _default_param(solver, 1, device)
    ctx0, u0_0, val_0 = _forward(solver, obs, p0)
    du0_an_full = _sens(solver, ctx0, "du0_dp")
    dv_an_full = _sens(solver, ctx0, "dvalue_dp")

    du0_an = du0_an_full[0, 0, :] if du0_an_full.ndim == 3 else du0_an_full[0, :]
    dv_an = dv_an_full[0, 0, :] if dv_an_full.ndim == 3 else dv_an_full[0, :]

    du0_fd = np.zeros(n_par)
    dv_fd = np.zeros(n_par)
    p0_np = defaults.copy()

    for j in range(n_par):
        eps = eps_rel * (abs(p0_np[j]) + 1e-8)
        pp = p0_np.copy()
        pp[j] += eps
        pm_ = p0_np.copy()
        pm_[j] -= eps
        _, u_p, v_p = _forward(
            solver, obs, torch.tensor(pp[None, :], dtype=torch.float64, device=device)
        )
        _, u_m, v_m = _forward(
            solver, obs, torch.tensor(pm_[None, :], dtype=torch.float64, device=device)
        )
        du0_fd[j] = (float(u_p.flat[0]) - float(u_m.flat[0])) / (2 * eps)
        dv_fd[j] = (float(v_p.flat[0]) - float(v_m.flat[0])) / (2 * eps)

    def _rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-12)

    hdr = f"  {'param':<15}  {'∂u₀/∂p analytic':>18}  {'FD':>14}  {'rel-err':>9}"
    print("\n" + hdr)
    print("  " + "─" * (len(hdr) - 2))
    for j, k in enumerate(keys):
        print(
            f"  {k:<15}  {du0_an[j]:>18.4e}  {du0_fd[j]:>14.4e}  {_rel(du0_an[j], du0_fd[j]):>9.2e}"
            f"  |  {dv_an[j]:>14.4e}  {dv_fd[j]:>14.4e}  {_rel(dv_an[j], dv_fd[j]):>9.2e}"
        )

    x, w = np.arange(n_par), 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f"Gradient check at default params, Ti₀ = {Ti_mid - 273.15:.1f} °C"
        f"  (FD step {eps_rel:.0e}·|p|)",
        fontsize=11,
    )
    for ax, an, fd, ylabel, title in [
        (ax1, du0_an, du0_fd, "∂u₀/∂p  [W / unit]", "du₀/dp"),
        (ax2, dv_an, dv_fd, "∂J/∂p", "dJ/dp"),
    ]:
        ax.bar(x - w / 2, an, w, label="analytic", color="steelblue", alpha=0.9)
        ax.bar(x + w / 2, fd, w, label="FD central", color="coral", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.5)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "fd_check.pdf")
    plt.close(fig)
    print("  Saved fd_check.pdf")


# ── solver factory ────────────────────────────────────────────────────────────


def _build_solver(
    variant: str,
    N: int,
    nom: HydronicParameters,
    reuse_code_dir: Path | None,
) -> _SolverWrapper:
    interface, granularity = "reference_dynamics", "global"

    if variant == "resistance":
        params = make_default_hvac_params(interface, granularity, N, nom)
        pm = AcadosParameterManager(params, N)
        ocp = export_parametric_ocp(pm, N, name="hvac_r")
    elif variant == "conductance":
        params = make_default_hvac_params_conductance(interface, granularity, N, nom)
        pm = AcadosParameterManager(params, N)
        ocp = export_parametric_ocp_conductance(pm, N, name="hvac_c")
    elif variant == "normalized":
        params = make_default_hvac_params_normalized(interface, granularity, N, nom)
        pm = AcadosParameterManager(params, N)
        ocp = export_parametric_ocp_normalized(pm, N, name="hvac_n", hydronic_params=nom)
    elif variant == "normalized_conductance":
        params = make_default_hvac_params_normalized_conductance(interface, granularity, N, nom)
        pm = AcadosParameterManager(params, N)
        ocp = export_parametric_ocp_normalized_conductance(
            pm, N, name="hvac_nc", hydronic_params=nom
        )
    elif variant == "parameter_linear":
        params = make_default_hvac_params_parameter_linear(interface, granularity, N, nom)
        pm = AcadosParameterManager(params, N)
        ocp = export_parametric_ocp_parameter_linear(pm, N, name="hvac_pl")
    elif variant == "discrete_matrix":
        params = make_default_hvac_params_discrete_matrix(interface, granularity, N, nom)
        pm = AcadosParameterManager(params, N)
        ocp = export_parametric_ocp_discrete_matrix(pm, N, name="hvac_dm", hydronic_params=nom)
    else:
        raise ValueError(f"Unknown variant '{variant}'")

    code_dir = (reuse_code_dir / variant) if reuse_code_dir else None
    if code_dir:
        code_dir.mkdir(parents=True, exist_ok=True)

    diff_mpc = AcadosDiffMpcTorch(
        ocp,
        export_directory=code_dir,
        n_batch_init=64,
        num_threads_batch_solver=1,
        verbose=False,
    )
    return _SolverWrapper(diff_mpc, pm, N)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for all three HVAC OCP parameterisations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument(
        "--n-horizon", type=int, default=24, help="MPC horizon steps. Shorter = faster compilation."
    )
    parser.add_argument(
        "--n-pts", type=int, default=20, help="Points per parameter in the param sweep."
    )
    parser.add_argument(
        "--reuse-code-dir",
        type=Path,
        default=None,
        help="Reuse compiled acados solver from this directory.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "resistance",
            "conductance",
            "normalized",
            "normalized_conductance",
            "parameter_linear",
            "discrete_matrix",
        ],
        choices=[
            "resistance",
            "conductance",
            "normalized",
            "normalized_conductance",
            "parameter_linear",
            "discrete_matrix",
        ],
        help="Which solver variants to analyse.",
    )
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument("--skip-state-sweep", action="store_true")
    parser.add_argument("--skip-param-sweep", action="store_true")
    parser.add_argument("--skip-fd-check", action="store_true")
    args = parser.parse_args()

    N_forecast = args.n_horizon + 2
    nom = HydronicParameters()

    for variant in args.variants:
        print(f"\n{'═' * 60}")
        print(f"  Variant: {variant.upper()}")
        print(f"{'═' * 60}")

        print(f"  Building solver (N_horizon={args.n_horizon}) …")
        solver = _build_solver(variant, args.n_horizon, nom, args.reuse_code_dir)

        keys, defaults, lbs, ubs = _param_info(solver)
        print(f"\n  {'Parameter':<15}  {'default':>15}  {'lb':>15}  {'ub':>15}")
        print("  " + "─" * 63)
        for k, d, lb, ub in zip(keys, defaults, lbs, ubs):
            print(f"  {k:<15}  {d:>15.5g}  {lb:>15.5g}  {ub:>15.5g}")

        out_dir = args.output_dir / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_heatmap:
            print("\n  ── Scenario heatmaps ──")
            scenarios = _build_scenarios(solver, N_forecast, args.device)
            plot_heatmaps(solver, scenarios, out_dir, args.device)

        if not args.skip_state_sweep:
            print("\n  ── State sweep (Ti) ──")
            plot_state_sweep(solver, N_forecast, out_dir, args.device)

        if not args.skip_param_sweep:
            print("\n  ── Parameter sweep ──")
            plot_param_sweep(solver, N_forecast, out_dir, args.device, n_pts=args.n_pts)

        if not args.skip_fd_check:
            print("\n  ── FD gradient check ──")
            plot_fd_check(solver, N_forecast, out_dir, args.device)

        print(f"\n  Figures saved to: {out_dir}")

    print(f"\nDone. All figures under: {args.output_dir}")


if __name__ == "__main__":
    main()
