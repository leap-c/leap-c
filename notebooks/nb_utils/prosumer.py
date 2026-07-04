"""The prosumer OCP used by notebook 08, plus its plot helpers.

An R1C1-heated building (the model from ``heating.py``) with a heat pump of
constant COP, a lossless battery, a PV panel and a grid connection with
asymmetric prices: electricity is bought at a dynamic tariff and sold at the
(much lower) feed-in tariff. Battery power is eliminated as a control — it is
the CasADi expression

    P_bat = g_buy - g_sell + p_pv - q

so the OCP needs box constraints only (no general equality constraints).
"""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import torch
from acados_template import AcadosOcp
from matplotlib.figure import Figure

from leap_c.parameters import AcadosParameterManager

from .heating import C_THERMAL, R_THERMAL

COP = 3.0  # heat pump coefficient of performance [kW heat / kW electric]


def build_prosumer_ocp(
    N_horizon: int,
    dt: float = 0.25,
    e_max: float = 10.0,
    q_max: float = 4.0,
    g_max: float = 10.0,
    t_band: tuple[float, float] = (19.0, 23.0),
    c_wear: float = 0.005,
    eps_reg: float = 1e-3,
    name: str = "prosumer",
) -> tuple[AcadosOcp, AcadosParameterManager]:
    """Build the parametric prosumer OCP and its parameter manager.

    States are the room temperature T [degC] and the battery energy E [kWh];
    controls are the heat-pump electric power q, the grid purchase g_buy and
    the grid feed-in g_sell [kW] (all nonnegative).

    Args:
        N_horizon: Horizon length (number of shooting intervals).
        dt: Time step [h].
        e_max: Battery capacity [kWh].
        q_max: Upper bound on the heat-pump electric power [kW].
        g_max: Grid connection limit, both directions [kW].
        t_band: Comfort band (lower, upper) on the room temperature [degC],
            enforced softly via slacks.
        c_wear: Quadratic battery wear cost [EUR h/kW^2].
        eps_reg: Quadratic regularization on each control [EUR h/kW^2] — the
            source of a full-rank control Hessian (see notebook 08).
        name: acados model name. Use distinct names when building several
            instances in one session so their generated code does not collide.

    Always builds the OCP and the manager together, fresh: a manager is
    finalized by ``AcadosDiffMpcTorch`` (via ``assign_to_ocp``) and must not
    be reused for a second OCP.
    """
    manager = AcadosParameterManager(N_horizon=N_horizon)

    # Tariff forecast: differentiable, one value per stage — *the* parameter
    # this notebook is about.
    price_buy = manager.register_parameter(
        name="price_buy", default=np.array([0.25]), differentiable=True, splits="stagewise"
    )
    # Feed-in tariff: fixed by regulation, one differentiable value.
    price_sell = manager.register_parameter(
        name="price_sell", default=np.array([0.079]), differentiable=True
    )
    # Value of energy still stored at the end of the horizon [EUR/kWh].
    terminal_value = manager.register_parameter(
        name="terminal_value", default=np.array([0.12]), differentiable=True
    )
    # Forecasts: changeable per stage at runtime, but no gradients.
    outdoor_temp = manager.register_parameter(
        name="outdoor_temp", default=np.array([8.0]), differentiable=False
    )
    p_pv = manager.register_parameter(
        name="p_pv", default=np.array([0.0]), differentiable=False
    )

    ocp = AcadosOcp()
    ocp.model.name = name

    T = ca.SX.sym("T")  # room temperature [degC]
    E = ca.SX.sym("E")  # battery energy [kWh]
    q = ca.SX.sym("q")  # heat pump electric power [kW]
    g_buy = ca.SX.sym("g_buy")  # grid purchase [kW]
    g_sell = ca.SX.sym("g_sell")  # grid feed-in [kW]
    ocp.model.x = ca.vertcat(T, E)
    ocp.model.u = ca.vertcat(q, g_buy, g_sell)

    # Battery power balances the household bus — eliminated, not a control.
    P_bat = g_buy - g_sell + p_pv - q

    ocp.model.disc_dyn_expr = ca.vertcat(
        T + dt * ((outdoor_temp - T) / (R_THERMAL * C_THERMAL) + COP * q / C_THERMAL),
        E + dt * P_bat,
    )

    # Economic cost: cash flow, battery wear, and a small full-rank
    # regularization on the controls; terminal credit for stored energy.
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (
        price_buy * g_buy
        - price_sell * g_sell
        + c_wear * P_bat**2
        + eps_reg * (q**2 + g_buy**2 + g_sell**2)
    )
    ocp.model.cost_expr_ext_cost_e = -terminal_value * E

    # Initial state — a nominal value, overwritten on every solve.
    ocp.constraints.x0 = np.array([21.0, 5.0])

    # Comfort band on T (soft, slacked) and battery capacity on E (hard).
    ocp.constraints.idxbx = np.array([0, 1])
    ocp.constraints.lbx = np.array([t_band[0], 0.0])
    ocp.constraints.ubx = np.array([t_band[1], e_max])
    ocp.constraints.idxsbx = np.array([0])  # soften only the T row
    ocp.cost.Zl = ocp.cost.Zu = np.array([1e2])
    ocp.cost.zl = ocp.cost.zu = np.array([0.0])
    ocp.constraints.idxbx_e = np.array([0, 1])
    ocp.constraints.lbx_e = np.array([t_band[0], 0.0])
    ocp.constraints.ubx_e = np.array([t_band[1], e_max])
    ocp.constraints.idxsbx_e = np.array([0])
    ocp.cost.Zl_e = ocp.cost.Zu_e = np.array([1e2])
    ocp.cost.zl_e = ocp.cost.zu_e = np.array([0.0])

    # All three controls are nonnegative and individually box-bounded.
    ocp.constraints.idxbu = np.array([0, 1, 2])
    ocp.constraints.lbu = np.array([0.0, 0.0, 0.0])
    ocp.constraints.ubu = np.array([q_max, g_max, g_max])

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon * dt
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp, manager


def step_prosumer(
    x: np.ndarray, u: np.ndarray, outdoor_temp: float, p_pv: float, dt: float = 0.25
) -> np.ndarray:
    """The true prosumer update — identical to the model inside the OCP."""
    T, E = x
    q, g_buy, g_sell = u
    p_bat = g_buy - g_sell + p_pv - q
    return np.array(
        [
            T + dt * ((outdoor_temp - T) / (R_THERMAL * C_THERMAL) + COP * q / C_THERMAL),
            E + dt * p_bat,
        ]
    )


def gnet_price_jacobian(u: torch.Tensor, price_buy: torch.Tensor) -> np.ndarray:
    """Full Jacobian of the planned net grid exchange w.r.t. the stagewise buy price.

    Row ``k`` is the gradient of ``g_net,k = g_buy,k - g_sell,k`` w.r.t. the
    whole price profile — one backward pass per stage (reverse mode pays per
    output component, see notebook 07). Summing over the batch inside each
    pass is exact because batch elements are independent: element ``b``'s
    plan only depends on its own price row.

    Args:
        u: Control solution ``(B, N, 3)``, still attached to the autograd graph.
        price_buy: The ``(B, N+1, 1)`` leaf tensor the solve was given.

    Returns:
        The Jacobian, shape ``(B, N, N+1)``, as a numpy array.
    """
    rows = []
    for k in range(u.shape[1]):
        (g,) = torch.autograd.grad(
            (u[:, k, 1] - u[:, k, 2]).sum(), price_buy, retain_graph=True
        )
        rows.append(g[:, :, 0].numpy())
    return np.stack(rows, axis=1)


def plot_prosumer_day(
    t: np.ndarray,
    outdoor_temp: np.ndarray,
    price_buy: np.ndarray,
    p_pv: np.ndarray,
    price_sell: float,
) -> Figure:
    """Overview of the prosumer's day: tariff, PV generation, weather."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)

    axes[0].step(t, price_buy, where="post", color="tab:purple", label="buy (dynamic tariff)")
    axes[0].axhline(price_sell, ls="--", color="tab:green", label=f"sell (feed-in, {price_sell:.3f})")
    axes[0].set_ylabel("Price [EUR/kWh]")
    axes[0].set_ylim(0.0, 0.55)
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].fill_between(t, p_pv, color="tab:orange", alpha=0.35)
    axes[1].plot(t, p_pv, color="tab:orange")
    axes[1].set_ylabel("PV generation [kW]")

    axes[2].plot(t, outdoor_temp, color="tab:blue")
    axes[2].set_ylabel("Outdoor temp [degC]")
    axes[2].set_xlabel("Time since midnight [h]")
    axes[2].set_xticks(np.arange(0, t[-1] + 1, 3))

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("A prosumer's day")
    fig.tight_layout()
    return fig


def plot_prosumer_plan(
    t: np.ndarray,
    price_buy: np.ndarray,
    p_pv: np.ndarray,
    x_plan: np.ndarray,
    u_plan: np.ndarray,
    e_max: float,
    t_band: tuple[float, float],
    title: str,
) -> Figure:
    """The planned day: tariff, signed grid exchange, battery energy, room temperature.

    Args:
        t: Stage times [h], shape ``(N+1,)``.
        price_buy: Buy-price profile, shape ``(N+1,)``.
        p_pv: PV forecast, shape ``(N+1,)``.
        x_plan: State plan ``(N+1, 2)`` — columns T, E.
        u_plan: Control plan ``(N, 3)`` — columns q, g_buy, g_sell.
        e_max: Battery capacity [kWh].
        t_band: Comfort band (lower, upper) [degC].
        title: Figure title.
    """
    g_net = u_plan[:, 1] - u_plan[:, 2]
    dt = t[1] - t[0]

    fig, axes = plt.subplots(4, 1, figsize=(9, 8.5), sharex=True)

    axes[0].step(t, price_buy, where="post", color="tab:purple")
    axes[0].set_ylabel("Buy price\n[EUR/kWh]")
    axes[0].set_ylim(0.0, 0.55)

    colors = np.where(g_net >= 0, "tab:purple", "tab:green")
    axes[1].bar(t[:-1], g_net, width=0.9 * dt, align="edge", color=colors)
    axes[1].plot(t, p_pv, color="tab:orange", lw=1.0, label="PV forecast")
    axes[1].axhline(0.0, lw=0.8, color="gray")
    axes[1].set_ylabel("Net grid exchange\n[kW]  (buy +, sell -)")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].plot(t, x_plan[:, 1], "-o", markersize=3, color="tab:blue")
    axes[2].axhline(e_max, ls="--", lw=0.8, color="gray")
    axes[2].axhline(0.0, ls="--", lw=0.8, color="gray")
    axes[2].set_ylabel("Battery energy\n[kWh]")
    axes[2].set_ylim(-0.5, e_max + 0.5)

    axes[3].plot(t, x_plan[:, 0], color="tab:red")
    axes[3].axhspan(t_band[0], t_band[1], color="tab:red", alpha=0.08)
    axes[3].axhline(t_band[0], ls="--", lw=0.8, color="gray")
    axes[3].axhline(t_band[1], ls="--", lw=0.8, color="gray")
    axes[3].set_ylabel("Room temp\n[degC]")
    axes[3].set_ylim(t_band[0] - 1.0, t_band[1] + 1.0)
    axes[3].set_xlabel("Time since midnight [h]")
    axes[3].set_xticks(np.arange(0, t[-1] + 1, 3))

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_gnet_price_jacobian_col(
    t: np.ndarray, S_b: np.ndarray, j: int, price_buy: np.ndarray
) -> Figure:
    """One column of the plan-vs-price Jacobian: the response to a bump of price_j.

    Args:
        t: Stage times [h], shape ``(N+1,)``.
        S_b: Jacobian of one batch element, shape ``(N, N+1)``.
        j: The perturbed price stage.
        price_buy: Buy-price profile, shape ``(N+1,)``, for context.
    """
    dt = t[1] - t[0]
    col = S_b[:, j]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True,
                             height_ratios=[1.0, 2.0])

    axes[0].step(t, price_buy, where="post", color="tab:purple")
    axes[0].axvline(t[j], color="black", lw=1.2)
    axes[0].set_ylabel("Buy price\n[EUR/kWh]")
    axes[0].set_ylim(0.0, 0.55)
    axes[0].set_title(f"Perturbed stage: {t[j]:.2f} h")

    colors = np.where(col >= 0, "tab:purple", "tab:green")
    axes[1].bar(t[:-1], col, width=0.9 * dt, align="edge", color=colors)
    axes[1].axvline(t[j], color="black", lw=1.2)
    axes[1].axhline(0.0, lw=0.8, color="gray")
    axes[1].set_ylabel("∂g_net,k / ∂price_j\n[kW per EUR/kWh]")
    axes[1].set_xlabel("Time since midnight [h]")
    axes[1].set_xticks(np.arange(0, t[-1] + 1, 6))

    # The own-price response (k = j) dwarfs the substitution lobes around
    # it; scale to the off-diagonal structure and annotate the clipped bar.
    off_diag = np.delete(col, j) if j < col.shape[0] else col
    lim = 1.3 * max(np.abs(off_diag).max(), 1e-3)
    axes[1].set_ylim(-lim, lim)
    if j < col.shape[0] and np.abs(col[j]) > lim:
        axes[1].annotate(
            f"{col[j]:.0f} (clipped)",
            xy=(t[j], -0.93 * lim if col[j] < 0 else 0.93 * lim),
            fontsize=8, ha="left", va="center", xytext=(6, 0),
            textcoords="offset points",
        )

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_gnet_price_jacobian_heatmap(t: np.ndarray, S_b: np.ndarray) -> Figure:
    """The full plan-vs-price Jacobian of one batch element as a heatmap.

    Rows are plan stages k, columns are price stages j; the color is
    ∂g_net,k/∂price_j, symmetric diverging around zero. The color scale is
    set by the *off-diagonal* structure (the substitution pattern); the
    much stronger own-price diagonal saturates.
    """
    mask = np.eye(S_b.shape[0], S_b.shape[1], dtype=bool)
    vmax = max(np.abs(S_b[~mask]).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(
        S_b,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=(t[0], t[-1], t[0], t[-2]),
    )
    ax.set_xlabel("Price stage j [h]")
    ax.set_ylabel("Plan stage k [h]")
    ax.set_title("∂g_net,k / ∂price_j  [kW per EUR/kWh]  (diagonal saturated)")
    fig.colorbar(im, ax=ax, shrink=0.85, extend="both")
    fig.tight_layout()
    return fig
