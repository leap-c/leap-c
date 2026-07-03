"""Part 6 — battery arbitrage: an economic MPC.

The heating notebooks (04, 05) mixed a comfort *tracking* term with an energy
price. Here the cost is money and nothing else: a battery buys (charges) when
electricity is cheap and sells (discharges) when it is expensive. This is the
smallest possible showcase of an *economic* cost — one state (stored energy),
one input (charge power), a stagewise price — and it surfaces two questions
every economic MPC must answer: where does the Hessian come from when the
cost is linear, and what is energy still in the battery worth at the end of
the horizon?
"""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 — battery arbitrage: an economic MPC

    In notebooks 04 and 05 the OCP *tracked* a comfort setpoint and the
    price only tilted the trade-off. In an **economic MPC** there is no
    reference at all — the stage cost is the cash flow itself, and the
    optimal behavior (buy low, sell high) emerges from the price forecast
    alone.

    A battery makes this minimal: state $E$ = stored energy [kWh], input
    $u$ = charge power [kW] (positive buys, negative sells), and a
    stagewise electricity price. Along the way we meet two questions every
    economic MPC must answer:

    - a purely **linear** cost has zero curvature — where does the solver's
      Hessian come from?
    - energy left in the battery at the end of the horizon is not
      worthless — what is it worth?

    The finale differentiates the optimal cost with respect to every future
    price: unlike the heater in 05 (which only buys), the battery's price
    sensitivities come out **signed**.
    """)
    return


@app.cell
def _():
    import casadi as ca
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from acados_template import AcadosOcp

    from nb_utils.data import make_day_profiles
    from nb_utils.params import p_global_slice

    from leap_c.ocp.acados.parameters import AcadosParameterManager
    from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

    torch.set_default_dtype(torch.float64)
    return (
        AcadosDiffMpcTorch,
        AcadosOcp,
        AcadosParameterManager,
        ca,
        make_day_profiles,
        np,
        p_global_slice,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The OCP

    A lossless battery is a pure integrator,

    $$E_{k+1} = E_k + \Delta t \, u_k, \qquad
    E \in [0, E_{\max}], \quad u \in [-p_{\max}, p_{\max}],$$

    and the objective is the money spent (negative = earned):

    $$\sum_{k=0}^{N-1} \big(\text{price}_k \, u_k
    + c_\text{wear}\, u_k^2\big)\,\Delta t \;-\; \lambda\, E_N.$$

    Two terms deserve a defense:

    - **The wear term** $c_\text{wear} u^2$ is *mandatory*, not cosmetic. A
      purely linear cost has zero Hessian; the exact-Hessian SQP would face
      a singular QP, and the sensitivity solver leap-c builds for the
      backward pass strips *all* regularization (it must — regularization
      would falsify the KKT sensitivities). The curvature has to come from
      the cost itself, and a small quadratic wear cost is the physically
      honest way to provide it. We pick $c_\text{wear} = 0.005$
      EUR·h/kW²: at full power the marginal wear is
      $2 \cdot 0.005 \cdot 4 = 0.04$ EUR/kWh, far below the 0.20 EUR/kWh
      peak/off-peak spread — the economics stay in charge.
    - **The terminal value** $-\lambda E_N$ prices the energy still stored
      at the end of the horizon. Without it, leftover energy is worthless
      and the optimal plan dumps every last kWh before stage $N$ — an
      artifact of the finite horizon, not a property of good battery
      operation. We set $\lambda = 0.15$ EUR/kWh, exactly the off-peak
      price: stored energy is worth what it costs to replace.

    Three differentiable parameters enter the problem:

    | name | default | splits | role |
    |---|---|---|---|
    | `price` | 0.15 | `"stagewise"` | electricity price forecast [EUR/kWh] |
    | `terminal_value` ($\lambda$) | 0.15 | global | value of stored energy at stage $N$ |
    | `c_wear` | 0.005 | global | quadratic wear cost [EUR·h/kW²] |

    (No non-differentiable parameter this time — 04 and 05 covered that
    interface.)
    """)
    return


@app.cell
def _(AcadosOcp, AcadosParameterManager, ca, np):
    def build_battery_ocp(N_horizon, dt=0.25, e_max=8.0, p_max=4.0, name="battery"):
        # NOTE: manager and OCP are built together, fresh — a manager is
        # finalized by AcadosDiffMpcTorch (via assign_to_ocp) and must not be
        # reused for a second OCP.
        manager = AcadosParameterManager(N_horizon=N_horizon)

        # Price forecast: differentiable, one value per stage.
        price = manager.register_parameter(
            name="price", default=np.array([0.15]), differentiable=True, splits="stagewise"
        )
        # Value of energy still stored at the end of the horizon [EUR/kWh].
        terminal_value = manager.register_parameter(
            name="terminal_value", default=np.array([0.15]), differentiable=True
        )
        # Quadratic wear cost — the source of the Hessian (see above).
        c_wear = manager.register_parameter(
            name="c_wear", default=np.array([0.005]), differentiable=True
        )

        ocp = AcadosOcp()
        ocp.model.name = name

        E = ca.SX.sym("E")  # stored energy [kWh]
        u = ca.SX.sym("u")  # charge power [kW], u < 0 discharges/sells
        ocp.model.x = E
        ocp.model.u = u

        # A lossless battery is a pure integrator.
        ocp.model.disc_dyn_expr = E + dt * u

        # Economic cost: cash flow plus wear; terminal credit for stored
        # energy. acados scales the stage cost by dt, but not the terminal one.
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = price * u + c_wear * u**2
        ocp.model.cost_expr_ext_cost_e = -terminal_value * E

        # Initial state — a nominal value, overwritten on every solve.
        ocp.constraints.x0 = np.array([4.0])

        # Hard state box (a 1-state integrator can always hold its state with
        # u = 0, so hard bounds cannot cause infeasibility). idxbx covers the
        # intermediate stages only — stage N needs its own idxbx_e, otherwise
        # the last step may overshoot e_max.
        ocp.constraints.idxbx = np.array([0])
        ocp.constraints.lbx = np.array([0.0])
        ocp.constraints.ubx = np.array([e_max])
        ocp.constraints.idxbx_e = np.array([0])
        ocp.constraints.lbx_e = np.array([0.0])
        ocp.constraints.ubx_e = np.array([e_max])

        # Charge and discharge power limits.
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([-p_max])
        ocp.constraints.ubu = np.array([p_max])

        ocp.solver_options.N_horizon = N_horizon
        ocp.solver_options.tf = N_horizon * dt
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

        return ocp, manager

    def step_battery(E, u, dt=0.25):
        """The true battery update — identical to the model inside the OCP."""
        return E + dt * u

    return build_battery_ocp, step_battery


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use the same grid as notebook 05 — 15-minute steps, an 8-hour
    horizon, and one forecast window per quarter-hour of the day — so all
    the window bookkeeping carries over unchanged.
    """)
    return


@app.cell
def _(AcadosDiffMpcTorch, build_battery_ocp, torch):
    N_BAT = 32  # 32 stages of 15 min = an 8 h look-ahead
    DT_BAT = 0.25  # time step [h]
    N_STARTS = 96  # one forecast window per quarter-hour of the day
    E_MAX = 8.0  # battery capacity [kWh]
    P_MAX = 4.0  # charge/discharge power limit [kW]
    E0 = 4.0  # initial stored energy [kWh] — half full

    mpc_bat = AcadosDiffMpcTorch(
        *build_battery_ocp(N_BAT, DT_BAT, e_max=E_MAX, p_max=P_MAX),
        dtype=torch.float64,
        n_batch_init=N_STARTS,
        verbose=False,
    )
    return DT_BAT, E0, E_MAX, N_BAT, N_STARTS, P_MAX, mpc_bat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The price day

    We reuse the synthetic day from `nb_utils.data.make_day_profiles` and
    ignore its weather output: 0.15 EUR/kWh off-peak, stepping up to 0.35
    during the morning (7–9 h) and evening (17–20 h) peaks. The 0.20 spread
    is what the battery gets paid for shifting energy; at 0.04 EUR/kWh of
    worst-case marginal wear, arbitrage is clearly profitable.
    """)
    return


@app.cell
def _(DT_BAT, make_day_profiles, np, plt):
    t_day, _, price_day = make_day_profiles(n_steps=144, dt_hours=DT_BAT)

    price_fig, price_ax = plt.subplots(figsize=(9, 2.8))
    price_ax.step(t_day, price_day, where="post", color="tab:purple")
    price_ax.set_xlabel("Time since midnight [h]")
    price_ax.set_ylabel("Price [EUR/kWh]")
    price_ax.set_xticks(np.arange(0, 37, 6))
    price_ax.set_ylim(0.0, 0.5)
    price_ax.grid(True, alpha=0.3)
    for _lo, _hi in [(7.0, 9.0), (17.0, 20.0), (31.0, 33.0)]:
        price_ax.axvspan(_lo, _hi, color="tab:purple", alpha=0.12)
    price_fig.suptitle("Synthetic electricity price (36 h)")
    price_fig.tight_layout()
    price_fig
    return price_day, t_day


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One solve, read the plan

    We solve a single instance at 05:00 — the morning price peak sits in
    the middle of the 8-hour window — starting from a half-full battery.
    """)
    return


@app.cell
def _(DT_BAT, E0, E_MAX, N_BAT, P_MAX, mo, mpc_bat, np, plt, price_day, t_day, torch):
    _s = 20  # 05:00
    _t_win = t_day[_s : _s + N_BAT + 1]
    _price_win = price_day[_s : _s + N_BAT + 1]

    _ctx, _, _x, _u, _value = mpc_bat(
        x0=torch.tensor([[E0]]),
        params={"price": torch.tensor(_price_win.reshape(1, -1, 1))},
    )
    _E_plan = _x.detach().numpy()[0, :, 0]
    _u_plan = _u.detach().numpy()[0, :, 0]

    plan_fig, plan_axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    plan_axes[0].step(_t_win, _price_win, where="post", color="tab:purple")
    plan_axes[0].set_ylabel("Price [EUR/kWh]")
    plan_axes[0].set_ylim(0.0, 0.5)
    plan_axes[1].plot(_t_win, _E_plan, "-o", markersize=3, color="tab:blue")
    plan_axes[1].axhline(E_MAX, ls="--", lw=0.8, color="gray")
    plan_axes[1].axhline(0.0, ls="--", lw=0.8, color="gray")
    plan_axes[1].set_ylabel("Stored energy [kWh]")
    plan_axes[1].set_ylim(-0.5, E_MAX + 0.5)
    plan_axes[2].step(_t_win[:-1], _u_plan, where="post", color="tab:orange")
    plan_axes[2].axhline(0.0, lw=0.8, color="gray")
    plan_axes[2].axhline(P_MAX, ls="--", lw=0.8, color="gray")
    plan_axes[2].axhline(-P_MAX, ls="--", lw=0.8, color="gray")
    plan_axes[2].set_ylabel("Charge power [kW]")
    plan_axes[2].set_xlabel("Time since midnight [h]")
    for _ax in plan_axes:
        _ax.grid(True, alpha=0.3)
        _ax.axvspan(7.0, 9.0, color="tab:purple", alpha=0.12)
    plan_fig.suptitle("Open-loop plan from 05:00")
    plan_fig.tight_layout()

    mo.vstack([
        plan_fig,
        mo.md(
            f"Solver status `{int(np.asarray(_ctx.status).item())}` (0 = success), "
            f"optimal cost `{float(_value.detach()):.3f}` EUR — negative, a net earning. "
            "The plan fills the battery on the flat pre-peak price (the wear "
            "term spreads the buying evenly — the economics are indifferent, "
            "the wear is not), sells at the full 4 kW through the 7–9 h peak, "
            "and then *stays empty*: with the terminal value equal to the "
            "off-peak price, re-buying for the terminal credit is pure wear "
            "loss."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One batched solve, 96 forecast windows

    As in notebook 05, batch element $i$ receives the price window starting
    at quarter-hour $i$ (always from a half-full battery), so a whole day
    of open-loop plans is precomputed in one call. The price windows and
    the terminal value are passed as leaf tensors with
    `requires_grad=True` — the same solve powers the gradient section at
    the end. Drag the slider; nothing is re-solved.
    """)
    return


@app.cell
def _(E0, N_BAT, N_STARTS, mpc_bat, np, price_day, torch):
    price_windows = torch.tensor(
        np.stack([price_day[s : s + N_BAT + 1] for s in range(N_STARTS)])[..., None],
        requires_grad=True,
    )  # (N_STARTS, N+1, 1)
    lam_batch = torch.full((N_STARTS, 1), 0.15, requires_grad=True)

    ctx_bat, _, x_bat, u_bat, value_bat = mpc_bat(
        x0=torch.full((N_STARTS, 1), E0),
        params={"price": price_windows, "terminal_value": lam_batch},
    )

    x_plans = x_bat.detach().numpy()  # (N_STARTS, N+1, 1)
    u_plans = u_bat.detach().numpy()  # (N_STARTS, N, 1)
    return ctx_bat, lam_batch, price_windows, u_plans, value_bat, x_plans


@app.cell
def _(N_STARTS, mo):
    start_slider = mo.ui.slider(
        start=0,
        stop=N_STARTS - 1,
        step=1,
        value=20,  # 05:00 — the morning peak sits mid-horizon
        label="forecast window start [quarter-hours since midnight]",
        show_value=True,
    )
    return (start_slider,)


@app.cell
def _(
    E_MAX,
    N_BAT,
    P_MAX,
    mo,
    np,
    plt,
    price_day,
    start_slider,
    t_day,
    u_plans,
    x_plans,
):
    # No solve here — the slider picks one precomputed forecast window.
    _s = start_slider.value
    _t_win = t_day[_s : _s + N_BAT + 1]

    exp_fig, exp_axes = plt.subplots(2, 1, figsize=(9, 6))

    # Top: the price day with the active horizon window shaded.
    exp_axes[0].step(t_day[:129], price_day[:129], where="post", color="tab:purple")
    exp_axes[0].set_ylabel("Price [EUR/kWh]")
    exp_axes[0].set_ylim(0.0, 0.5)
    exp_axes[0].axvspan(_t_win[0], _t_win[-1], color="gold", alpha=0.25, label="horizon window")
    exp_axes[0].set_xticks(np.arange(0, 33, 4))
    exp_axes[0].grid(True, alpha=0.3)
    exp_axes[0].legend(loc="upper left", fontsize=8)
    exp_axes[0].set_title(f"Forecast window starting at {_t_win[0]:.2f} h")

    # Bottom: the plan inside the window.
    exp_axes[1].plot(_t_win, x_plans[_s, :, 0], "-o", markersize=3, color="tab:blue",
                     label="planned stored energy")
    exp_axes[1].axhline(E_MAX, ls="--", lw=0.8, color="gray")
    exp_axes[1].axhline(0.0, ls="--", lw=0.8, color="gray")
    exp_axes[1].set_ylabel("Stored energy [kWh]", color="tab:blue")
    exp_axes[1].set_ylim(-0.5, E_MAX + 0.5)
    exp_axes[1].set_xlabel("Time since midnight [h]")
    exp_axes[1].grid(True, alpha=0.3)
    exp_axes[1].legend(loc="upper left", fontsize=8)
    _ax_u = exp_axes[1].twinx()
    _ax_u.step(_t_win[:-1], u_plans[_s, :, 0], where="post", color="tab:orange")
    _ax_u.axhline(0.0, lw=0.5, color="tab:orange", alpha=0.4)
    _ax_u.set_ylabel("Charge power [kW]", color="tab:orange")
    _ax_u.set_ylim(-1.2 * P_MAX, 1.2 * P_MAX)

    exp_fig.tight_layout()
    mo.vstack([start_slider, exp_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to look for while sliding:

    - As a price peak enters the window from the right, the plan starts
      **pre-charging** — gently, because the wear term spreads the buying
      over the cheap hours.
    - Around $s \approx 40$ (10:00) the window contains **no peak at all**:
      the plan simply holds the stored energy. The horizon cannot see money
      it cannot reach, and with $\lambda$ equal to the off-peak price there
      is no incentive to move.
    - Late-evening windows already see the *next morning's* peak and start
      buying overnight.

    ## Closing the loop

    A receding-horizon controller repeats this every quarter-hour: solve
    with the current window, apply the first charge decision, step the
    (here: perfectly modelled) battery, shift the window. This runs 96
    sequential solves once at cell execution — not on interaction.
    """)
    return


@app.cell
def _(DT_BAT, E0, N_BAT, N_STARTS, mpc_bat, np, price_day, step_battery, torch):
    E_sim = np.empty(N_STARTS + 1)
    u_sim = np.empty(N_STARTS)
    E_sim[0] = E0

    for _k in range(N_STARTS):
        _price_w = torch.tensor(price_day[_k : _k + N_BAT + 1].reshape(1, -1, 1))
        _, _u0, _, _, _ = mpc_bat(
            x0=torch.tensor([[E_sim[_k]]]),
            params={"price": _price_w},
        )
        u_sim[_k] = float(_u0.detach())
        E_sim[_k + 1] = step_battery(E_sim[_k], u_sim[_k], dt=DT_BAT)

    # Cumulative cash flow: buying (u > 0) costs money, selling earns it.
    cash = -np.cumsum(price_day[:N_STARTS] * u_sim * DT_BAT)
    # Realized profit values the change in stored energy at lambda.
    profit_total = cash[-1] + 0.15 * (E_sim[-1] - E_sim[0])
    return E_sim, cash, profit_total, u_sim


@app.cell
def _(DT_BAT, E_MAX, E_sim, N_STARTS, cash, np, plt, price_day, profit_total, t_day, u_sim):
    _t_state = DT_BAT * np.arange(N_STARTS + 1)
    _t_ctrl = DT_BAT * np.arange(N_STARTS)

    cl_fig, cl_axes = plt.subplots(4, 1, figsize=(9, 8.5), sharex=True)
    cl_axes[0].step(t_day[:N_STARTS], price_day[:N_STARTS], where="post", color="tab:purple")
    cl_axes[0].set_ylabel("Price [EUR/kWh]")
    cl_axes[1].plot(_t_state, E_sim, color="tab:blue")
    cl_axes[1].axhline(E_MAX, ls="--", lw=0.8, color="gray")
    cl_axes[1].axhline(0.0, ls="--", lw=0.8, color="gray")
    cl_axes[1].set_ylabel("Stored energy [kWh]")
    cl_axes[2].step(_t_ctrl, u_sim, where="post", color="tab:orange")
    cl_axes[2].axhline(0.0, lw=0.8, color="gray")
    cl_axes[2].set_ylabel("Charge power [kW]")
    cl_axes[3].plot(_t_ctrl, cash, color="tab:green")
    cl_axes[3].axhline(0.0, lw=0.8, color="gray")
    cl_axes[3].set_ylabel("Cumulative cash [EUR]")
    cl_axes[3].set_xlabel("Time since midnight [h]")
    for _ax_cl in cl_axes:
        _ax_cl.grid(True, alpha=0.3)
        for _lo, _hi in [(7.0, 9.0), (17.0, 20.0)]:
            _ax_cl.axvspan(_lo, _hi, color="tab:purple", alpha=0.08)
    cl_fig.suptitle(
        f"Receding-horizon day — realized profit {profit_total:.2f} EUR "
        "(incl. stored energy at λ)"
    )
    cl_fig.tight_layout()
    cl_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cash trace dips while the battery buys cheap energy and jumps during
    the purple price peaks when it sells — the day ends in profit. Note the
    overnight buying at the very end: the horizon already sees the next
    morning's peak.

    ## Gradients: the value function in EUR

    The optimal cost $V$ is money, so its sensitivities have currency
    units too. Because the cost is linear in the price, the envelope
    theorem gives

    $$\frac{\partial V}{\partial \text{price}_k} = u_k \,\Delta t
    \qquad \text{[kWh]},$$

    the energy *traded* at stage $k$ — **positive** where the plan buys (a
    dearer price hurts) and **negative** where it sells (a dearer peak
    helps). Contrast notebook 05, where the heater could only buy and every
    bar was positive. Likewise

    $$\frac{\partial V}{\partial \lambda} = -E_N,$$

    one extra credit per kWh left in the battery. Both come out of the
    exact KKT sensitivity `dvalue_dp_global`, read off the batched context
    from the window explorer (no new solve) and cross-checked against
    autograd.
    """)
    return


@app.cell
def _(
    DT_BAT,
    N_BAT,
    ctx_bat,
    lam_batch,
    mo,
    mpc_bat,
    np,
    p_global_slice,
    plt,
    price_windows,
    t_day,
    u_plans,
    value_bat,
    x_plans,
):
    _s = 20  # the 05:00 window again
    _t_win = t_day[_s : _s + N_BAT + 1]

    # Exact KKT sensitivities for the whole batch: (B, 1, P). The parameter
    # columns inside flat p_global are located from the registration order.
    _dV_dp = mpc_bat.diff_mpc_fun.sensitivity(ctx_bat, "dvalue_dp_global")
    dV_dprice = _dV_dp[_s, 0, p_global_slice(mpc_bat.parameter_manager, "price")]
    dV_dlam = _dV_dp[:, 0, p_global_slice(mpc_bat.parameter_manager, "terminal_value")][:, 0]

    # Cross-check against autograd: one backward pass over the summed value
    # recovers all per-window gradients.
    value_bat.sum().backward()
    assert np.allclose(dV_dprice, price_windows.grad[_s, :, 0].numpy(), rtol=1e-3, atol=1e-6)
    assert np.allclose(dV_dlam, lam_batch.grad[:, 0].numpy(), rtol=1e-3, atol=1e-6)
    # The punchline: dV/dlambda is exactly minus the planned final energy.
    assert np.allclose(dV_dlam, -x_plans[:, -1, 0], atol=1e-6)

    sens_fig, sens_axes = plt.subplots(2, 1, figsize=(9, 6.5))

    _colors = np.where(dV_dprice >= 0, "tab:purple", "tab:green")
    sens_axes[0].bar(_t_win, dV_dprice, width=0.8 * DT_BAT, color=_colors)
    sens_axes[0].axhline(0.0, lw=0.8, color="gray")
    sens_axes[0].plot(_t_win[:-1], u_plans[_s, :, 0] * DT_BAT, ".", color="black",
                      markersize=4, label="$u_k \\Delta t$ from the plan")
    sens_axes[0].set_xlabel("Time since midnight [h]")
    sens_axes[0].set_ylabel("∂V/∂price$_k$  [kWh]")
    sens_axes[0].grid(True, alpha=0.3)
    sens_axes[0].legend(loc="upper right", fontsize=8)
    sens_axes[0].set_title("The 05:00 window buys early (purple, up) and sells the peak (green, down)")

    sens_axes[1].plot(t_day[:96], dV_dlam, color="tab:blue", label="∂V/∂λ")
    sens_axes[1].plot(t_day[:96], -x_plans[:, -1, 0], "--", color="black", lw=1.0,
                      label="$-E_N$ from the plan")
    sens_axes[1].set_xlabel("Window start [h]")
    sens_axes[1].set_ylabel("∂V/∂λ  [kWh]")
    sens_axes[1].grid(True, alpha=0.3)
    sens_axes[1].legend(loc="upper right", fontsize=8)
    sens_axes[1].set_title("How much one extra EUR/kWh of terminal value is worth, per window")

    sens_fig.tight_layout()

    mo.vstack([
        sens_fig,
        mo.md(
            "Each bar is literally the energy traded at that stage (the black "
            "dots overlay $u_k \\Delta t$ from the plan). The last bar is "
            "exactly zero: the stage-$N$ price sits in `p_global` but enters "
            "no cost expression — stage $N$ only carries the terminal cost. "
            "The exact batched sensitivities agree with autograd to numerical "
            "precision."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrap-up

    On top of the series, this notebook added:

    - an **economic cost** — no reference, no tracking, the stage cost is
      the cash flow itself;
    - the two ingredients that keep it well-posed: a quadratic **wear
      term** supplying the Hessian, and a **terminal value** $\lambda$ for
      stored energy — itself a differentiable parameter with the clean
      sensitivity $\partial V / \partial \lambda = -E_N$;
    - **signed** price sensitivities: the battery both buys and sells.

    Natural extensions: charge/discharge efficiency (needs either two
    inputs or a smooth loss model — the nonsmooth $\max(u, 0)$ would break
    the exact-Hessian solver), richer degradation models, a price profile
    with a cheap night valley, or real market prices in place of the
    synthetic day (see the README roadmap).
    """)
    return


if __name__ == "__main__":
    app.run()
