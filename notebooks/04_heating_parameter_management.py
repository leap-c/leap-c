"""Part 4 — parameter management: global vs. stagewise vs. splits.

A heating problem is the natural playground for parameter management: the
weather is an exogenous forecast (changeable, not learnable), the comfort
setpoint is one shared quantity (learnable), and the electricity price varies
over the horizon (learnable, with a stage structure you choose).

This notebook builds the same R1C1 room model with three different stage
structures for the price and compares what the MPC can "see" in each case.
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
    # 04 — parameter management on a heating problem

    The mass-spring-damper notebooks only needed **global** parameters: one
    value per solve, shared by every stage. Real problems are richer:

    - the **weather** is a *forecast* — it changes every solve and every
      stage, but there is nothing to learn about it,
    - the **comfort setpoint** is one value shared by all stages — and we
      may want gradients with respect to it,
    - the **electricity price** varies over the horizon — and how finely the
      OCP resolves that variation is a *structural* choice: `splits`.

    `AcadosParameterManager.register_parameter` covers all of these with two
    arguments: `differentiable` (gradients or not) and `splits` (stage
    structure).
    """)
    return


@app.cell
def _():
    import casadi as ca
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from acados_template import AcadosOcp

    from nb_utils.diagrams import draw_rc_thermal
    from nb_utils.params import average_per_segment, expand_to_stages

    from leap_c.ocp.acados.parameters import AcadosParameterManager
    from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

    torch.set_default_dtype(torch.float64)
    return (
        AcadosDiffMpcTorch,
        AcadosOcp,
        AcadosParameterManager,
        average_per_segment,
        ca,
        draw_rc_thermal,
        expand_to_stages,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The R1C1 room model

    One room temperature $T$ (the state), one heating power $q$ (the
    control). Heat leaks to the outdoors through a thermal resistance $R$;
    the room stores heat in a capacitance $C$:

    $$T_{k+1} = T_k + \Delta t \left( \frac{T_{\mathrm{out},k} - T_k}{R\,C}
    + \frac{q_k}{C} \right).$$

    The stage cost trades comfort against energy cost,

    $$\ell_k = (T_k - T_{\mathrm{set}})^2 + \mathrm{price}_k \, q_k,$$

    with a comfort-only terminal cost. The heater can only heat:
    $0 \le q \le q_{\max}$.
    """)
    return


@app.cell(hide_code=True)
def _(draw_rc_thermal):
    draw_rc_thermal()
    return


@app.cell
def _(AcadosOcp, AcadosParameterManager, ca, np):
    R_THERMAL = 2.0  # thermal resistance to the outdoors [K/kW]
    C_THERMAL = 1.5  # thermal capacitance of the room [kWh/K]

    def build_heating_ocp(N_horizon, dt=0.25, price_splits="global", q_max=8.0, name="heating"):
        # NOTE: manager and OCP are always built together, fresh (see notebook 01).
        # The same builder is importable as nb_utils.heating.build_heating_ocp.
        manager = AcadosParameterManager(N_horizon=N_horizon)

        # Weather forecast: changeable per stage at runtime, but no gradients.
        outdoor_temp = manager.register_parameter(
            name="outdoor_temp", default=np.array([10.0]), differentiable=False
        )
        # Comfort reference: one differentiable value shared by all stages.
        comfort_setpoint = manager.register_parameter(
            name="comfort_setpoint", default=np.array([21.0]), differentiable=True
        )
        # Electricity price: differentiable, stage structure set by `price_splits`.
        price = manager.register_parameter(
            name="price", default=np.array([0.15]), differentiable=True, splits=price_splits
        )

        ocp = AcadosOcp()
        ocp.model.name = name

        T = ca.SX.sym("T")  # room temperature [degC]
        q = ca.SX.sym("q")  # heating power [kW]
        ocp.model.x = T
        ocp.model.u = q

        ocp.model.disc_dyn_expr = T + dt * (
            (outdoor_temp - T) / (R_THERMAL * C_THERMAL) + q / C_THERMAL
        )

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = (T - comfort_setpoint) ** 2 + price * q
        ocp.model.cost_expr_ext_cost_e = (T - comfort_setpoint) ** 2

        ocp.constraints.x0 = np.array([20.0])

        # The heater can only heat, up to q_max.
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([0.0])
        ocp.constraints.ubu = np.array([q_max])

        ocp.solver_options.N_horizon = N_horizon
        ocp.solver_options.tf = N_horizon * dt
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

        return ocp, manager

    return (build_heating_ocp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The interface matrix

    | registered as | lives in | override shape in `params` | gradients |
    |---|---|---|---|
    | `differentiable=False` | `model.p` (stagewise runtime values) | `(B, N+1, dim)` | no |
    | `differentiable=True, splits="global"` | `model.p_global` | `(B, dim)` | yes |
    | `differentiable=True, splits=[...]` or `int` | `model.p_global`, one block per *segment* | `(B, n_segments, dim)` | yes |
    | `differentiable=True, splits="stagewise"` | `model.p_global`, one block per *stage* | `(B, N+1, dim)` | yes |

    Stage-varying differentiable parameters are implemented with a one-hot
    *indicator* appended to `model.p`: at stage $k$ the symbolic expression
    returned by `register_parameter` evaluates to the block covering $k$.
    All of this is bookkeeping the manager does for you.

    Below we build **three MPC instances** that differ *only* in the price's
    `splits` — one price for the whole horizon, two price blocks, and one
    price per stage.
    """)
    return


@app.cell
def _(AcadosDiffMpcTorch, build_heating_ocp, torch):
    N_HEAT = 24  # 24 stages of 15 min = a 6 h horizon
    DT_HEAT = 0.25  # time step [h]
    N_SETPOINTS = 17  # size of the interactive setpoint sweep below

    # Distinct model names keep the generated solver code of the three
    # instances from colliding.
    mpc_global = AcadosDiffMpcTorch(
        *build_heating_ocp(N_HEAT, DT_HEAT, price_splits="global", name="heating_global"),
        dtype=torch.float64,
        n_batch_init=1,
        verbose=False,
    )
    mpc_blocks = AcadosDiffMpcTorch(
        *build_heating_ocp(N_HEAT, DT_HEAT, price_splits=[11, N_HEAT], name="heating_blocks"),
        dtype=torch.float64,
        n_batch_init=1,
        verbose=False,
    )
    mpc_stagewise = AcadosDiffMpcTorch(
        *build_heating_ocp(N_HEAT, DT_HEAT, price_splits="stagewise", name="heating_stagewise"),
        dtype=torch.float64,
        n_batch_init=N_SETPOINTS,
        verbose=False,
    )
    return DT_HEAT, N_HEAT, N_SETPOINTS, mpc_blocks, mpc_global, mpc_stagewise


@app.cell(hide_code=True)
def _(mo, mpc_stagewise):
    mo.md(
        "The manager's `repr` shows the resulting parameter layout — here for "
        "the stagewise instance (note the per-stage `price_k_k` blocks and the "
        "`indicator` appended to the non-differentiable parameters):\n\n"
        f"```\n{mpc_stagewise.parameter_manager!r}\n```"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Splits are compile-time structure

    `splits` decides how many price *symbols* exist inside the generated
    solver — changing it changes the `p_global` layout and requires building
    a new solver. The price *values* are runtime: pass them per solve through
    `params`, shaped `(B, n_segments, 1)`.

    To compare the three structures fairly, we give each instance the **same
    underlying stagewise price signal** — cheap energy with an expensive peak
    — projected onto its segments by averaging. The room starts cool at
    18 °C with the setpoint at 21 °C.
    """)
    return


@app.cell
def _(
    DT_HEAT,
    N_HEAT,
    average_per_segment,
    expand_to_stages,
    mpc_blocks,
    mpc_global,
    mpc_stagewise,
    np,
    plt,
    torch,
):
    # The "true" per-stage price signal: cheap, with an expensive peak.
    price_stage_profile = np.full(N_HEAT + 1, 0.15)
    price_stage_profile[14:21] = 0.45

    x0_heat = torch.tensor([[18.0]])
    t_stages = DT_HEAT * np.arange(N_HEAT + 1)

    split_fig, split_axes = plt.subplots(3, 3, figsize=(11, 7), sharex=True, sharey="row")
    for _col, (_label, _mpc) in enumerate(
        [
            ('price_splits="global"', mpc_global),
            ("price_splits=[11, 24]", mpc_blocks),
            ('price_splits="stagewise"', mpc_stagewise),
        ]
    ):
        _manager = _mpc.parameter_manager
        # Project the stagewise signal onto this instance's segments (averaging),
        # then expand back to stages to plot what the OCP actually sees.
        _segments = average_per_segment(_manager, "price", price_stage_profile)
        # Override shape per the interface matrix: (B, n_segments, dim) for a
        # stage-varying parameter, plain (B, dim) for a global one.
        if _manager.parameters["price"].is_stage_varying:
            _price_param = torch.tensor(_segments).reshape(1, -1, 1)
        else:
            _price_param = torch.tensor(_segments).reshape(1, 1)
        _price_seen = expand_to_stages(_manager, "price", _segments)

        _, _, _x, _u, _ = _mpc(x0=x0_heat, params={"price": _price_param})

        split_axes[0, _col].step(t_stages, price_stage_profile, where="post",
                                 color="lightgray", label="true signal")
        split_axes[0, _col].step(t_stages, _price_seen, where="post",
                                 color="tab:purple", label="price seen by OCP")
        split_axes[0, _col].set_title(_label, fontsize=10)
        split_axes[1, _col].plot(t_stages, _x[0].detach().numpy().ravel(), color="tab:blue")
        split_axes[1, _col].axhline(21.0, ls="--", lw=0.8, color="gray")
        split_axes[2, _col].step(t_stages[:-1], _u[0].detach().numpy().ravel(),
                                 where="post", color="tab:orange")

    split_axes[0, 0].set_ylabel("Price [EUR/kWh]")
    split_axes[1, 0].set_ylabel("Room temp [degC]")
    split_axes[2, 0].set_ylabel("Heating [kW]")
    for _ax in split_axes.ravel():
        _ax.grid(True, alpha=0.3)
    for _ax in split_axes[2]:
        _ax.set_xlabel("Time [h]")
    split_axes[0, 0].legend(fontsize=8)
    split_fig.suptitle("The same price signal under three different `splits` structures")
    split_fig.tight_layout()
    split_fig
    return (x0_heat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The global instance cannot see the peak at all — it heats as if energy
    always cost the average price. The two-block instance knows the second
    half is more expensive on average. Only the stagewise instance
    **pre-heats**: it buys heat just before the peak and coasts through it.

    ### Shape contract

    Stage-varying differentiable overrides are shaped
    `(B, n_segments, dim)` — *per segment*, **not** per stage (they only
    coincide for `splits="stagewise"`, where `n_segments = N+1`). Passing
    the wrong second dimension raises immediately with the expected shape.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A global parameter, swept interactively

    The comfort setpoint is global and differentiable, so a sweep over it is
    one batched solve — exactly like the `r_diag_sqrt` sweep in notebook 01,
    just on a different problem. The slider picks a precomputed solution;
    it never solves.
    """)
    return


@app.cell
def _(N_HEAT, N_SETPOINTS, mpc_stagewise, np, torch, x0_heat):
    setpoints = np.linspace(17.0, 25.0, N_SETPOINTS)

    _price_profile = np.full(N_HEAT + 1, 0.15)
    _price_profile[14:21] = 0.45
    _price_batch = torch.tensor(
        np.tile(_price_profile.reshape(1, -1, 1), (N_SETPOINTS, 1, 1))
    )

    _, _, x_sp, u_sp, _ = mpc_stagewise(
        x0=x0_heat.repeat(N_SETPOINTS, 1),
        params={
            "comfort_setpoint": torch.tensor(setpoints).reshape(-1, 1),
            "price": _price_batch,
        },
    )
    x_sp = x_sp.detach().numpy()  # (N_SETPOINTS, N+1, 1)
    u_sp = u_sp.detach().numpy()  # (N_SETPOINTS, N, 1)
    return setpoints, u_sp, x_sp


@app.cell
def _(N_SETPOINTS, mo):
    setpoint_slider = mo.ui.slider(
        start=0,
        stop=N_SETPOINTS - 1,
        step=1,
        value=N_SETPOINTS // 2,
        label="sweep index for the comfort setpoint",
        show_value=True,
    )
    return (setpoint_slider,)


@app.cell
def _(DT_HEAT, mo, np, plt, setpoint_slider, setpoints, u_sp, x_sp):
    # No solve here — the slider picks one of the precomputed solutions.
    _i = setpoint_slider.value
    _t_state = DT_HEAT * np.arange(x_sp.shape[1])
    _t_ctrl = DT_HEAT * np.arange(u_sp.shape[1])

    sp_fig, sp_axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for _j in range(len(setpoints)):
        sp_axes[0].plot(_t_state, x_sp[_j, :, 0], color="lightgray", lw=1)
        sp_axes[1].step(_t_ctrl, u_sp[_j, :, 0], where="post", color="lightgray", lw=1)
    sp_axes[0].plot(
        _t_state, x_sp[_i, :, 0], "-o", markersize=3, color="tab:blue",
        label=f"T_set = {setpoints[_i]:.1f} °C",
    )
    sp_axes[0].axhline(setpoints[_i], ls="--", lw=0.8, color="tab:red", label="setpoint")
    sp_axes[1].step(
        _t_ctrl, u_sp[_i, :, 0], where="post", color="tab:orange",
        label=f"T_set = {setpoints[_i]:.1f} °C",
    )
    sp_axes[0].set_ylabel("Room temp [degC]")
    sp_axes[1].set_ylabel("Heating [kW]")
    sp_axes[1].set_xlabel("Time [h]")
    for ax_sp in sp_axes:
        ax_sp.grid(True, alpha=0.3)
        ax_sp.legend(loc="upper right")
    sp_fig.suptitle("Comfort setpoint sweep (one batched solve, peak price)")
    sp_fig.tight_layout()
    mo.vstack([setpoint_slider, sp_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Next:** `05_heating_forecasts.py` slides a real day of weather and
    price forecasts through the horizon — the non-differentiable stagewise
    interface in action — and takes gradients with respect to a stagewise
    price forecast.
    """)
    return


if __name__ == "__main__":
    app.run()
