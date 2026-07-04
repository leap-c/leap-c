"""Part 4 — everything AcadosParameterManager.

The complete parameter model in one place: differentiable vs.
non-differentiable, the stage structure (`splits`), the override shape
contract, and the guard rails that catch the classic mistakes. The R1C1
heating OCP from notebook 02 is the playground; the electricity price is the
parameter whose stage structure we vary.
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
    # 04 — everything `AcadosParameterManager`

    So far the parameters just *appeared*: notebook 01 registered a global
    reference, notebook 02 mixed a differentiable scalar (`R`) with a
    non-differentiable stagewise forecast (`outdoor_temp`). This notebook
    lays out the full design space behind those choices:

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
def _(mo):
    import sys

    sys.path.insert(0, str(mo.notebook_dir().parent))  # notebooks/ root -> nb_utils

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from nb_utils.diagrams import draw_rc_thermal
    from nb_utils.heating import build_heating_ocp
    from nb_utils.params import average_per_segment, expand_to_stages

    from leap_c.parameters import AcadosParameterManager
    from leap_c.torch import AcadosDiffMpcTorch

    return (
        AcadosDiffMpcTorch,
        AcadosParameterManager,
        average_per_segment,
        build_heating_ocp,
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

    $$T_{k+1} = T_k + \Delta t \left( \frac{T_{\mathrm{out},k} - T_k}{R\,C}+ \frac{q_k}{C} \right).$$

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The OCP builder is the one taught line by line in notebook 02
    (`nb_utils.heating.build_heating_ocp` is its synced copy) — the only
    news is the `price_splits` argument it forwards to `register_parameter`:

    ```python
    # Weather forecast: changeable per stage at runtime, but no gradients.
    outdoor_temp = manager.register_parameter(
        name="outdoor_temp", default=np.array([10.0]), differentiable=False
    )
    # Envelope quality: one differentiable value shared by all stages.
    R = manager.register_parameter(
        name="R", default=np.array([2.0]), differentiable=True
    )
    # Comfort reference: one differentiable value shared by all stages.
    comfort_setpoint = manager.register_parameter(
        name="comfort_setpoint", default=np.array([21.0]), differentiable=True
    )
    # Electricity price: differentiable, stage structure set by `price_splits`.
    price = manager.register_parameter(
        name="price", default=np.array([0.15]), differentiable=True, splits=price_splits
    )
    ```

    Each call returns a CasADi symbol that goes straight into the dynamics
    and cost expressions — the manager remembers everything else.
    """)
    return


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

    Values in `params` may be numpy arrays or torch tensors; only
    differentiable parameters may carry `requires_grad=True` tensors (the
    guard rails below enforce this). Stage-varying differentiable parameters
    are implemented with a one-hot *indicator* appended to `ocp.model.p`: at
    stage $k$ the symbolic expression returned by `register_parameter`
    evaluates to the block covering $k$. The manager does all of this
    bookkeeping for you — including assembling the flat `p_global`/`p`
    vectors before each solve (its `combine_*` methods are internal plumbing
    you never call yourself). Index `N` of a stagewise value parameterizes
    the *terminal* stage; a parameter that only appears in the dynamics
    (like a weather forecast) is simply inert there — pass the full
    `(B, N+1, dim)` window anyway.

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
def _(mo):
    mo.md("""
    The manager's `repr` shows the resulting parameter layout with categories `differentiable` vs `non-differentiable` and `name`, `shape`, and `default` for each parameter. Compare the structure of each of the managers of the mpc instances we just defined, ranging from one global parameter value over two values to full resolution for every stage.
    """)
    return


@app.cell
def _(mpc_blocks, mpc_global, mpc_stagewise):
    for _label, _mpc in [
        ('price_splits="global"', mpc_global),
        ("price_splits=[11, 24]", mpc_blocks),
        ('price_splits="stagewise"', mpc_stagewise),
    ]:
        print(f"===== {_label} =====")
        print(_mpc.parameter_manager)
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Splits are compile-time structure

    `splits` decides how many price *symbols* exist inside the generated
    solver — changing it changes the `p_global` layout inside the `ocp` and requires building
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
    ## Guard rails

    The manager and the layer catch the classic mistakes early, with
    explicit messages. Three worth knowing before they find you:

    1. a **non-differentiable** parameter cannot carry a grad-tracking
       tensor — register it `differentiable=True` or `.detach()` it,
    2. registration is **closed** once the manager is assigned to an OCP
       (which `AcadosDiffMpcTorch` does in its constructor) — build a fresh
       OCP *and* manager together instead,
    3. a `splits` list must cover the horizon — its last entry has to be
       `N-1` or `N`.
    """)
    return


@app.cell
def _(AcadosParameterManager, N_HEAT, mpc_blocks, mpc_global, np, torch, x0_heat):
    # 1. Non-differentiable parameter with a grad-tracking override.
    try:
        mpc_global(
            x0=x0_heat,
            params={"outdoor_temp": torch.full((1, N_HEAT + 1, 1), 5.0, requires_grad=True)},
        )
    except ValueError as guard1:
        print(f"1. {guard1}\n")

    # 2. Registering on a manager that is already assigned to an OCP.
    try:
        mpc_global.parameter_manager.register_parameter(
            name="too_late", default=np.array([1.0])
        )
    except ValueError as guard2:
        print(f"2. {guard2}\n")

    # 3. A splits list that does not cover the horizon.
    try:
        _bad_manager = AcadosParameterManager(N_horizon=N_HEAT)
        _bad_manager.register_parameter(
            name="price", default=np.array([0.15]), differentiable=True, splits=[5, 11]
        )
    except ValueError as guard3:
        print(f"3. {guard3}\n")

    # Bonus: the shape contract from above, violated on purpose — the two-block
    # instance expects (B, 2, 1), we pass per-stage values.
    try:
        mpc_blocks(
            x0=x0_heat,
            params={"price": torch.full((1, N_HEAT + 1, 1), 0.15)},
        )
    except ValueError as guard4:
        print(f"4. {guard4}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A global parameter, swept interactively

    The comfort setpoint is global and differentiable, so a sweep over it is
    one batched solve — exactly like the `x_ref` override in notebook 01,
    just on a different problem.
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
    ### Two footnotes

    - `AcadosParameterManager(N_horizon, casadi_type="MX")` builds MX
      symbols instead of SX — only needed when your model uses features SX
      cannot express; everything in this series is SX.
    - The manager also stores each parameter's default, so any parameter you
      do *not* override in `params` silently keeps its registered default —
      that is why the sweep above only had to pass two of the four.

    **Next:** `05_batched_solves_and_forecasts.py` puts the batch dimension
    to work — many initial states, many parameter sets, and a day of sliding
    forecast windows in single solver calls.
    """)
    return


if __name__ == "__main__":
    app.run()
