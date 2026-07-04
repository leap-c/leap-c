"""Part 2 — you already have an AcadosOcp.

Most leap-c users arrive with a working acados problem. This notebook builds a
house-heating OCP twice: first the classic way — hand-made ``ca.SX.sym``
parameters in ``model.p``, an ``AcadosOcpSolver``, a bookkeeping loop — then
the leap-c way, where ``AcadosParameterManager`` owns the parameters and
``AcadosDiffMpcTorch`` owns the solver. Same model, same solution, and at the
end: a gradient the classic solver cannot give you.

NOTE: the heating builder taught here is kept in sync with its importable
copy ``nb_utils.heating.build_heating_ocp`` used by the later notebooks.
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
    # 02 — from acados to diff-MPC

    The rest of this series lives in one house: a single room, heated by an
    electric heater, losing heat to the weather outside. This notebook
    formulates the heating OCP **twice**:

    1. **classic acados** — parameters are hand-built symbols in `model.p`,
       set stage by stage on an `AcadosOcpSolver`,
    2. **leap-c** — the same OCP, with the parameters registered on an
       `AcadosParameterManager` and the solver wrapped in
       `AcadosDiffMpcTorch`.

    The two produce the *same solution*, and the diff between the two code
    cells is the honest, complete conversion recipe.
    """)
    return


@app.cell
def _(mo):
    import sys

    sys.path.insert(0, str(mo.notebook_dir().parent))  # notebooks/ root -> nb_utils

    import tempfile
    from pathlib import Path

    import casadi as ca
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from acados_template import AcadosOcp, AcadosOcpSolver

    from nb_utils.data import make_day_profiles
    from nb_utils.diagrams import draw_rc_thermal

    from leap_c.parameters import AcadosParameterManager
    from leap_c.torch import AcadosDiffMpcTorch

    return (
        AcadosDiffMpcTorch,
        AcadosOcp,
        AcadosOcpSolver,
        AcadosParameterManager,
        Path,
        ca,
        draw_rc_thermal,
        make_day_profiles,
        np,
        plt,
        tempfile,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The room: an R1C1 model

    One room temperature $T$ (the state), one heating power $q$ (the
    control). Heat leaks to the outdoors through a thermal resistance $R$
    [K/kW]; the room stores heat in a capacitance $C$ [kWh/K]:

    $$T_{k+1} = T_k + \Delta t \left( \frac{T_{\mathrm{out},k} - T_k}{R\,C}
    + \frac{q_k}{C} \right).$$

    The stage cost trades comfort against the electricity bill,
    $\ell_k = (T_k - T_\mathrm{set})^2 + \mathrm{price}_k\, q_k$, with a
    comfort-only terminal cost, and the heater can only heat:
    $0 \le q \le q_{\max}$.

    One acados convention to know: stage costs are **scaled by the step
    length** $\Delta t$ by default (the sum approximates an integral, the
    terminal cost is not scaled). So with price in EUR/kWh and $q$ in kW,
    the objective is in EUR — and every gradient of `value` inherits that
    scaling.
    """)
    return


@app.cell(hide_code=True)
def _(draw_rc_thermal):
    draw_rc_thermal()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Round 1: classic acados

    Runtime parameters (the weather and price forecasts) are hand-made
    symbols stacked into `model.p`; everything we do not expect to change —
    the physics $R$, $C$ and the setpoint — is a hardcoded constant.
    """)
    return


@app.cell
def _(AcadosOcp, AcadosOcpSolver, ca, np, tempfile):
    N = 32  # 32 stages of 15 min = an 8 h horizon
    DT = 0.25  # time step [h]
    Q_MAX = 12.0  # heater limit [kW]
    R_THERMAL = 2.0  # thermal resistance [K/kW]   (hardcoded!)
    C_THERMAL = 1.5  # thermal capacitance [kWh/K] (hardcoded!)
    T_SET = 21.0  # comfort setpoint [degC]      (hardcoded!)

    ocp_classic = AcadosOcp()
    ocp_classic.model.name = "heating_classic"

    _T = ca.SX.sym("T")
    _q = ca.SX.sym("q")
    ocp_classic.model.x = _T
    ocp_classic.model.u = _q

    # Runtime parameters: hand-built symbols, stacked into model.p.
    _outdoor = ca.SX.sym("outdoor_temp")
    _price = ca.SX.sym("price")
    ocp_classic.model.p = ca.vertcat(_outdoor, _price)
    ocp_classic.parameter_values = np.array([10.0, 0.15])  # defaults

    ocp_classic.model.disc_dyn_expr = _T + DT * (
        (_outdoor - _T) / (R_THERMAL * C_THERMAL) + _q / C_THERMAL
    )

    ocp_classic.cost.cost_type = "EXTERNAL"
    ocp_classic.cost.cost_type_e = "EXTERNAL"
    ocp_classic.model.cost_expr_ext_cost = (_T - T_SET) ** 2 + _price * _q
    ocp_classic.model.cost_expr_ext_cost_e = (_T - T_SET) ** 2

    ocp_classic.constraints.x0 = np.array([19.0])
    ocp_classic.constraints.idxbu = np.array([0])
    ocp_classic.constraints.lbu = np.array([0.0])
    ocp_classic.constraints.ubu = np.array([Q_MAX])

    ocp_classic.solver_options.N_horizon = N
    ocp_classic.solver_options.tf = N * DT
    ocp_classic.solver_options.integrator_type = "DISCRETE"
    ocp_classic.solver_options.nlp_solver_type = "SQP"
    ocp_classic.solver_options.hessian_approx = "EXACT"
    ocp_classic.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    # Keep the generated C code out of the notebook folder.
    _classic_dir = tempfile.mkdtemp(prefix="acados_heating_classic_")
    ocp_classic.code_export_directory = f"{_classic_dir}/c_generated_code"
    solver_classic = AcadosOcpSolver(
        ocp_classic, json_file=f"{_classic_dir}/heating_classic.json", verbose=False
    )
    return DT, N, Q_MAX, solver_classic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Solving means bookkeeping: pin the initial state, then walk the horizon
    and `set` the two parameter values at every stage, every solve.
    """)
    return


@app.cell
def _(DT, N, make_day_profiles, np, plt, solver_classic):
    # An 8 h forecast window starting at 05:00 — the morning price peak
    # sits mid-horizon (make_day_profiles starts at midnight).
    _t_day, outdoor_day, price_day = make_day_profiles(n_steps=144, dt_hours=DT)
    START = 20  # 05:00 in quarter-hours
    outdoor_win = outdoor_day[START : START + N + 1]
    price_win = price_day[START : START + N + 1]
    T0 = 19.0

    # Pin the initial state and set p = [outdoor_k, price_k] at every stage.
    solver_classic.set(0, "lbx", np.array([T0]))
    solver_classic.set(0, "ubx", np.array([T0]))
    for _k in range(N + 1):
        solver_classic.set(_k, "p", np.array([outdoor_win[_k], price_win[_k]]))

    status_classic = solver_classic.solve()
    x_classic = np.array([solver_classic.get(_k, "x")[0] for _k in range(N + 1)])
    u_classic = np.array([solver_classic.get(_k, "u")[0] for _k in range(N)])
    value_classic = solver_classic.get_cost()

    _t = START * DT + DT * np.arange(N + 1)
    classic_fig, classic_axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    classic_axes[0].plot(_t, x_classic, "-o", markersize=3, color="tab:blue")
    classic_axes[0].axhline(21.0, ls="--", lw=0.8, color="gray")
    classic_axes[0].set_ylabel("Room temp [degC]")
    classic_axes[1].step(_t[:-1], u_classic, where="post", color="tab:orange")
    classic_axes[1].set_ylabel("Heating [kW]")
    classic_axes[1].set_xlabel("Time since midnight [h]")
    _ax_p = classic_axes[1].twinx()
    _ax_p.step(_t, price_win, where="post", color="tab:purple", alpha=0.6)
    _ax_p.set_ylabel("Price [EUR/kWh]", color="tab:purple")
    for _ax in classic_axes:
        _ax.grid(True, alpha=0.3)
    classic_fig.suptitle(f"Classic acados plan (status {status_classic})")
    classic_fig.tight_layout()
    classic_fig
    return T0, outdoor_win, price_win, u_classic, value_classic, x_classic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This works — acados is doing all the heavy lifting — but three things
    are **out of reach**:

    - **no batching**: one solver, one problem; a sweep or a training batch
      means a Python loop,
    - **no gradients**: nothing connects the solution to $R$, the setpoint,
      or the prices,
    - **manual bookkeeping**: the `p`-layout (`[outdoor, price]`, stage by
      stage) lives in your head, not in the code.

    ## Round 2: the conversion

    The recipe — what carries over and what changes:

    | | classic | leap-c |
    |---|---|---|
    | dynamics / cost / constraint *expressions* | ✓ | **identical lines** |
    | solver options | ✓ | **identical lines** |
    | parameter symbols | `ca.SX.sym` + `model.p` by hand | `manager.register_parameter(...)` returns the symbol |
    | `model.p` / `p_global` / `parameter_values` | you assign them | **leave unset** — the layer calls `manager.assign_to_ocp(ocp)` |
    | per-stage values | `solver.set(k, "p", ...)` loop | one `params={"outdoor_temp": (B, N+1, 1) array}` |
    | hardcoded constants worth questioning | `R_THERMAL = 2.0` | `differentiable=True` parameter |

    We promote $R$ to a **differentiable** parameter — "would better
    insulation pay off?" is a question a gradient can answer — and register
    the forecasts as **non-differentiable** (values change per solve, no
    gradients needed). One warning for converts: if your existing OCP
    already carries `model.p`, `assign_to_ocp` will *overwrite* it — the
    parameter symbols must come from the manager, or code generation fails
    on the orphaned symbols.
    """)
    return


@app.cell
def _(AcadosOcp, AcadosParameterManager, ca, np):
    def build_heating_ocp(
        N_horizon,
        dt=0.25,
        price_splits="global",
        q_max=8.0,
        name="heating",
    ):
        """Build the parametric heating OCP and its parameter manager.

        NOTE: taught here, importable as ``nb_utils.heating.build_heating_ocp``
        (kept in sync). Always build the OCP and the manager together, fresh: a
        manager is finalized by ``AcadosDiffMpcTorch`` (via ``assign_to_ocp``)
        and must not be reused for a second OCP.
        """
        manager = AcadosParameterManager(N_horizon=N_horizon)

        # Weather forecast: changeable per stage at runtime, but no gradients.
        outdoor_temp = manager.register_parameter(
            name="outdoor_temp", default=np.array([10.0]), differentiable=False
        )
        # Envelope quality: differentiable, so the solver can answer
        # "would better insulation (a larger R) pay off?". Omitting `splits`
        # means splits="global": one value shared by all stages.
        R = manager.register_parameter(
            name="R", default=np.array([2.0]), differentiable=True
        )
        # Comfort reference: one differentiable value shared by all stages.
        comfort_setpoint = manager.register_parameter(
            name="comfort_setpoint", default=np.array([21.0]), differentiable=True
        )
        # Electricity price: differentiable, stage structure set by `price_splits`
        # (notebook 04 explores this argument).
        price = manager.register_parameter(
            name="price", default=np.array([0.15]), differentiable=True, splits=price_splits
        )

        C_THERMAL = 1.5  # thermal capacitance of the room [kWh/K]

        ocp = AcadosOcp()
        ocp.model.name = name

        T = ca.SX.sym("T")  # room temperature [degC]
        q = ca.SX.sym("q")  # heating power [kW]
        ocp.model.x = T
        ocp.model.u = q

        # Identical to the classic cell — only the symbols come from the manager.
        ocp.model.disc_dyn_expr = T + dt * (
            (outdoor_temp - T) / (R * C_THERMAL) + q / C_THERMAL
        )

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = (T - comfort_setpoint) ** 2 + price * q
        ocp.model.cost_expr_ext_cost_e = (T - comfort_setpoint) ** 2

        ocp.constraints.x0 = np.array([20.0])
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([0.0])
        ocp.constraints.ubu = np.array([q_max])

        ocp.solver_options.N_horizon = N_horizon
        ocp.solver_options.tf = N_horizon * dt
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

        # NOTE: model.p / model.p_global stay unset — assign_to_ocp fills them.
        return ocp, manager

    return (build_heating_ocp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrap it

    Two practical constructor arguments while we are here: `verbose=False`
    silences the code generation log, and `export_directory=` pins where
    the generated C code lives (handy for inspection; by default it goes to
    a self-cleaning temporary directory). Printing the module summarizes
    the OCP and every registered parameter.
    """)
    return


@app.cell
def _(
    AcadosDiffMpcTorch,
    DT,
    N,
    Path,
    Q_MAX,
    build_heating_ocp,
    tempfile,
    torch,
):
    export_dir = Path(tempfile.mkdtemp(prefix="leapc_heating_"))

    diff_mpc = AcadosDiffMpcTorch(
        *build_heating_ocp(N, DT, price_splits="stagewise", q_max=Q_MAX, name="heating_leapc"),
        dtype=torch.float64,
        n_batch_init=1,
        verbose=False,
        export_directory=export_dir,
    )

    print(diff_mpc)
    print(f"\ngenerated code in {export_dir}:",
          sorted(p.name for p in export_dir.iterdir())[:4], "...")
    return (diff_mpc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Same problem, same solution

    One call replaces the whole classic bookkeeping loop: the forecasts go
    in as named `(B, N+1, 1)` windows, the initial state as a `(B, 1)`
    tensor. We assert that the plan matches the classic solver's to solver
    precision.
    """)
    return


@app.cell
def _(
    N,
    T0,
    diff_mpc,
    np,
    outdoor_win,
    price_win,
    torch,
    u_classic,
    value_classic,
    x_classic,
):
    ctx, u0, x_leapc, u_leapc, value_leapc = diff_mpc(
        x0=torch.tensor([[T0]], dtype=torch.float64),
        params={
            "outdoor_temp": outdoor_win.reshape(1, N + 1, 1),
            "price": torch.tensor(price_win.reshape(1, N + 1, 1)),
        },
    )

    np.testing.assert_allclose(x_leapc[0, :, 0].detach().numpy(), x_classic, atol=1e-5)
    np.testing.assert_allclose(u_leapc[0, :, 0].detach().numpy(), u_classic, atol=1e-5)
    np.testing.assert_allclose(value_leapc.item(), value_classic, atol=1e-5)

    print(f"status {ctx.status.tolist()} — plans match the classic solver "
          f"(max |ΔT| = {np.abs(x_leapc[0, :, 0].detach().numpy() - x_classic).max():.2e} K)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What you bought: a gradient

    The classic solver ends at the solution. The leap-c layer keeps going:
    pass $R$ as a leaf tensor and ask what one more unit of insulation is
    worth over this horizon.
    """)
    return


@app.cell
def _(N, T0, diff_mpc, mo, outdoor_win, price_win, torch):
    R_param = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    _, _, _, _, value_R = diff_mpc(
        x0=torch.tensor([[T0]], dtype=torch.float64),
        params={
            "outdoor_temp": outdoor_win.reshape(1, N + 1, 1),
            "price": torch.tensor(price_win.reshape(1, N + 1, 1)),
            "R": R_param,
        },
    )
    value_R.backward()

    mo.md(
        f"""
        $\\partial V / \\partial R$ = `{R_param.grad.item():.3f}` per K/kW of
        extra insulation, over one 8 h horizon.

        Negative: a better-insulated house loses less heat, so the optimal
        mix of discomfort and electricity cost gets cheaper. Scale it up —
        thousands of horizons per heating season — and this single number
        is the start of an insulation business case. Notebook 03 makes a
        systematic tour of these gradients.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | classic acados | leap-c |
    |---|---|
    | `ca.SX.sym` + `model.p` + `parameter_values` | `manager.register_parameter(...)` |
    | `solver.set(k, "p", ...)` per stage, per solve | `params={"name": values}` per call |
    | one problem per solve | batch dimension on everything |
    | no derivatives | `.backward()` through the solver |

    The builder taught here is importable as
    `nb_utils.heating.build_heating_ocp` — the later notebooks pull it from
    there instead of repeating it.

    **Next:** `03_gradients_through_the_solver.py` — every way to get a
    gradient out of the layer, and where gradients die.
    """)
    return


if __name__ == "__main__":
    app.run()
