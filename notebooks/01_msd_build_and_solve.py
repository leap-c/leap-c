"""Part 1 — build a parametric OCP and solve it with ``AcadosDiffMpcTorch``.

This notebook shows how to register parameters with ``AcadosParameterManager``,
build an ``AcadosOcp`` around them, wrap everything in leap-c's
``AcadosDiffMpcTorch``, solve for one observation, and read back the planned
trajectory. An interactive slider at the end sweeps a cost parameter.

The control problem (a mass-spring-damper) is deliberately trivial so that the
focus stays on the leap-c API. Unlike ``intro.py`` (which uses the high-level
``create_planner`` wrapper), here we build the OCP by hand to expose every step.
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
    # 01 — build a parametric OCP and solve it

    `AcadosDiffMpcTorch` is a `torch.nn.Module` that wraps a *parametric*
    acados OCP. You hand it a batch of initial states (observations) and a
    dictionary of parameters; it solves the OCP and returns the optimal
    trajectory **and** lets gradients flow back to the parameters — so an
    MPC controller can live inside a PyTorch computation graph.

    This first notebook of the series walks through:

    1. registering parameters and building the OCP,
    2. constructing the differentiable MPC,
    3. solving for one observation and plotting the planned trajectory,
    4. sweeping a parameter interactively.

    The series continues with value/policy maps (02), gradients through the
    solver (03), and parameter management on a heating problem (04, 05).
    """)
    return


@app.cell
def _():
    import casadi as ca
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from acados_template import AcadosOcp

    from leap_c.ocp.acados.parameters import AcadosParameterManager
    from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

    # acados is double precision — use float64 tensors throughout so gradients
    # and shape checks line up exactly.
    torch.set_default_dtype(torch.float64)
    return (
        AcadosDiffMpcTorch,
        AcadosOcp,
        AcadosParameterManager,
        ca,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The parametric OCP

    We regulate a 1-D mass-spring-damper to the origin. The state is
    $x = [\,p,\ v\,]$ (position, velocity) and the control is the force $F$.
    With mass $m$, damping $c$ and stiffness $k$, the (Euler-discretised)
    dynamics are $x_{t+1} = A\,x_t + B\,F_t$ with

    $$A = \begin{bmatrix} 1 & \Delta t \\ -\Delta t\,k/m & 1 - \Delta t\,c/m \end{bmatrix},
    \qquad B = \begin{bmatrix} 0 \\ \Delta t / m \end{bmatrix}.$$

    The cost is a quadratic regulator to zero, with diagonal weights stored
    as their square roots so they stay positive: stage weight
    $\mathrm{diag}(q,\,r)^2$ and terminal weight $\mathrm{diag}(p)^2$.
    Position and velocity have soft box constraints $[-2, 2]$ and the force
    is hard-bounded to $[-0.5, 0.5]$.

    Every quantity in **bold** below — `q_diag_sqrt`, `r_diag_sqrt`,
    `p_diag_sqrt`, `mass`, `damping`, `stiffness` — is a *differentiable*
    parameter we can override at solve time and differentiate through.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src=str(mo.notebook_dir() / "assets" / "mass_spring_damper.svg"),
        width=440,
        caption=(
            "The mass-spring-damper we regulate: a block of mass m on a spring "
            "(stiffness k) and a damper (constant c), driven by the force F. "
            "Its displacement x ≡ our position p; the velocity v = ẋ completes "
            "the state x = [p, v]. Diagram by Ilmari Karonen, public domain "
            "(Wikimedia Commons)."
        ),
    )
    return


@app.cell
def _(AcadosOcp, AcadosParameterManager, ca, np):
    def build_msd_ocp(N_horizon, dt):
        # NOTE: manager and OCP are usually built together, fresh: the manager is
        # finalized when AcadosDiffMpcTorch assigns it to the OCP and must not be
        # does not allow to register new parameters after
        manager = AcadosParameterManager(N_horizon=N_horizon)

        def register(name, default):
            # differentiable=True -> part of p_global, gradients flow.
            return manager.register_parameter(
                name=name, default=default, differentiable=True
            )

        # Cost parameters
        q_diag_sqrt = register("q_diag_sqrt", np.sqrt(np.array([5.0, 0.2])))
        r_diag_sqrt = register("r_diag_sqrt", np.sqrt(np.array([0.08])))
        p_diag_sqrt = register("p_diag_sqrt", np.sqrt(np.array([5.0, 0.5])))

        # Model parameters
        mass = register("mass", np.array([1.5]))
        damping = register("damping", np.array([0.7]))
        stiffness = register("stiffness", np.array([2.0]))

        ocp = AcadosOcp()
        ocp.model.name = "mass_spring_damper"

        # State [position, velocity] and control [force].
        ocp.model.x = ca.vertcat(ca.SX.sym("p"), ca.SX.sym("v"))
        ocp.model.u = ca.SX.sym("F")

        # Parametric discrete-time dynamics x_{t+1} = A x + B F.
        A = ca.vertcat(
            ca.horzcat(1.0, dt),
            ca.horzcat(-dt * stiffness / mass, 1.0 - dt * damping / mass),
        )
        B = ca.vertcat(0.0, dt / mass)
        ocp.model.disc_dyn_expr = A @ ocp.model.x + B @ ocp.model.u

        # Quadratic regulator cost, weights built from the sqrt-diagonal parameters.
        W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = W_sqrt @ W_sqrt.T
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
        ocp.cost.yref = np.zeros((3,))

        W_e_sqrt = ca.diag(p_diag_sqrt)
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = W_e_sqrt @ W_e_sqrt.T
        ocp.model.cost_y_expr_e = ocp.model.x
        ocp.cost.yref_e = np.zeros((2,))

        # Initial state — a nominal value, overwritten on every solve.
        ocp.constraints.x0 = np.array([1.0, 0.0])

        # Soft box constraints on the state.
        ocp.constraints.idxbx = np.array([0, 1])
        ocp.constraints.lbx = np.array([-2.0, -2.0])
        ocp.constraints.ubx = np.array([2.0, 2.0])
        ocp.constraints.idxsbx = np.array([0, 1])
        ocp.cost.Zl = ocp.cost.Zu = np.array([1e3, 1e3])
        ocp.cost.zl = ocp.cost.zu = np.array([0.0, 0.0])

        # Hard box constraint on the force.
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([-0.5])
        ocp.constraints.ubu = np.array([0.5])

        ocp.solver_options.N_horizon = N_horizon
        ocp.solver_options.tf = N_horizon * dt
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

        # NOTE: we never set ocp.model.p / ocp.model.p_global ourselves — the
        # AcadosDiffMpcTorch constructor calls manager.assign_to_ocp(ocp) for us
        # and sets them from the registered parameters.
        return ocp, manager

    return (build_msd_ocp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the differentiable MPC

    `AcadosDiffMpcTorch(ocp, manager)` wires the parameters into the OCP,
    generates the C code, and compiles a *batched* acados solver. We size
    the batch (`n_batch_init`) to the largest batch we will use — the
    9-point parameter sweep at the end of this notebook.
    """)
    return


@app.cell
def _(AcadosDiffMpcTorch, build_msd_ocp, torch):
    N_HORIZON = 50  # MPC horizon (stages 0 .. N_HORIZON)
    DT = 0.1  # time step [s]
    N_SWEEP = 9  # size of the interactive parameter sweep below

    ocp, manager = build_msd_ocp(N_horizon=N_HORIZON, dt=DT)

    diff_mpc = AcadosDiffMpcTorch(
        ocp,
        manager,
        dtype=torch.float64,
        n_batch_init=N_SWEEP,
        verbose=False,
    )
    return DT, N_SWEEP, diff_mpc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve for one observation

    Calling the module returns `(ctx, u0, x, u, value)`:

    - `ctx` — solver context (status, warm-start iterate, cached sensitivities),
    - `u0` — first optimal action, shape `(B, nu)`,
    - `x` — planned state trajectory, shape `(B, N+1, nx)`,
    - `u` — planned control trajectory, shape `(B, N, nu)`,
    - `value` — optimal cost, shape `(B, 1)`.

    The batch dimension `B` always comes first — here `B = 1`. With no
    `params` given, the registered defaults are used.
    """)
    return


@app.cell
def _(diff_mpc, mo, torch):
    x0 = torch.tensor([[0.5, 0.0]])  # one observation: displaced 0.5 m, at rest

    ctx, u0, x_traj, u_traj, value = diff_mpc(x0=x0)

    mo.md(
        f"""
        **Solver status:** `{ctx.status.tolist()}`  (0 = converged)

        **First action F₀:** `{u0.item():.4f}` N

        **MPC value V(x₀):** `{value.item():.4f}`

        **Trajectory shapes:** `x {tuple(x_traj.shape)}`, `u {tuple(u_traj.shape)}`
        """
    )
    return u_traj, x0, x_traj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Planned trajectory

    The MPC drives the displaced mass back to the origin while respecting
    the force limits.
    """)
    return


@app.cell
def _(DT, np, plt, u_traj, x_traj):
    x_plan = x_traj[0].detach().numpy()  # (N+1, nx)
    u_plan = u_traj[0].detach().numpy()  # (N, nu)
    t_state = DT * np.arange(x_plan.shape[0])
    t_ctrl = DT * np.arange(u_plan.shape[0])

    traj_fig, traj_axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    traj_axes[0].plot(t_state, x_plan[:, 0], "-o", markersize=3)
    traj_axes[0].set_ylabel("Position [m]")
    traj_axes[1].plot(t_state, x_plan[:, 1], "-o", markersize=3, color="tab:green")
    traj_axes[1].set_ylabel("Velocity [m/s]")
    traj_axes[2].step(t_ctrl, u_plan[:, 0], where="post", color="tab:orange")
    traj_axes[2].axhline(0.5, ls="--", lw=0.8, color="gray")
    traj_axes[2].axhline(-0.5, ls="--", lw=0.8, color="gray")
    traj_axes[2].set_ylabel("Force [N]")
    traj_axes[2].set_xlabel("Time [s]")
    for ax_t in traj_axes:
        ax_t.grid(True, alpha=0.3)
    traj_fig.suptitle("Planned trajectory (default parameters)")
    traj_fig.tight_layout()
    traj_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Passing a parameter

    Parameters are overridden by name through the `params` dict. A global
    differentiable parameter takes shape `(B, ds)` — one row per batch
    element. So a *sweep* over a parameter is just a **single batched
    solve**: below, batch element $i$ uses the $i$-th value of
    `r_diag_sqrt`, the (square-root) control-cost weight.
    """)
    return


@app.cell
def _(N_SWEEP, diff_mpc, np, torch, x0):
    r_values = np.linspace(0.05, 1.0, N_SWEEP)

    # One batched solve: row i of the params dict pairs with row i of x0.
    _x0_batch = x0.repeat(N_SWEEP, 1)
    _params = {"r_diag_sqrt": torch.tensor(r_values).reshape(-1, 1)}
    _, _, x_sweep, u_sweep, _ = diff_mpc(x0=_x0_batch, params=_params)

    x_sweep = x_sweep.detach().numpy()  # (N_SWEEP, N+1, nx)
    u_sweep = u_sweep.detach().numpy()  # (N_SWEEP, N, nu)
    return r_values, u_sweep, x_sweep


@app.cell
def _(N_SWEEP, mo):
    r_slider = mo.ui.slider(
        start=0,
        stop=N_SWEEP - 1,
        step=1,
        value=N_SWEEP // 2,
        label="sweep index for r_diag_sqrt",
        show_value=True,
    )
    return (r_slider,)


@app.cell
def _():
    return


@app.cell
def _(DT, mo, np, plt, r_slider, r_values, u_sweep, x_sweep):
    _i = r_slider.value

    _t_state = DT * np.arange(x_sweep.shape[1])
    _t_ctrl = DT * np.arange(u_sweep.shape[1])

    sweep_fig, sweep_axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for _j in range(len(r_values)):
        sweep_axes[0].plot(_t_state, x_sweep[_j, :, 0], color="lightgray", lw=1)
        sweep_axes[1].step(_t_ctrl, u_sweep[_j, :, 0], where="post", color="lightgray", lw=1)
    sweep_axes[0].plot(
        _t_state, x_sweep[_i, :, 0], "-o", markersize=3, color="tab:blue",
        label=f"r_diag_sqrt = {r_values[_i]:.2f}",
    )
    sweep_axes[1].step(
        _t_ctrl, u_sweep[_i, :, 0], where="post", color="tab:orange",
        label=f"r_diag_sqrt = {r_values[_i]:.2f}",
    )
    sweep_axes[1].axhline(0.5, ls="--", lw=0.8, color="gray")
    sweep_axes[1].axhline(-0.5, ls="--", lw=0.8, color="gray")
    sweep_axes[0].set_ylabel("Position [m]")
    sweep_axes[1].set_ylabel("Force [N]")
    sweep_axes[1].set_xlabel("Time [s]")
    for ax_s in sweep_axes:
        ax_s.grid(True, alpha=0.3)
        ax_s.legend(loc="upper right")
    sweep_fig.suptitle("Effect of the control-cost weight r_diag_sqrt")
    sweep_fig.tight_layout()
    mo.vstack([r_slider, sweep_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A larger `r_diag_sqrt` penalises the force more, so the controller acts
    more gently and the mass returns to the origin more slowly.

    **Next in the series:**

    - `02_msd_value_policy_maps.py` — the MPC as a value function and a
      policy over the whole state space,
    - `03_msd_sensitivities.py` — differentiating through the solver.
    """)
    return


if __name__ == "__main__":
    app.run()
