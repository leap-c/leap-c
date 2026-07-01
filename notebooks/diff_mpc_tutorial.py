"""A differentiable-MPC tutorial built around ``AcadosDiffMpcTorch``.

This notebook shows how to take a *parametric* optimal control problem (OCP),
wrap it in leap-c's ``AcadosDiffMpcTorch``, solve it for a given observation,
and read back the planned trajectory, the value function, the policy, and
gradients — all in a few lines.

The control problem (a mass-spring-damper) is deliberately trivial so that the
focus stays on the leap-c API. Because the state is only two-dimensional
(position, velocity), we can plot the MPC value function and policy as full
2-D maps over the state space.

Unlike ``intro.py`` (which uses the high-level ``create_planner`` wrapper),
here we build the OCP and the differentiable MPC by hand to expose every step.
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
    # leap-c — differentiable MPC with `AcadosDiffMpcTorch`

    `AcadosDiffMpcTorch` is a `torch.nn.Module` that wraps a *parametric*
    acados OCP. You hand it a batch of initial states (observations) and a
    dictionary of parameters; it solves the OCP and returns the optimal
    trajectory **and** lets gradients flow back to the parameters — so an
    MPC controller can live inside a PyTorch computation graph.

    This tutorial walks through:

    1. registering parameters and building the OCP,
    2. constructing the differentiable MPC,
    3. solving for one observation and plotting the planned trajectory,
    4. seeing how a parameter changes the solution,
    5. plotting the **value function** and **policy** over the state space,
    6. taking a gradient of the value with respect to a parameter.
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
    With mass $m$, damping $b$ and stiffness $k$, the (Euler-discretised)
    dynamics are $x_{t+1} = A\,x_t + B\,F_t$ with

    $$A = \begin{bmatrix} 1 & \Delta t \\ -\Delta t\,k/m & 1 - \Delta t\,b/m \end{bmatrix},
    \qquad B = \begin{bmatrix} 0 \\ \Delta t / m \end{bmatrix}.$$

    The cost is a quadratic regulator to zero, with diagonal weights stored
    as their square roots so they stay positive: stage weight
    $\mathrm{diag}(q,\,r)^2$ and terminal weight $\mathrm{diag}(p)^2$.
    Position and velocity have soft box constraints $[-2, 2]$ and the force
    is hard-bounded to $[-0.5, 0.5]$.

    Every quantity in **bold** below — `q_diag_sqrt`, `r_diag_sqrt`,
    `p_diag_sqrt`, `mass`, `damping`, `stiffness` — is a *learnable*
    parameter we can override at solve time and differentiate through.
    """)
    return


@app.cell
def _(AcadosOcp, AcadosParameterManager, ca, np):
    def export_ocp(N_horizon, dt):
        # Build the parameter manager and the OCP together, returning both as a fresh
        # pair on every call. Keeps marimo re-runs safe.
        manager = AcadosParameterManager(N_horizon=N_horizon)

        def register(name, default):
            # differentiable=True -> learnable.
            return manager.register_parameter(
                name=name, default=default, differentiable=True
            )

        q_diag_sqrt = register("q_diag_sqrt", np.sqrt(np.array([5.0, 0.2])))
        r_diag_sqrt = register("r_diag_sqrt", np.sqrt(np.array([0.08])))
        p_diag_sqrt = register("p_diag_sqrt", np.sqrt(np.array([5.0, 0.5])))
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
        ocp.constraints.lbu = np.array([-.5])
        ocp.constraints.ubu = np.array([.5])

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

    return (export_ocp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the differentiable MPC

    `AcadosDiffMpcTorch(ocp, manager)` wires the parameters into the OCP,
    generates the C code, and compiles a *batched* acados solver. We size
    the batch (`n_batch_init`) to the largest batch we will use — the state
    grid for the value/policy maps further down.
    """)
    return


@app.cell
def _(AcadosDiffMpcTorch, export_ocp, torch):
    N_HORIZON = 50  # MPC horizon (stages 0 .. N_HORIZON)
    DT = 0.1  # time step [s]
    GRID_N = 8  # value/policy maps are evaluated on a GRID_N x GRID_N grid

    ocp, manager = export_ocp(N_horizon=N_HORIZON, dt=DT)

    diff_mpc = AcadosDiffMpcTorch(
        ocp,
        manager,
        dtype=torch.float64,
        n_batch_init=GRID_N * GRID_N,
        verbose=False,
    )
    return DT, GRID_N, diff_mpc


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

    Parameters are overridden by name through the `params` dict. A learnable
    global parameter takes shape `(B, ds)`. Here we sweep `r_diag_sqrt`, the
    (square-root) control-cost weight: a larger value penalises the force
    more, so the controller acts more gently and the mass returns slowly.
    """)
    return


@app.cell
def _(DT, diff_mpc, np, plt, torch, x0):
    r_values = [0.05, 0.15, 0.30, 0.5, 1.0]

    sweep_fig, sweep_axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    for r in r_values:
        params = {"r_diag_sqrt": torch.tensor([[r]])}
        _, _, xr, ur, _ = diff_mpc(x0=x0, params=params)
        xr = xr[0].detach().numpy()
        ur = ur[0].detach().numpy()
        sweep_axes[0].plot(
            DT * np.arange(xr.shape[0]), xr[:, 0], "-o", markersize=3, label=f"r_sqrt={r}"
        )
        sweep_axes[1].step(
            DT * np.arange(ur.shape[0]), ur[:, 0], where="post", label=f"r_sqrt={r}"
        )
    sweep_axes[0].set_ylabel("Position [m]")
    sweep_axes[1].set_ylabel("Force [N]")
    sweep_axes[1].set_xlabel("Time [s]")
    for ax_s in sweep_axes:
        ax_s.grid(True, alpha=0.3)
        ax_s.legend()
    sweep_fig.suptitle("Effect of the control-cost weight r_diag_sqrt")
    sweep_fig.tight_layout()
    sweep_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Value function and policy over the state space

    Because the state is 2-D we can solve the MPC from a whole grid of
    initial states in a single batched call, then plot:

    - the **value function** $V(x_0)$ — the optimal cost-to-go, a bowl
      centred at the origin, and
    - the **policy** $\pi(x_0) = F_0^{*}(x_0)$ — the first optimal action,
      a feedback law that saturates at the $\pm 0.5$ force limit.

    We evaluate on an *inner* grid $[-1.8, 1.8]^2$ to stay away from the
    soft state bounds, whose slack penalties would otherwise distort the
    value bowl near the edges.
    """)
    return


@app.cell
def _(GRID_N, diff_mpc, np, torch):
    axis = np.linspace(-1.8, 1.8, GRID_N)
    pos_grid, vel_grid = np.meshgrid(axis, axis)  # (G, G) each
    grid = torch.tensor(
        np.stack([pos_grid.ravel(), vel_grid.ravel()], axis=1)
    )  # (G*G, 2)

    _, u0_g, _, _, value_g = diff_mpc(x0=grid)
    value_map = value_g.detach().numpy().reshape(GRID_N, GRID_N)
    policy_map = u0_g.detach().numpy().reshape(GRID_N, GRID_N)
    return policy_map, pos_grid, value_map, vel_grid


@app.cell
def _(plt, policy_map, pos_grid, value_map, vel_grid):
    map_fig, map_axes = plt.subplots(1, 2, figsize=(11, 4.5))

    c0 = map_axes[0].contourf(pos_grid, vel_grid, value_map, levels=30, cmap="viridis")
    map_fig.colorbar(c0, ax=map_axes[0])
    map_axes[0].set_title("Value function V(x₀)")

    c1 = map_axes[1].contourf(pos_grid, vel_grid, policy_map, levels=30, cmap="coolwarm")
    map_fig.colorbar(c1, ax=map_axes[1], label="Force F₀ [N]")
    map_axes[1].set_title("Policy π(x₀) = F₀*(x₀)")

    for ax_m in map_axes:
        ax_m.set_xlabel("Position [m]")
        ax_m.set_ylabel("Velocity [m/s]")
    map_fig.tight_layout()
    map_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Differentiability — the whole point

    The solver sits inside the autograd graph, so we can differentiate its
    outputs with respect to any learnable parameter. Pass the parameter as a
    leaf tensor with `requires_grad=True`, call `.backward()`, and read the
    gradient — for example $\partial V / \partial m$, how the optimal cost
    responds to the mass.
    """)
    return


@app.cell
def _(diff_mpc, mo, torch, x0):
    mass_param = torch.tensor([[1.5]], requires_grad=True)

    _, _, _, _, value_m = diff_mpc(x0=x0, params={"mass": mass_param})
    value_m.sum().backward()  # reverse-mode autograd populates mass_param.grad

    # The same gradient via torch.autograd.functional.jacobian: the MPC solve is
    # just another differentiable op, so any output (here the value) can be
    # differentiated with no hand-written sensitivity code.
    j_val = torch.autograd.functional.jacobian(
        lambda m: diff_mpc(x0=x0, params={"mass": m})[4],
        mass_param,
    )

    mo.md(
        f"""
        **Value at m = 1.5:** `{value_m.item():.4f}`

        **∂V/∂mass via `.backward()`:** `{mass_param.grad.item():.4f}`

        **∂V/∂mass via `autograd.functional.jacobian`:** `{j_val.reshape(-1).item():.4f}`

        Both routes agree — the gradient is exact (from the solver's KKT system),
        so the MPC can be dropped straight into a PyTorch training loop.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Policy gradient away from the constraints

    At $x_0 = [0.5,\,0]$ the optimal force rides the $\pm 0.5$ bound, so the
    policy is locally *flat* in the parameters and $\partial F_0^{*}/\partial p$
    is clipped towards zero.

    To see the gradient cleanly we move to a near-origin state
    $x_0 = [0.05,\,0]$, where the force stays strictly inside $(-0.5, 0.5)$, and
    sweep the (square-root) control-cost weight `r_diag_sqrt`. A larger weight
    penalises the force more, so both the policy $F_0^{*}$ and the value $V$
    vary **smoothly** with it.

    We then check the analytic policy gradient at one sweep point: the exact
    solution sensitivity from the solver's KKT system should match both the
    slope of the swept curve and a finite difference.
    """)
    return


@app.cell
def _(diff_mpc, np, torch):
    x0_far = torch.tensor([[0.05, 0.0]])  # near the origin — the force never saturates
    r_sweep = np.linspace(0.2, 1.2, 25)  # 25 <= n_batch_init (64): one batched solve

    # Sweep r_diag_sqrt across the batch in a single call.
    x0_batch = x0_far.repeat(len(r_sweep), 1)
    sweep_params = {"r_diag_sqrt": torch.tensor(r_sweep).reshape(-1, 1)}
    _, u0_sweep, _, _, value_sweep = diff_mpc(x0=x0_batch, params=sweep_params)

    u0_curve = u0_sweep.detach().numpy().ravel()
    v_curve = value_sweep.detach().numpy().ravel()

    # The whole point: the force bound is inactive everywhere on this sweep.
    assert np.abs(u0_curve).max() < 0.5 - 1e-3
    return r_sweep, u0_curve, v_curve, x0_far


@app.cell
def _(diff_mpc, np, plt, r_sweep, torch, u0_curve, v_curve, x0_far):
    r0 = 0.4  # a point mid-sweep, well away from the force bound

    # Analytic gradient via autograd through the solver (exact KKT sensitivity).
    r_param = torch.tensor([[r0]], requires_grad=True)
    ctx0, u0_0, _, _, value_0 = diff_mpc(x0=x0_far, params={"r_diag_sqrt": r_param})
    du0_dr = torch.autograd.grad(u0_0.sum(), r_param, retain_graph=True)[0].item()
    dV_dr = torch.autograd.grad(value_0.sum(), r_param)[0].item()

    # The same number via the explicit solution-sensitivity accessor. du0_dp_global
    # is (B, nu, P) over the flat learnable vector; locate the r_diag_sqrt column
    # from the manager's registration order (no hard-coded index).
    _pm = diff_mpc.parameter_manager
    _offset = 0
    for _name in _pm.learnable_parameter_names:
        if _name == "r_diag_sqrt":
            break
        _offset += _pm.parameters[_name].default.size
    du0_dp = diff_mpc.diff_mpc_fun.sensitivity(ctx0, "du0_dp_global")
    du0_dr_sens = float(du0_dp[0, 0, _offset])
    assert np.isclose(du0_dr, du0_dr_sens, rtol=1e-3, atol=1e-6)

    # Finite-difference cross-check — the ground truth that the gradient is right.
    def _u0_at(r):
        _, u0_r, _, _, _ = diff_mpc(x0=x0_far, params={"r_diag_sqrt": torch.tensor([[r]])})
        return u0_r.item()

    h = 1e-2
    du0_dr_fd = (_u0_at(r0 + h) - _u0_at(r0 - h)) / (2 * h)
    assert np.isclose(du0_dr, du0_dr_fd, rtol=5e-2, atol=1e-3)

    u0_r0 = u0_0.item()
    v_r0 = value_0.item()

    # Plot the swept policy and value, with the analytic tangent overlaid at r0.
    grad_fig, grad_axes = plt.subplots(1, 2, figsize=(11, 4.5))
    tan = np.array([r_sweep.min(), r_sweep.max()])

    grad_axes[0].plot(r_sweep, u0_curve, "-o", markersize=3, label="policy F₀*(r)")
    grad_axes[0].plot(
        tan, u0_r0 + du0_dr * (tan - r0), "--", color="tab:red",
        label=f"tangent  dF₀/dr = {du0_dr:.3f}",
    )
    grad_axes[0].plot([r0], [u0_r0], "s", color="tab:red")
    grad_axes[0].set_ylabel("First action F₀* [N]")
    grad_axes[0].set_title("Policy vs. control weight")

    grad_axes[1].plot(r_sweep, v_curve, "-o", markersize=3, color="tab:green", label="value V(r)")
    grad_axes[1].plot(
        tan, v_r0 + dV_dr * (tan - r0), "--", color="tab:red",
        label=f"tangent  dV/dr = {dV_dr:.3f}",
    )
    grad_axes[1].plot([r0], [v_r0], "s", color="tab:red")
    grad_axes[1].set_ylabel("Value V")
    grad_axes[1].set_title("Value vs. control weight")

    for ax_g in grad_axes:
        ax_g.set_xlabel("r_diag_sqrt")
        ax_g.grid(True, alpha=0.3)
        ax_g.legend()
    grad_fig.suptitle(f"Analytic gradient vs. sweep at x₀ = [0.05, 0], r = {r0}")
    grad_fig.tight_layout()
    grad_fig
    return dV_dr, du0_dr, du0_dr_fd, du0_dr_sens, r0, u0_r0


@app.cell
def _(dV_dr, du0_dr, du0_dr_fd, du0_dr_sens, mo, r0, u0_r0):
    mo.md(f"""
    At `r_diag_sqrt = {r0}` (force `F₀* = {u0_r0:.4f}` N, strictly inside the
    ±0.5 bound) the policy gradient is:

    - **∂F₀/∂r via `.backward()`:** `{du0_dr:.5f}`
    - **∂F₀/∂r via `sensitivity(ctx, "du0_dp_global")`:** `{du0_dr_sens:.5f}`
    - **∂F₀/∂r via finite difference:** `{du0_dr_fd:.5f}`
    - **∂V/∂r via `.backward()`:** `{dV_dr:.5f}`

    All three routes for the policy gradient agree, and the tangent lies flush
    with the swept curve — the sensitivity is exact where no constraint binds.
    Contrast this with the saturated case above, where the force sits on the
    bound and its gradient would collapse to ≈ 0.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
