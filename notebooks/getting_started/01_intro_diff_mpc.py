"""Part 1 — differentiable MPC in five minutes.

The end-to-end loop on the smallest interesting problem: build a tiny
parametric OCP (a mass-spring-damper holding a position reference), wrap it in
``AcadosDiffMpcTorch``, solve it, override the parameter, and backpropagate
through the solver. Everything else in this series is a refinement of what
happens here.
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
    # 01 — differentiable MPC in five minutes

    `AcadosDiffMpcTorch` is a `torch.nn.Module` that wraps a *parametric*
    acados OCP. You hand it a batch of initial states and (optionally) a
    dictionary of parameter values; it solves the OCP and returns the
    optimal plan — **and** lets gradients flow back to the parameters, so an
    MPC can live inside a PyTorch computation graph like any other layer.

    Three steps, three cells:

    1. build the OCP, registering its parameters with
       `AcadosParameterManager`,
    2. wrap it in `AcadosDiffMpcTorch` and solve,
    3. call `.backward()` on the result.
    """)
    return


@app.cell
def _(mo):
    import sys

    sys.path.insert(0, str(mo.notebook_dir().parent))  # notebooks/ root -> nb_utils

    import casadi as ca
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from acados_template import AcadosOcp

    from leap_c.parameters import AcadosParameterManager
    from leap_c.torch import AcadosDiffMpcTorch

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
    ## The problem

    A mass on a spring and a damper, pushed by a force $F$: the state is
    $x = [\,p,\ v\,]$ (position, velocity), and we want the mass held at a
    **position reference** $p_\mathrm{ref}$ — our single parameter. The cost
    penalizes the distance to the reference, the velocity, and the force.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src=str(mo.notebook_dir().parent / "assets" / "mass_spring_damper.svg"),
        width=440,
        caption=(
            "The mass-spring-damper: a block of mass m on a spring "
            "(stiffness k) and a damper (constant c), driven by the force F. "
            "Diagram by Ilmari Karonen, public domain (Wikimedia Commons)."
        ),
    )
    return


@app.cell
def _(AcadosOcp, AcadosParameterManager, ca, np):
    N_HORIZON = 40  # stages 0 .. N_HORIZON
    DT = 0.1  # time step [s]
    MASS, DAMPING, STIFFNESS = 1.0, 0.4, 2.0  # fixed physics

    # 1. Register the parameter: differentiable=True puts it into
    #    model.p_global, so gradients can flow back to it.
    manager = AcadosParameterManager(N_horizon=N_HORIZON)
    position_ref = manager.register_parameter(
        name="position_ref", default=np.array([0.3]), differentiable=True
    )

    # 2. Build the OCP around the returned CasADi symbol.
    ocp = AcadosOcp()
    ocp.model.name = "msd_intro"
    ocp.model.x = ca.vertcat(ca.SX.sym("p"), ca.SX.sym("v"))
    ocp.model.u = ca.SX.sym("F")

    _A = ca.vertcat(
        ca.horzcat(1.0, DT),
        ca.horzcat(-DT * STIFFNESS / MASS, 1.0 - DT * DAMPING / MASS),
    )
    _B = ca.vertcat(0.0, DT / MASS)
    ocp.model.disc_dyn_expr = _A @ ocp.model.x + _B @ ocp.model.u

    # Least-squares cost on [p - p_ref, v, F]; the parameter enters the
    # residual, so the reference is adjustable at solve time.
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x[0] - position_ref, ocp.model.x[1], ocp.model.u)
    ocp.cost.W = np.diag([5.0, 0.2, 0.1])
    ocp.cost.yref = np.zeros(3)
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr_e = ca.vertcat(ocp.model.x[0] - position_ref, ocp.model.x[1])
    ocp.cost.W_e = np.diag([5.0, 0.5])
    ocp.cost.yref_e = np.zeros(2)

    ocp.constraints.x0 = np.array([0.0, 0.0])  # nominal, overwritten per solve
    ocp.constraints.idxbu = np.array([0])  # hard force limits
    ocp.constraints.lbu = np.array([-2.0])
    ocp.constraints.ubu = np.array([2.0])

    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.tf = N_HORIZON * DT
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    # NOTE: we never set ocp.model.p / ocp.model.p_global ourselves — the
    # AcadosDiffMpcTorch constructor calls manager.assign_to_ocp(ocp) and
    # fills them from the registered parameters.
    return DT, manager, ocp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrap and solve

    `AcadosDiffMpcTorch(ocp, manager)` generates and compiles a batched
    acados solver (the first call takes a moment). Calling the module
    returns a 5-tuple `(ctx, u0, x, u, value)`:

    - `ctx` — solver context: status, warm-start iterate, cached
      sensitivities. Pass it back in as `ctx=` to warm-start a later solve,
    - `u0` — first optimal action, shape `(B, nu)`,
    - `x` — planned state trajectory, shape `(B, N+1, nx)`,
    - `u` — planned control trajectory, shape `(B, N, nu)`,
    - `value` — optimal cost, shape `(B, 1)`.

    The batch dimension `B` always comes first — here `B = 1`. With no
    `params`, the registered defaults are used; outputs arrive in the
    module's `dtype` (we pass `torch.float64`).
    """)
    return


@app.cell
def _(AcadosDiffMpcTorch, manager, mo, ocp, torch):
    # n_batch_init sizes the pre-allocated batch solver — 2 is the largest
    # batch in this notebook (notebook 05 has more to say about batching).
    diff_mpc = AcadosDiffMpcTorch(
        ocp, manager, dtype=torch.float64, n_batch_init=2, verbose=False
    )

    x0 = torch.tensor([[0.0, 0.0]], dtype=torch.float64)  # at rest at the origin
    ctx, u0, x_traj, u_traj, value = diff_mpc(x0=x0)

    mo.md(
        f"""
        **Solver status:** `{ctx.status.tolist()}`  (0 = converged)

        **First action F₀:** `{u0.item():.4f}` N — pushing towards the
        default reference `position_ref = 0.3`

        **MPC value V(x₀):** `{value.item():.4f}`

        **Trajectory shapes:** `x {tuple(x_traj.shape)}`, `u {tuple(u_traj.shape)}`
        """
    )
    return diff_mpc, x0, x_traj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Override the parameter

    Parameter values are passed per solve through the `params` dict, keyed
    by the registered name. A global differentiable parameter takes shape
    `(B, dim)` — so two rows solve for two references in one batched call.
    """)
    return


@app.cell
def _(DT, diff_mpc, np, plt, torch, x0, x_traj):
    refs = torch.tensor([[0.3], [0.8]], dtype=torch.float64)
    _, _, x_two, _, _ = diff_mpc(x0=x0.repeat(2, 1), params={"position_ref": refs})

    _t = DT * np.arange(x_traj.shape[1])
    ref_fig, ref_ax = plt.subplots(figsize=(8, 3.4))
    for _i, _r in enumerate(refs[:, 0].tolist()):
        ref_ax.plot(_t, x_two[_i, :, 0].detach().numpy(), label=f"position_ref = {_r:.1f}")
        ref_ax.axhline(_r, ls="--", lw=0.8, color="gray")
    ref_ax.set_xlabel("Time [s]")
    ref_ax.set_ylabel("Position [m]")
    ref_ax.grid(True, alpha=0.3)
    ref_ax.legend()
    ref_fig.suptitle("The same solver, two references — one batched solve")
    ref_fig.tight_layout()
    ref_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Differentiate through the solver

    Pass the parameter as a **leaf tensor** with `requires_grad=True`, call
    `.backward()` on the solver output — the gradient lands on your tensor,
    like for any other torch module. No extra API.
    """)
    return


@app.cell
def _(diff_mpc, mo, torch, x0):
    ref_param = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
    _, _, _, _, value_g = diff_mpc(x0=x0, params={"position_ref": ref_param})
    value_g.backward()

    mo.md(
        f"""
        $\\partial V / \\partial p_\\mathrm{{ref}}$ = `{ref_param.grad.item():.4f}`

        Positive, as it must be: the mass starts at the origin, so moving
        the reference further away makes the optimal cost larger. This
        gradient is **exact** — leap-c computes it from the solver's KKT
        system, not by finite differences.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What just happened

    | step | API |
    |---|---|
    | declare a parameter | `manager.register_parameter(name, default, differentiable=True)` |
    | use it in the OCP | the returned CasADi symbol, directly in cost/dynamics |
    | build the layer | `AcadosDiffMpcTorch(ocp, manager)` |
    | solve | `ctx, u0, x, u, value = diff_mpc(x0=..., params={...})` |
    | differentiate | `value.backward()` → `param.grad` |

    **Next:** `02_from_acados_to_diff_mpc.py` starts from a *plain* acados
    OCP — the situation you are in if you already use acados — and converts
    it, on a house-heating problem that carries through the rest of the
    series.
    """)
    return


if __name__ == "__main__":
    app.run()
