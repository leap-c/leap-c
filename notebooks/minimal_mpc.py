"""Minimal leap-c MPC example.

This marimo notebook builds a tiny scalar integrator MPC, solves it with
``AcadosDiffMpcTorch``, and shows how to collate contexts for batched warm starts.
"""

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Minimal MPC with batched warm starts

        This notebook uses the smallest useful MPC example: a scalar integrator
        
        \[
        x_{k+1} = x_k + \Delta t u_k
        \]
        
        with a box constraint on `u`. The target position is a differentiable
        parameter, so gradients can flow from the MPC value back to `x_ref`.

        The final cells show the user-facing collation story: if you store
        `AcadosDiffMpcCtx` objects for warm starts, `collate_torch` lets PyTorch
        collate normal tensors while leap-c handles the MPC context.
        """
    )
    return


@app.cell
def _():
    import casadi as ca
    import numpy as np
    import torch
    from acados_template import AcadosOcp

    from leap_c.parameters import AcadosParameterManager
    from leap_c.torch import AcadosDiffMpcTorch
    from leap_c.utils import collate_torch

    return AcadosDiffMpcTorch, AcadosOcp, AcadosParameterManager, ca, collate_torch, np, torch


@app.cell
def _(AcadosOcp, AcadosParameterManager, ca, np):
    def create_integrator_ocp() -> tuple[AcadosOcp, AcadosParameterManager]:
        """Create a scalar integrator OCP and its parameter manager."""
        n_horizon = 20
        dt = 0.1

        ocp = AcadosOcp()
        ocp.model.name = "minimal_integrator"
        ocp.model.x = ca.SX.sym("x", 1)
        ocp.model.u = ca.SX.sym("u", 1)
        ocp.model.disc_dyn_expr = ocp.model.x + dt * ocp.model.u

        manager = AcadosParameterManager(N_horizon=n_horizon)
        x_ref = manager.register_parameter("x_ref", np.array([0.0]), differentiable=True)
        u_ref = manager.register_parameter("u_ref", np.array([0.0]), differentiable=True)

        ocp.cost.cost_type_0 = "NONLINEAR_LS"
        ocp.model.cost_y_expr_0 = ca.vertcat(ocp.model.x - x_ref, ocp.model.u - u_ref)
        ocp.cost.yref_0 = np.zeros(2)
        ocp.cost.W_0 = np.diag([10.0, 0.1])

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.x - x_ref, ocp.model.u - u_ref)
        ocp.cost.yref = np.zeros(2)
        ocp.cost.W = np.diag([10.0, 0.1])

        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr_e = ocp.model.x - x_ref
        ocp.cost.yref_e = np.zeros(1)
        ocp.cost.W_e = np.diag([10.0])

        ocp.constraints.x0 = np.array([0.0])
        ocp.constraints.lbu = np.array([-2.0])
        ocp.constraints.ubu = np.array([2.0])
        ocp.constraints.idxbu = np.array([0])

        ocp.solver_options.N_horizon = n_horizon
        ocp.solver_options.tf = n_horizon * dt
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_ric_alg = 1
        ocp.solver_options.with_value_sens_wrt_params = True
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.with_batch_functionality = True

        return ocp, manager

    return (create_integrator_ocp,)


@app.cell
def _(AcadosDiffMpcTorch, create_integrator_ocp, torch):
    class MinimalMpcPlanner:
        """Tiny convenience wrapper around the leap-c torch layer."""

        def __init__(self) -> None:
            ocp, manager = create_integrator_ocp()
            self.layer = AcadosDiffMpcTorch(ocp=ocp, parameter_manager=manager)

        def solve(self, x0: torch.Tensor, x_ref: torch.Tensor, ctx=None):
            """Solve a batch of scalar integrator MPC problems."""
            x0 = x0.reshape(-1, 1)
            x_ref = x_ref.reshape(x0.shape[0], 1).to(device=x0.device, dtype=x0.dtype)
            u_ref = torch.zeros_like(x_ref)
            return self.layer(x0=x0, params={"x_ref": x_ref, "u_ref": u_ref}, ctx=ctx)

    planner = MinimalMpcPlanner()
    return MinimalMpcPlanner, planner


@app.cell
def _(mo, planner, torch):
    x0 = torch.tensor([[2.0]], dtype=torch.float64)
    x_ref = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)

    ctx, u0, x_traj, u_traj, value = planner.solve(x0=x0, x_ref=x_ref)
    value.sum().backward()

    mo.md(
        f"""
        ## Single solve

        Solver status: `{ctx.status.tolist()}`

        First control: `{u0.detach().flatten().tolist()}`

        Value: `{value.item():.4f}`

        Gradient wrt target `x_ref`: `{x_ref.grad.flatten().tolist()}`
        """
    )
    return ctx, u0, value, x0, x_ref, x_traj


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Collating contexts for batched warm starts

        A warm start context is not a tensor, so PyTorch's default collation does
        not know how to stack it. `collate_torch` keeps PyTorch's default behavior
        for tensors and only adds the custom rule for `AcadosDiffMpcCtx`.
        """
    )
    return


@app.cell
def _(collate_torch, mo, planner, torch):
    samples = []
    for initial_state in [2.0, -1.5, 0.5]:
        sample_x0 = torch.tensor([initial_state], dtype=torch.float64)
        sample_x_ref = torch.tensor([0.0], dtype=torch.float64)
        sample_ctx, *_ = planner.solve(x0=sample_x0, x_ref=sample_x_ref)
        samples.append({"ctx": sample_ctx, "x0": sample_x0, "x_ref": sample_x_ref})

    batch = collate_torch(samples)

    mo.md(
        f"""
        Collated `x0` shape: `{tuple(batch["x0"].shape)}`

        Collated context batch size: `{batch["ctx"].iterate.N_batch}`

        Collated solver statuses: `{batch["ctx"].status.tolist()}`
        """
    )
    return batch, samples


@app.cell
def _(batch, mo, planner):
    # Explicit tensor movement stays visible to users.  If you need CUDA, move
    # tensor leaves before calling the layer; the context itself stays numpy/acados.
    x0_batch = batch["x0"]
    x_ref_batch = batch["x_ref"]
    warmstart_ctx = batch["ctx"]

    ctx2, u0_batch, x_batch, u_batch, value_batch = planner.solve(
        x0=x0_batch,
        x_ref=x_ref_batch,
        ctx=warmstart_ctx,
    )

    mo.md(
        f"""
        ## Batched warm-started solve

        Solver statuses: `{ctx2.status.tolist()}`

        First controls: `{u0_batch.detach().flatten().tolist()}`

        Values: `{value_batch.detach().flatten().tolist()}`
        """
    )
    return ctx2, u0_batch, value_batch


if __name__ == "__main__":
    app.run()
