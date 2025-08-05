"""Provides an implemenation of differentiable MPC based on acados."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from acados_template import AcadosOcp
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate

from leap_c.autograd.function import DiffFunction
from leap_c.ocp.acados.data import (
    AcadosOcpSolverInput,
    collate_acados_flattened_batch_iterate_fn,
    collate_acados_ocp_solver_input,
)
from leap_c.ocp.acados.initializer import (
    AcadosDiffMpcInitializer,
    ZeroDiffMpcInitializer,
)
from leap_c.ocp.acados.utils.create_solver import create_forward_backward_batch_solvers
from leap_c.ocp.acados.utils.prepare_solver import prepare_batch_solver_for_backward
from leap_c.ocp.acados.utils.solve import solve_with_retry


N_BATCH_MAX = 256
NUM_THREADS_BATCH_SOLVER = 4


@dataclass
class AcadosDiffMpcCtx:
    """Context for differentiable MPC with acados.

    This context holds the results of the forward pass, including the solution
    iterate, solver status, log, and solver input. This information is needed
    for the backward pass and to calculate the sensitivities. It also contains
    fields for caching the sensitivity calculations.
    """

    iterate: AcadosOcpFlattenedBatchIterate
    status: np.ndarray
    log: dict[str, float] | None
    solver_input: AcadosOcpSolverInput

    # backward pass
    needs_input_grad: list[bool] | None = None

    # sensitivity fields
    du0_dp_global: np.ndarray | None = None
    du0_dx0: np.ndarray | None = None
    dvalue_du0: np.ndarray | None = None
    dvalue_dx0: np.ndarray | None = None
    dx_dp_global: np.ndarray | None = None
    du_dp_global: np.ndarray | None = None
    dvalue_dp_global: np.ndarray | None = None


def collate_acados_diff_mpc_ctx(
    batch: Sequence[AcadosDiffMpcCtx],
    collate_fn_map: dict[str, Callable] | None = None,
) -> AcadosDiffMpcCtx:
    """Collates a batch of AcadosDiffMpcCtx objects into a single object."""
    return AcadosDiffMpcCtx(
        iterate=collate_acados_flattened_batch_iterate_fn(
            [ctx.iterate for ctx in batch]
        ),
        log=None,
        status=np.array([ctx.status for ctx in batch]),
        solver_input=collate_acados_ocp_solver_input(
            [ctx.solver_input for ctx in batch]
        ),
    )


AcadosDiffMpcSensitivityOptions = Literal[
    "du0_dp_global",
    "du0_dx0",
    "dx_dp_global",
    "du_dp_global",
    "dvalue_dp_global",
    "dvalue_du0",
    "dvalue_dx0",
]


class AcadosDiffMpcFunction(DiffFunction):
    """Differentiable MPC based on acados."""

    def __init__(
        self,
        ocp: AcadosOcp,
        initializer: AcadosDiffMpcInitializer | None = None,
        sensitivity_ocp: AcadosOcp | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
    ) -> None:
        self.ocp = ocp
        self.forward_batch_solver, self.backward_batch_solver = (
            create_forward_backward_batch_solvers(
                ocp=ocp,
                sensitivity_ocp=sensitivity_ocp,
                discount_factor=discount_factor,
                export_directory=export_directory,
                n_batch_max=N_BATCH_MAX,
                num_threads=NUM_THREADS_BATCH_SOLVER,
            )
        )

        if initializer is None:
            self.initializer = ZeroDiffMpcInitializer(ocp)
        else:
            self.initializer = initializer

    def forward(  # type: ignore
        self,
        ctx: AcadosDiffMpcCtx | None,
        x0: np.ndarray,
        u0: np.ndarray | None = None,
        p_global: np.ndarray | None = None,
        p_stagewise: np.ndarray | None = None,
        p_stagewise_sparse_idx: np.ndarray | None = None,
    ) -> tuple[AcadosDiffMpcCtx, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the forward pass by solving the OCP.

        Args:
            ctx: An `AcadosDiffMpcCtx` object for storing context. Defaults to `None`.
            x0: Initial states with shape `(B, x_dim)`.
            u0: Initial actions with shape `(B, u_dim)`. Defaults to `None`.
            p_global: Global parameters shared across all stages,
                shape `(B, p_global_dim)`. Defaults to `None`.
            p_stagewise: Stagewise parameters.
                If `p_stagewise_sparse_idx` is `None`, shape is
                `(B, N+1, p_stagewise_dim)`.
                If `p_stagewise_sparse_idx` is provided, shape is
                `(B, N+1, len(p_stagewise_sparse_idx))`.
            p_stagewise_sparse_idx: Indices for sparsely setting stagewise
                parameters. Shape is `(B, N+1, n_p_stagewise_sparse_idx)`.
        """
        batch_size = x0.shape[0]

        solver_input = AcadosOcpSolverInput(
            x0=x0,
            u0=u0,
            p_global=p_global,
            p_stagewise=p_stagewise,
            p_stagewise_sparse_idx=p_stagewise_sparse_idx,
        )
        ocp_iterate = None if ctx is None else ctx.iterate

        status, log = solve_with_retry(
            self.forward_batch_solver,
            initializer=self.initializer,
            ocp_iterate=ocp_iterate,
            solver_input=solver_input,
        )

        # fetch output
        active_solvers = self.forward_batch_solver.ocp_solvers[:batch_size]
        sol_iterate = self.forward_batch_solver.store_iterate_to_flat_obj(
            n_batch=batch_size
        )
        ctx = AcadosDiffMpcCtx(
            iterate=sol_iterate, log=log, status=status, solver_input=solver_input
        )
        sol_value = np.array([s.get_cost() for s in active_solvers])
        sol_u0 = sol_iterate.u[:, : self.ocp.dims.nu]

        x = sol_iterate.x.reshape(batch_size, self.ocp.dims.N + 1, -1)  # type: ignore
        u = sol_iterate.u.reshape(batch_size, self.ocp.dims.N, -1)  # type: ignore

        return ctx, sol_u0, x, u, sol_value

    def backward(  # type: ignore
        self,
        ctx: AcadosDiffMpcCtx,
        u0_grad: np.ndarray | None,
        x_grad: np.ndarray | None,
        u_grad: np.ndarray | None,
        value_grad: np.ndarray | None,
    ):
        """
        Perform the backward pass via implicit differentiation.

        Args:
            ctx: The `AcadosDiffMpcCtx` object from the forward pass.
            p_global_grad: Gradient with respect to `p_global`.
            p_stagewise_idx_grad: Gradient with respect to
                `p_stagewise_sparse_idx`.
            p_stagewise_grad: Gradient with respect to `p_stagewise`.
        """
        if ctx.needs_input_grad is None:
            return None, None, None, None, None

        prepare_batch_solver_for_backward(
            self.backward_batch_solver, ctx.iterate, ctx.solver_input
        )

        def _adjoint(x_seed, u_seed, with_respect_to: str):
            # backpropagation via the adjoint operator
            if x_seed is None and u_seed is None:
                return None

            # check if x_seed and u_seed are all zeros
            # TODO (Jasper): Optimize this such that we also
            #   filter out individual stages
            dx_zero = np.all(x_seed == 0) if x_seed is not None else True
            du_zero = np.all(u_seed == 0) if u_seed is not None else True
            if dx_zero and du_zero:
                return None

            x_seed_with_stage = (
                [
                    (stage_idx, x_seed[:, stage_idx][..., None])
                    for stage_idx in range(0, self.ocp.dims.N + 1)  # type: ignore
                ]
                if x_seed is not None and not dx_zero
                else []
            )

            u_seed_with_stage = (
                [
                    (stage_idx, u_seed[:, stage_idx][..., None])
                    for stage_idx in range(self.ocp.dims.N)  # type: ignore
                ]
                if u_seed is not None and not du_zero
                else []
            )
            grad = self.backward_batch_solver.eval_adjoint_solution_sensitivity(
                seed_x=x_seed_with_stage,
                seed_u=u_seed_with_stage,
                with_respect_to=with_respect_to,
                sanity_checks=True,
            )[:, 0]

            return grad

        def _jacobian(output_grad, field_name: AcadosDiffMpcSensitivityOptions):
            if output_grad is None or np.all(output_grad == 0):
                return None

            if output_grad.ndim == 1:
                return np.einsum(
                    "bj,b->bj", self.sensitivity(ctx, field_name), output_grad
                )

            return np.einsum(
                "bij,bi->bj", self.sensitivity(ctx, field_name), output_grad
            )

        def _safe_sum(*args):
            filtered_args = [a for a in args if a is not None]
            if not filtered_args:
                return None
            return np.sum(filtered_args, axis=0)

        if ctx.needs_input_grad[1]:
            grad_x0 = _safe_sum(
                _jacobian(value_grad, "dvalue_dx0"),
                _jacobian(u0_grad, "du0_dx0"),
            )
        else:
            grad_x0 = None

        if ctx.needs_input_grad[2]:
            grad_u0 = _jacobian(value_grad, "dvalue_du0")
        else:
            grad_u0 = None

        if ctx.needs_input_grad[3]:
            grad_p_global = _safe_sum(
                _jacobian(value_grad, "dvalue_dp_global"),
                _jacobian(u0_grad, "du0_dp_global"),
                _adjoint(x_grad, u_grad, with_respect_to="p_global"),
            )
        else:
            grad_p_global = None

        return grad_x0, grad_u0, grad_p_global, None, None

    def sensitivity(
        self, ctx: AcadosDiffMpcCtx, field_name: AcadosDiffMpcSensitivityOptions
    ) -> np.ndarray:
        """
        Calculate a specific sensitivity field for a context.

        The `sensitivity` method retrieves a specific sensitivity field from the
        context object, or recalculates it if not already present.

        Args:
            ctx: The ctx object generated by the forward pass.
            field_name: The name of the sensitivity field to retrieve.
        """
        # check if already calculated
        if getattr(ctx, field_name) is not None:
            return getattr(ctx, field_name)

        prepare_batch_solver_for_backward(
            self.backward_batch_solver, ctx.iterate, ctx.solver_input
        )

        sens = None
        batch_size = ctx.solver_input.batch_size
        active_solvers = self.backward_batch_solver.ocp_solvers[:batch_size]

        if field_name == "du0_dp_global":
            single_seed = np.eye(self.ocp.dims.nu)  # type: ignore
            seed_vec = np.repeat(single_seed[None, :, :], batch_size, axis=0)
            sens = self.backward_batch_solver.eval_adjoint_solution_sensitivity(
                seed_x=[],
                seed_u=[(0, seed_vec)],
                with_respect_to="p_global",
                sanity_checks=True,
            )
        elif field_name == "dx_dp_global":
            single_seed = np.eye(self.ocp.dims.nx)  # type: ignore
            seed_vec = np.repeat(single_seed[None, :, :], batch_size, axis=0)
            seed_x = [(stage_idx, seed_vec) for stage_idx in range(self.ocp.dims.N + 1)]  # type: ignore
            sens = self.backward_batch_solver.eval_adjoint_solution_sensitivity(
                seed_x=seed_x,
                seed_u=[],
                with_respect_to="p_global",
                sanity_checks=True,
            )
        elif field_name == "du_dp_global":
            single_seed = np.eye(self.ocp.dims.nu)  # type: ignore
            seed_vec = np.repeat(single_seed[None, :, :], batch_size, axis=0)
            seed_u = [(stage_idx, seed_vec) for stage_idx in range(self.ocp.dims.N)]  # type: ignore
            sens = self.backward_batch_solver.eval_adjoint_solution_sensitivity(
                seed_x=[],
                seed_u=seed_u,
                with_respect_to="p_global",
                sanity_checks=True,
            )
        elif field_name == "du0_dx0":
            sens = np.array(
                [
                    s.eval_solution_sensitivity(
                        stages=0,
                        with_respect_to="initial_state",
                        return_sens_u=True,
                        return_sens_x=False,
                    )["sens_u"]
                    for s in active_solvers
                ]
            )
        elif field_name in ["dvalue_dp_global", "dvalue_dx0", "dvalue_du0"]:
            with_respect_to = {
                "dvalue_dp_global": "p_global",
                "dvalue_dx0": "initial_state",
                "dvalue_du0": "initial_control",
            }[field_name]
            sens = np.array(
                [
                    s.eval_and_get_optimal_value_gradient(with_respect_to)
                    for s in active_solvers
                ]
            )
        else:
            raise ValueError

        setattr(ctx, field_name, sens)

        return sens
