"""Provides a differentiable implicit function based on Acados."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from acados_template import AcadosOcp
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate
import numpy as np

from leap_c.autograd.function import DiffFunction
from leap_c.ocp.acados.data import AcadosSolverInput
from leap_c.ocp.acados.initializer import AcadosInitializer
from leap_c.ocp.acados.utils.prepare_solver import prepare_batch_solver_for_backward
from leap_c.ocp.acados.utils.solve import solve_with_retry


N_BATCH_MAX = 256
NUM_THREADS_BATCH_SOLVER = 4


@dataclass
class AcadosImplicitCtx:
    iterate: AcadosOcpFlattenedBatchIterate
    status: np.ndarray
    log: dict[str, float]
    solver_input: AcadosSolverInput

    # backward pass
    needs_input_grad: list[bool] | None = None

    # sensitivity fields
    du0_dp_global: np.ndarray | None = None
    du0_dx0: np.ndarray | None = None
    dvalue_du: np.ndarray | None = None
    dvalue_dx0: np.ndarray | None = None
    dx_dp_global: np.ndarray | None = None
    du_dp_global: np.ndarray | None = None


SensitivityField = Literal[
    "du0_dp_global",
    "dx_dp_global",
    "du_dp_global",
    "dvalue_dp_global",
    "dvalue_du0",
]


class AcadosImplicitFunction(DiffFunction):
    """Function for differentiable implicit function based on Acados."""

    def __init__(
        self,
        ocp: AcadosOcp,
        initializer: AcadosInitializer | None = None,
        ocp_sensitivity: AcadosOcp | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
    ):
        pass

    @property
    def N(self) -> int:
        return self.ocp.solver_options.N_horizon  # type: ignore

    def forward(  # type: ignore
        self,
        x0: np.ndarray,
        u0: np.ndarray | None = None,
        p_global: np.ndarray | None = None,
        p_stagewise: np.ndarray | None = None,
        p_stagewise_sparse_idx: np.ndarray | None = None,
        ctx: AcadosImplicitCtx | None = None,
    ):
        """Performs the forward pass of the implicit function.

        Args:
            x0: Initial states with shape `(B, x_dim)`.
            u0: Initial actions with shape `(B, u_dim)`. Defaults to `None`.
            p_global: Global parameters shared across all stages,
                shape `(B, p_global_dim)`. Defaults to `None`.
            p_stagewise: Stagewise parameters.
                If `p_stagewise_sparse_idx` is `None`, shape is
                `(B, N+1, p_stagewise_dim)`.
                If `p_stagewise_sparse_idx` is provided, shape is
                `(B, N+1, len(p_stagewise_sparse_idx))`.
                For multi-phase MPC with `p_stagewise_sparse_idx`, this is
                a list of arrays (one per phase). Defaults to `None`.
            p_stagewise_sparse_idx: Indices for sparsely setting stagewise
                parameters. Shape is `(B, N+1, n_p_stagewise_sparse_idx)`.
                For multi-phase MPC, this is a list of arrays (one per phase).
                Defaults to `None`.
            ctx: An `AcadosCtx` object for storing context. Defaults to `None`.
        """
        batch_size = x0.shape[0]

        solver_input = AcadosSolverInput(
            x0=x0,
            u0=u0,
            p_global=p_global,
            p_stagewise=p_stagewise,
            p_stagewise_sparse_idx=p_stagewise_sparse_idx,
        )
        ocp_iterate = None if ctx is None else ctx.iterate

        status, log = solve_with_retry(
            self.batch_solver,
            initializer=self.initializer,
            ocp_iterate=ocp_iterate,
            solver_input=solver_input,
        )

        # fetch output
        active_solvers = self.batch_solver.ocp_solvers[:batch_size]
        sol_iterate = self.batch_solver.store_iterate_to_flat_obj(n_batch=batch_size)
        ctx = AcadosImplicitCtx(
            iterate=sol_iterate, log=log, status=status, solver_input=solver_input
        )
        sol_value = np.array([s.get_cost() for s in active_solvers])
        sol_u0 = sol_iterate.u[0, :]

        return ctx, sol_u0, sol_iterate.x, sol_iterate.u, sol_value

    def backward(  # type: ignore
        self,
        ctx: AcadosImplicitCtx,
        u0_grad: np.ndarray | None,
        x_grad: np.ndarray | None,
        u_grad: np.ndarray | None,
        value_grad: np.ndarray | None,
    ):
        """Computes gradients of inputs given gradients of outputs.

        Args:
            ctx: The `AcadosCtx` object from the forward pass.
            p_global_grad: Gradient with respect to `p_global`.
            p_stagewise_idx_grad: Gradient with respect to
                `p_stagewise_sparse_idx`.
            p_stagewise_grad: Gradient with respect to `p_stagewise`.
        """
        if ctx.needs_input_grad is None:
            return (None, None, None, None, None, None, None)

        def _back(output_grad, field_name: SensitivityField):
            if output_grad is None:
                return None
            return self.sensitivity(ctx, field_name) @ output_grad

        def _safe_sum(*args):
            filtered_args = [arg for arg in args if arg is not None]
            if not filtered_args:
                return None
            return np.sum(filtered_args, axis=0)

        if ctx.needs_input_grad[0]:
            raise NotImplementedError

        if ctx.needs_input_grad[1]:
            # Use adjoint sensitivity for here :D
            grad_u0 = _back(value_grad, "dvalue_du0")
        else:
            grad_u0 = None

        if ctx.needs_input_grad[2]:
            grad_p_global = _safe_sum(
                _back(value_grad, "dvalue_dp_global"),
                _back(u0_grad, "du0_dp_global"),
                _back(x_grad, "dx_dp_global"),
                _back(u_grad, "du_dp_global"),
            )
        else:
            grad_p_global = None

        return (None, None, grad_u0, grad_p_global, None, None, None)

    def sensitivity(self, ctx, field_name: SensitivityField) -> np.ndarray:
        """Calculates a specific sensitivity field for a context.

        The `sensitivity` method retrieves a specific sensitivity field from the
        context object, or recalculates it if not already present.

        Args:
            ctx: The `AcadosCtx` object containing sensitivity information.
            field_name: The name of the sensitivity field to retrieve.
        """
        # check if already calculated
        if getattr(ctx, field_name) is not None:
            return getattr(ctx, field_name)

        prepare_batch_solver_for_backward(
            self.batch_solver, ctx.iterate, ctx.solver_input
        )

        sens = None
        batch_size = ctx.solver_input.batch_size
        ocp = self.batch_solver.ocp_solvers[0].ocp  # type: ignore

        if field_name == "du0_dp_global":
            single_seed = np.eye(ocp.dims.nu)
            seed_vec = np.repeat(single_seed[np.newaxis, :, :], batch_size, axis=0)
            sens = self.sensitivity_batch_solver.eval_adjoint_solution_sensitivity(
                seed_x=[],
                seed_u=[(0, seed_vec)],
                with_respect_to="p_global",
                sanity_checks=True,
            )
        elif field_name == "dx_dp_global":
            
        elif field_name == "du_dp_global":
            raise NotImplementedError
        elif field_name == "dvalue_dp_global":
            raise NotImplementedError
        elif field_name == "dvalue_du0":
            raise NotImplementedError

        setattr(ctx, field_name, sens)

        return sens
