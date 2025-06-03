"""Provides a differentiable implicit function based on Acados."""

from dataclasses import dataclass

from typing import Literal
from acados_template.acados_ocp_batch_solver import AcadosOcpBatchSolver
import numpy as np

from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate
from leap_c.ocp.acados.initializer import AcadosInitializer
from leap_c.ocp.acados.data import AcadosSolverInput
from leap_c.ocp.acados.utils.solve import solve_with_retry
from leap_c.ocp.acados.utils.prepare_solver import prepare_batch_solver

from leap_c.autograd.function import DiffFunction


@dataclass
class AcadosImplicitCtx:
    iterate: AcadosOcpFlattenedBatchIterate
    status: np.ndarray
    log: dict[str, float]
    solver_input: AcadosSolverInput

    # sensitivity fields
    du0_dp_global: np.ndarray | None = None
    du0_dx0: np.ndarray | None = None
    dvalue_du: np.ndarray | None = None
    dvalue_dx0: np.ndarray | None = None
    dx_dp_global: np.ndarray | None = None
    du_dp_global: np.ndarray | None = None


SensitivityField = Literal[
    "du0_dp_global",
    "du0_dx0",
    "dvalue_du",
    "dvalue_dx0",
    "dx_dp_global",
    "du_dp_global",
]


class AcadosImplicitFunction(DiffFunction):
    """Function for differentiable implicit function based on Acados."""

    def __init__(
        self,
        batch_solver: AcadosOcpBatchSolver,
        initializer: AcadosInitializer,
    ):
        """
        Initialize the MPC object.

        Args:
            batch_solver: The batch solver to build the implicit function from.
            initializer: The initializer used when no ctx is provided or we retry
                to solve a sample.
        """
        self.batch_solver = batch_solver
        self.initializer = initializer

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
            x0_grad: Gradient with respect to `x0`.
            u0_grad: Gradient with respect to `u0`.
            p_global_grad: Gradient with respect to `p_global`.
            p_stagewise_idx_grad: Gradient with respect to
                `p_stagewise_sparse_idx`.
            p_stagewise_grad: Gradient with respect to `p_stagewise`.
        """
        raise NotImplementedError

    def sensitvity(self, ctx, field_name: SensitivityField):
        """Calculates a specific sensitivity field for a context.

        The `sensitivity` method retrieves a specific sensitivity field from the
        context object, or recalculates it if not already present.

        Args:
            ctx: The `AcadosCtx` object containing sensitivity information.
            field_name: The name of the sensitivity field to retrieve.
        """
        raise NotImplementedError
