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

DEFAULT_N_BATCH_MAX = 256
DEFAULT_NUM_THREADS_BATCH_SOLVER = 4


@dataclass
class AcadosDiffMpcCtx:
    """Context for differentiable MPC with acados.

    This context holds the results of the forward pass. This information is needed for the backward
    pass and to calculate the sensitivities. It also contains fields for caching the sensitivity
    calculations.

    Attributes:
        iterate: The solution iterate from the forward pass. Can be used for, e.g., initializing the
            next solve.
        status: The status of the solver after the forward pass. 0 indicates success, non-zero
            values indicate various errors.
        log: Statistics from the forward solve containing info like success rates and timings.
        solver_input: The input used for the forward pass.
        needs_input_grad: A list of booleans indicating which inputs require gradients.
        du0_dp_global: Sensitivity of the control solution of the initial stage with respect to
            acados global parameters (i.e., learnable parameters).
        du0_dx0: Sensitivity of the control solution of the initial stage with respect to the
            initial state.
        dvalue_du0: Sensitivity of the objective value solution with respect to the control input of
            the first stage. Only available if said control was provided.
        dvalue_dx0: Sensitivity of the objective value solution solution with respect to the initial
            state.
        dx_dp_global: Sensitivity of the whole state trajectory solution with respect to acados
            global parameters (i.e., learnable parameters).
        du_dp_global: Sensitivity of the whole control trajectory solution with respect to acados
            global parameters (i.e., learnable parameters).
        dvalue_dp_global: Sensitivity of the objective value solution with respect to acados global.
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
        iterate=collate_acados_flattened_batch_iterate_fn([ctx.iterate for ctx in batch]),
        log=None,
        status=np.array([ctx.status for ctx in batch]),
        solver_input=collate_acados_ocp_solver_input([ctx.solver_input for ctx in batch]),
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
AcadosDiffMpcSensitivityOptions.__doc__ = """For an explanation, please refer to the corresponding
fields in `AcadosDiffMpcCtx`."""


class AcadosDiffMpcFunction(DiffFunction):
    """Differentiable MPC function based on acados.

    Attributes:
        ocp: The acados ocp object defining the optimal control problem structure.
        forward_batch_solver: The acados batch solver used for the forward pass.
        backward_batch_solver: The acados batch solver used for the backward pass.
        initializer: The initializer used to provide initial guesses for the solver, if none are
            provided explicitly or on a retry. Uses a zero iterate by default.
    """

    def __init__(
        self,
        ocp: AcadosOcp,
        initializer: AcadosDiffMpcInitializer | None = None,
        sensitivity_ocp: AcadosOcp | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
        n_batch_max: int | None = None,
        num_threads_batch_solver: int | None = None,
    ) -> None:
        """Initializes the differentiable MPC function.

        Args:
            ocp: The acados ocp object defining the optimal control problem structure.
            initializer: The initializer used to provide initial guesses for the solver, if none are
                provided explicitly or on a retry. Uses a zero iterate by default.
            sensitivity_ocp: An optional acados ocp object for obtaining the sensitivities.
                If none is provided, the sensitivity ocp will be derived from the given "normal"
                `ocp`.
            discount_factor: An optional discount factor for the sensitivity problem.
                If none is provided, the default acados weighting will be used, i.e., `1/N_horizon`
                on the stage cost and `1` on the terminal cost.
            export_directory: An optional directory to which the generated C code will be exported.
                If none is provided, a unique temporary directory will be created used.
            n_batch_max: Maximum batch size supported by the batch OCP solver.
                If `None`, a default value is used.
            num_threads_batch_solver: Number of parallel threads to use for the batch OCP solver.
                If `None`, a default value is used.
        """
        self.ocp = ocp
        self.forward_batch_solver, self.backward_batch_solver = (
            create_forward_backward_batch_solvers(
                ocp=ocp,
                sensitivity_ocp=sensitivity_ocp,
                discount_factor=discount_factor,
                export_directory=export_directory,
                n_batch_max=DEFAULT_N_BATCH_MAX if n_batch_max is None else n_batch_max,
                num_threads=DEFAULT_NUM_THREADS_BATCH_SOLVER
                if num_threads_batch_solver is None
                else num_threads_batch_solver,
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
        """Perform the forward pass by solving the problem instances.

        Args:
            ctx: A context object for the forward pass. If provided, it will be used to warmstart
                the solve (e.g., by using the saved iterate).
            x0: Initial states with shape `(B, x_dim)`.
            u0: Initial actions with shape `(B, u_dim)`. Defaults to `None`.
            p_global: Acados global parameters shared across all stages
                (i.e., learnable parameters), shape `(B, p_global_dim)`. If none is provided, the
                default values set in the acados ocp object are used.
            p_stagewise: Stagewise parameters.
                If none is provided, the default values set in the acados ocp object are used.
                If `p_stagewise_sparse_idx` is provided, this also has to be provided.
                If `p_stagewise_sparse_idx` is `None`, shape is `(B, N_horizon+1, p_stagewise_dim)`.
                If `p_stagewise_sparse_idx` is provided, shape is
                `(B, N_horizon+1, len(p_stagewise_sparse_idx))`.
            p_stagewise_sparse_idx: Indices for sparsely setting stagewise parameters. Shape is
                `(B, N_horizon+1, n_p_stagewise_sparse_idx)`.
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
        sol_iterate = self.forward_batch_solver.store_iterate_to_flat_obj(n_batch=batch_size)
        ctx = AcadosDiffMpcCtx(
            iterate=sol_iterate, log=log, status=status, solver_input=solver_input
        )
        sol_value = np.array([[s.get_cost()] for s in active_solvers])
        sol_u0 = sol_iterate.u[:, : self.ocp.dims.nu]

        x = sol_iterate.x.reshape(batch_size, self.ocp.solver_options.N_horizon + 1, -1)  # type: ignore
        u = sol_iterate.u.reshape(batch_size, self.ocp.solver_options.N_horizon, -1)  # type: ignore

        return ctx, sol_u0, x, u, sol_value

    def backward(  # type: ignore
        self,
        ctx: AcadosDiffMpcCtx,
        u0_grad: np.ndarray | None,
        x_grad: np.ndarray | None,
        u_grad: np.ndarray | None,
        value_grad: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, None, None]:
        """Perform the backward pass via implicit differentiation.

        Args:
            ctx: The ctx object from the forward pass.
            u0_grad: Gradient with respect to the control solution of the first stage.
            x_grad: Gradient with respect to the whole state trajectory solution.
            u_grad: Gradient with respect to the whole control trajectory solution.
            value_grad: Gradient with respect to the objective value solution.
        """
        if ctx.needs_input_grad is None:
            return None, None, None, None, None

        prepare_batch_solver_for_backward(self.backward_batch_solver, ctx.iterate, ctx.solver_input)

        def _adjoint(x_seed, u_seed, with_respect_to: str):
            # backpropagation via the adjoint operator
            if x_seed is None and u_seed is None:
                return None

            # check if x_seed and u_seed are all zeros
            dx_zero = np.all(x_seed == 0) if x_seed is not None else True
            du_zero = np.all(u_seed == 0) if u_seed is not None else True
            if dx_zero and du_zero:
                return None

            if x_seed is not None and not dx_zero:
                # Sum over batch dim and state dim to know which stages to seed
                take_stages = np.abs(x_seed).sum(axis=(0, 2)) > 0
                x_seed_with_stage = [
                    (stage_idx, x_seed[:, stage_idx][..., None])
                    for stage_idx in range(0, self.ocp.solver_options.N_horizon + 1)  # type: ignore
                    if take_stages[stage_idx]
                ]
            else:
                x_seed_with_stage = []

            if u_seed is not None and not du_zero:
                # Sum over batch dim and control dim to know which stages to seed
                take_stages = np.abs(u_seed).sum(axis=(0, 2)) > 0
                u_seed_with_stage = [
                    (stage_idx, u_seed[:, stage_idx][..., None])
                    for stage_idx in range(self.ocp.solver_options.N_horizon)  # type: ignore
                    if take_stages[stage_idx]
                ]
            else:
                u_seed_with_stage = []

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

            subscripts = "bj,b->bj" if output_grad.ndim == 1 else "bij,bi->bj"
            return np.einsum(subscripts, self.sensitivity(ctx, field_name), output_grad)

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
        """Retrieves a specific sensitivity field from the context object.

        Recalculates the sensitivity if not already present.

        Args:
            ctx: The ctx object generated by the forward pass.
            field_name: The name of the sensitivity field to retrieve.

        Returns:
            The requested sensitivity as a numpy array.

        Raises:
            ValueError: If `field_name` is not recognized.
        """
        # check if already calculated
        if getattr(ctx, field_name) is not None:
            return getattr(ctx, field_name)

        prepare_batch_solver_for_backward(self.backward_batch_solver, ctx.iterate, ctx.solver_input)

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
            seed_x = [
                (stage_idx, seed_vec) for stage_idx in range(self.ocp.solver_options.N_horizon + 1)
            ]  # type: ignore
            sens = self.backward_batch_solver.eval_adjoint_solution_sensitivity(
                seed_x=seed_x,
                seed_u=[],
                with_respect_to="p_global",
                sanity_checks=True,
            )
        elif field_name == "du_dp_global":
            single_seed = np.eye(self.ocp.dims.nu)  # type: ignore
            seed_vec = np.repeat(single_seed[None, :, :], batch_size, axis=0)
            seed_u = [
                (stage_idx, seed_vec) for stage_idx in range(self.ocp.solver_options.N_horizon)
            ]  # type: ignore
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
        elif field_name in ("dvalue_dp_global", "dvalue_dx0", "dvalue_du0"):
            match field_name:
                case "dvalue_dp_global":
                    with_respect_to = "p_global"
                case "dvalue_dx0":
                    with_respect_to = "initial_state"
                case "dvalue_du0":
                    with_respect_to = "initial_control"
                case _:
                    raise ValueError(f"Unexpected `field_name` {field_name} encountered.")
            sens = np.array(
                [[s.eval_and_get_optimal_value_gradient(with_respect_to)] for s in active_solvers]
            )
        else:
            raise ValueError

        setattr(ctx, field_name, sens)

        return sens
