"""Provides an implemenation of differentiable MPC based on acados."""

from dataclasses import dataclass
from functools import cache
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

DEFAULT_N_BATCH_INIT = 256
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
    needs_input_grad: tuple[bool] | None = None

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


TO_ACADOS_SOLVER_GRADOPTS: dict[str, str] = {
    "dvalue_dp_global": "p_global",
    "dvalue_dx0": "initial_state",
    "dvalue_du0": "initial_control",
}


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
        n_batch_init: int | None = None,
        num_threads_batch_solver: int | None = None,
        verbose: bool = True,
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
            n_batch_init: Initially supported batch size of the batch OCP solver.
                Using larger batches will trigger a delay for the creation of more solvers.
                If `None`, a default value is used.
            num_threads_batch_solver: Number of parallel threads to use for the batch OCP solver.
                If `None`, a default value is used.
            verbose: Whether to print the output while generating solvers.

        """
        self.ocp = ocp
        self.forward_batch_solver, self.backward_batch_solver = (
            create_forward_backward_batch_solvers(
                ocp=ocp,
                sensitivity_ocp=sensitivity_ocp,
                discount_factor=discount_factor,
                export_directory=export_directory,
                n_batch_init=DEFAULT_N_BATCH_INIT if n_batch_init is None else n_batch_init,
                num_threads=DEFAULT_NUM_THREADS_BATCH_SOLVER
                if num_threads_batch_solver is None
                else num_threads_batch_solver,
                verbose=verbose,
            )
        )
        self.initializer = ZeroDiffMpcInitializer(ocp) if initializer is None else initializer

        # these flags allow to run the sanity checks only once the first time a specific sensitivity
        # is requested, and then skip them for subsequent calls
        self._run_sanity_checks_in_du0_dp_global = True
        self._run_sanity_checks_in_dx_dp_global = True
        self._run_sanity_checks_in_du_dp_global = True
        self._run_sanity_checks_in_du0_dx0 = True

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

        Returns:
            A tuple containing:
            - ctx: The context object containing information from the forward pass.
            - sol_u0: The control solution of the first stage, shape `(B, u_dim)`.
            - x: The state trajectory solution, shape `(B, N_horizon + 1, x_dim)`.
            - u: The control trajectory solution, shape `(B, N_horizon, u_dim)`.
            - sol_value: The objective value solution, shape `(B, 1)`.
        """
        batch_size = x0.shape[0]

        solver_input = AcadosOcpSolverInput(x0, u0, p_global, p_stagewise, p_stagewise_sparse_idx)
        ocp_iterate = None if ctx is None else ctx.iterate

        status, log = solve_with_retry(
            self.forward_batch_solver, self.initializer, ocp_iterate, solver_input
        )

        # fetch output
        active_solvers = self.forward_batch_solver.ocp_solvers[:batch_size]
        sol_iterate = self.forward_batch_solver.get_flat_iterate(batch_size)
        ctx = AcadosDiffMpcCtx(sol_iterate, status, log, solver_input)
        sol_value = np.array([[s.get_cost()] for s in active_solvers])
        sol_u0 = sol_iterate.u[:, : self.ocp.dims.nu]

        N = self.ocp.solver_options.N_horizon
        x = sol_iterate.x.reshape(batch_size, N + 1, -1)  # type: ignore
        u = sol_iterate.u.reshape(batch_size, N, -1)  # type: ignore

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

        Returns:
            A tuple containing the gradients with respect to the inputs in the following order:
            - grad_x0: Gradient with respect to the initial state.
            - grad_u0: Gradient with respect to the initial control.
            - grad_p_global: Gradient with respect to the acados global parameters.
            - grad_p_stagewise: Always `None` (not supported for differentiation).
            - grad_p_stagewise_sparse_idx: Always `None` (not supported for differentiation).
        """
        if ctx.needs_input_grad is None:
            return None, None, None, None, None

        prepare_batch_solver_for_backward(self.backward_batch_solver, ctx.iterate, ctx.solver_input)

        needs_grad_x0, needs_grad_u0, needs_grad_p_global = ctx.needs_input_grad[1:4]
        grad_x0 = (
            self._safe_sum(
                self._jacobian(ctx, value_grad, "dvalue_dx0"),
                self._jacobian(ctx, u0_grad, "du0_dx0"),
            )
            if needs_grad_x0
            else None
        )
        grad_u0 = self._jacobian(ctx, value_grad, "dvalue_du0") if needs_grad_u0 else None
        grad_p_global = (
            self._safe_sum(
                self._jacobian(ctx, value_grad, "dvalue_dp_global"),
                self._jacobian(ctx, u0_grad, "du0_dp_global"),
                self._adjoint(x_grad, u_grad, "p_global"),
            )
            if needs_grad_p_global
            else None
        )
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

        match field_name:
            case "du0_dp_global":
                seed_u0 = self._get_seed_seq(1, self.ocp.dims.nu, batch_size)
                sens = self.backward_batch_solver.eval_adjoint_solution_sensitivity(
                    [], seed_u0, "p_global", self._run_sanity_checks_in_du0_dp_global
                )
                self._run_sanity_checks_in_du0_dp_global = False

            case "dx_dp_global":
                seed_x = self._get_seed_seq(
                    self.ocp.solver_options.N_horizon + 1, self.ocp.dims.nx, batch_size
                )
                sens = self.backward_batch_solver.eval_adjoint_solution_sensitivity(
                    seed_x, [], "p_global", self._run_sanity_checks_in_dx_dp_global
                )
                self._run_sanity_checks_in_dx_dp_global = False

            case "du_dp_global":
                seed_u = self._get_seed_seq(
                    self.ocp.solver_options.N_horizon, self.ocp.dims.nu, batch_size
                )
                sens = self.backward_batch_solver.eval_adjoint_solution_sensitivity(
                    [], seed_u, "p_global", self._run_sanity_checks_in_du_dp_global
                )
                self._run_sanity_checks_in_du_dp_global = False

            case "du0_dx0":
                sens = np.array(
                    [
                        s.eval_solution_sensitivity(
                            0,
                            "initial_state",
                            False,
                            sanity_checks=self._run_sanity_checks_in_du0_dx0,
                        )["sens_u"]
                        for s in active_solvers
                    ]
                )
                self._run_sanity_checks_in_du0_dx0 = False

            case "dvalue_dp_global" | "dvalue_dx0" | "dvalue_du0":
                with_respect_to = TO_ACADOS_SOLVER_GRADOPTS[field_name]
                sens = np.array(
                    [
                        [s.eval_and_get_optimal_value_gradient(with_respect_to)]
                        for s in active_solvers
                    ]
                )

            case _:
                raise ValueError(f"Unexpected `field_name` {field_name} encountered.")

        setattr(ctx, field_name, sens)
        return sens

    def _adjoint(
        self, x_seed: np.ndarray | None, u_seed: np.ndarray | None, with_respect_to: str
    ) -> np.ndarray | None:
        """Compute the adjoint sensitivity via backpropagation."""
        # backpropagation via the adjoint operator
        x_is_none = x_seed is None
        u_is_none = u_seed is None
        if x_is_none and u_is_none:
            return None

        # check if x_seed and u_seed are all zeros
        x_is_zero = x_is_none or not x_seed.any()
        u_is_zero = u_is_none or not u_seed.any()
        if x_is_zero and u_is_zero:
            return None

        if x_is_none or x_is_zero:
            x_seed_with_stage = []
        else:
            # Sum over batch dim and state dim to know which stages to seed
            (nonzero_stages,) = np.abs(x_seed).sum((0, 2)).nonzero()
            x_seed_with_stage = [(int(i), x_seed[:, i][..., None]) for i in nonzero_stages]

        if u_is_none or u_is_zero:
            u_seed_with_stage = []
        else:
            # Sum over batch dim and control dim to know which stages to seed
            (nonzero_stages,) = np.abs(u_seed).sum((0, 2)).nonzero()
            u_seed_with_stage = [(int(i), u_seed[:, i][..., None]) for i in nonzero_stages]

        return self.backward_batch_solver.eval_adjoint_solution_sensitivity(
            x_seed_with_stage, u_seed_with_stage, with_respect_to, True
        )[:, 0]

    def _jacobian(
        self,
        ctx: AcadosDiffMpcCtx,
        output_grad: np.ndarray | None,
        field_name: AcadosDiffMpcSensitivityOptions,
    ) -> np.ndarray | None:
        """Compute the jacobian."""
        if output_grad is None or not output_grad.any():
            return None

        subscripts = "bj,b->bj" if output_grad.ndim == 1 else "bij,bi->bj"
        return np.einsum(subscripts, self.sensitivity(ctx, field_name), output_grad)

    @staticmethod
    def _safe_sum(*args: np.ndarray | None) -> np.ndarray | None:
        """Sum the given arrays, ignoring any that are `None`."""
        filtered_args = [a for a in args if a is not None]
        if not filtered_args:
            return None
        return np.sum(filtered_args, 0)

    @staticmethod
    @cache
    def _get_seed_seq(stages: int, n: int, batch_size: int) -> list[tuple[int, np.ndarray]]:
        """Create the list of stages and `seed_vec` for state/action sensitivity.

        The shape of a single `seed_vec` is `(batch_size, n, n)`, and the list has length `stages`.
        """
        single_seed = np.eye(n)
        seed_vec = np.lib.stride_tricks.as_strided(
            single_seed, (batch_size, n, n), (0, *single_seed.strides), writeable=False
        )  # equivalent to `np.repeat(single_seed[None, :, :], batch_size, 0)` but without copy
        return [(stage, seed_vec) for stage in range(stages)]
