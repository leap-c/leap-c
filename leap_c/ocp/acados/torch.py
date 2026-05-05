"""Central interface to use acados in PyTorch."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from acados_template import AcadosOcp

from leap_c.autograd.torch import create_autograd_function
from leap_c.ocp.acados.diff_mpc import (
    AcadosDiffMpcCtx,
    AcadosDiffMpcFunction,
    AcadosDiffMpcSensitivityOptions,
)
from leap_c.ocp.acados.diff_ocp import AcadosDiffOcp
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer


class AcadosDiffMpcTorch(torch.nn.Module):
    """PyTorch module for differentiable MPC based on acados.

    This module wraps acados solvers to enable their use in differentiable machine learning
    pipelines. It provides an autograd compatible forward method and supports sensitivity
    computation with respect to various inputs (see `AcadosDiffMpcCtx`).


    Attributes:
        diff_mpc_fun: The differentiable MPC function wrapper for acados.
        autograd_fun: A PyTorch autograd function created from `diff_mpc_fun`.
    """

    diff_mpc_fun: AcadosDiffMpcFunction
    autograd_fun: type[torch.autograd.Function]

    def __init__(
        self,
        ocp: AcadosOcp | AcadosDiffOcp,
        initializer: AcadosDiffMpcInitializer | None = None,
        sensitivity_ocp: AcadosOcp | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
        n_batch_init: int | None = None,
        num_threads_batch_solver: int | None = None,
        dtype: torch.dtype | None = None,
        verbose: bool = True,
    ) -> None:
        """Initializes the AcadosDiffMpcTorch module.

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
            dtype: The output of the forward pass will automatically be cast to this type.
                If `None`, the default PyTorch dtype is used.
            verbose: Whether to print the output while generating solvers.

        """
        super().__init__()
        self.diff_mpc_fun = AcadosDiffMpcFunction(
            ocp=ocp,
            initializer=initializer,
            sensitivity_ocp=sensitivity_ocp,
            discount_factor=discount_factor,
            export_directory=export_directory,
            n_batch_init=n_batch_init,
            num_threads_batch_solver=num_threads_batch_solver,
            verbose=verbose,
        )
        self.autograd_fun = create_autograd_function(self.diff_mpc_fun)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

    def forward(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor | None = None,
        p_global: torch.Tensor | None = None,
        p_stagewise: torch.Tensor | None = None,
        p_stagewise_sparse_idx: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the forward pass by solving the provided problem instances.

        In the background, PyTorch builds a computational graph that can be used for
        backpropagation. The context `ctx` is used to store intermediate values required for the
        backward pass, warmstart the solver for subsequent calls and to compute specific
        sensitivities.

        Args:
            ctx: An object for storing context. If provided, it will be used to warmstart the solve
                (e.g., by using the saved iterate). Defaults to `None`.
            x0: Initial states with shape `(B, x_dim)`.
            u0: Initial actions with shape `(B, u_dim)`. Defaults to `None`.
            p_global: Learnable parameters, i.e., allowing backpropagation, shape
                `(B, p_global_dim)`. Correspond to learnable acados parameters. If none is provided,
                the default values set in the acados ocp object are used.
            p_stagewise: Acados stagewise parameters, i.e., not allowing backpropagation, shape
                `(B, N_horizon+1, p_stagewise_dim)`. Correspond to `"non-learnable"` acados
                parameters.
                If none is provided, the default values set in the acados ocp object are used.
                If `p_stagewise_sparse_idx` is provided, this also has to be provided.
                If `p_stagewise_sparse_idx` is `None`, shape is `(B, N_horizon+1, p_stagewise_dim)`.
                If `p_stagewise_sparse_idx` is provided, shape is
                `(B, N_horizon+1, len(p_stagewise_sparse_idx))`.
            p_stagewise_sparse_idx: Indices for sparsely setting stagewise parameters. Shape is
                `(B, N_horizon+1, n_p_stagewise_sparse_idx)`.

        Returns:
            ctx: A new context object from solving the problems.
            u0: Solution of initial control input.
            x: The solution of the whole state trajectory.
            u: The solution of the whole control trajectory.
            value: The cost value of the computed trajectory.
        """
        ctx, u_star, x, u, value = self.autograd_fun.apply(
            ctx,
            x0,
            u0,
            p_global,  # type:ignore
            p_stagewise,
            p_stagewise_sparse_idx,
        )
        u_star = u_star.to(dtype=self.dtype)
        x = x.to(dtype=self.dtype)
        u = u.to(dtype=self.dtype)
        value = value.to(dtype=self.dtype)
        return ctx, u_star, x, u, value  # type:ignore

    def set_constraint_bounds(
        self,
        lbx: np.ndarray,
        ubx: np.ndarray,
        stages: Sequence[int],
    ) -> None:
        """Set lbx/ubx constraints on the forward solvers for the given stages.

        Args:
            lbx: Lower bounds, shape ``(batch_size, len(stages), constraint_dim)``.
            ubx: Upper bounds, shape ``(batch_size, len(stages), constraint_dim)``.
            stages: Stage indices at which to apply the bounds.
        """
        solvers = self.diff_mpc_fun.forward_batch_solver.ocp_solvers
        batch_size = lbx.shape[0]
        for i in range(batch_size):
            solver = solvers[i]
            for j, stage in enumerate(stages):
                solver.constraints_set(stage, "lbx", lbx[i, j])
                solver.constraints_set(stage, "ubx", ubx[i, j])

    def get_solver_stats(self, ctx: AcadosDiffMpcCtx) -> list[dict]:
        """Collect per-instance solver statistics from the most recent solve.

        Queries every field exposed by :meth:`AcadosOcpSolver.get_stats` for
        each solver instance in the batch.  Fields that are unavailable for the
        current solver configuration are silently skipped.

        The batch-level log (``ctx.log``) keys ``solving_time``,
        ``success_rate``, and ``retry_rate`` are included under the special key
        ``"_batch"`` in the first element of the returned list.

        Args:
            ctx: Context returned by the most recent :meth:`forward` call.

        Returns:
            A list of dicts, one per batch instance, each containing all
            retrievable stat fields plus ``"status"``.
        """
        _SCALAR_FIELDS = [
            "time_tot",
            "time_lin",
            "time_sim",
            "time_sim_ad",
            "time_sim_la",
            "time_qp",
            "time_qp_solver_call",
            "time_qp_xcond",
            "time_qpscaling",
            "time_glob",
            "time_solution_sensitivities",
            "time_reg",
            "time_preparation",
            "time_feedback",
            "nlp_iter",
            "sqp_iter",
            "stat_m",
            "stat_n",
            "qpscaling_status",
        ]
        _ARRAY_FIELDS = ["residuals", "qp_iter", "qp_stat", "alpha", "statistics"]

        batch_size = len(ctx.status)
        solvers = self.diff_mpc_fun.forward_batch_solver.ocp_solvers
        results = []
        for i in range(batch_size):
            solver = solvers[i]
            entry: dict = {"status": int(ctx.status[i])}
            for field in _SCALAR_FIELDS + _ARRAY_FIELDS:
                try:
                    entry[field] = solver.get_stats(field)
                except Exception:
                    pass
            results.append(entry)
        return results

    def sensitivity(self, ctx, field_name: AcadosDiffMpcSensitivityOptions) -> np.ndarray:
        """Retrieves a specific sensitivity field from the context object.

        Args:
            ctx: The ctx object generated by the forward pass.
            field_name: The name of the sensitivity field to retrieve.
        """
        return self.diff_mpc_fun.sensitivity(ctx, field_name)
