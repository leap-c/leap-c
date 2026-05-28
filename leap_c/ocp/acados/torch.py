"""Central interface to use acados in PyTorch."""

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

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
        ocp: AcadosDiffOcp,
        initializer: AcadosDiffMpcInitializer | None = None,
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
            discount_factor=discount_factor,
            export_directory=export_directory,
            n_batch_init=n_batch_init,
            num_threads_batch_solver=num_threads_batch_solver,
            verbose=verbose,
        )
        self.parameter_manager = ocp.parameter_manager
        self.autograd_fun = create_autograd_function(self.diff_mpc_fun)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

    def forward(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor | None = None,
        params: dict[str, torch.Tensor] | None = None,
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
            params: A dictionary containing the parameters. The keys should correspond to the
                parameter names defined in the acados ocp object, and the value should be a tensor
                of shape `(B, param_dim)` for global parameters or `(B, N_horizon+1, param_dim)` for
                stagewise parameters. You can provide sparse indices for stagewise parameters using
                an additional key with the suffix `"_sparse_idx"` containing a tensor of shape
                `(B, N_horizon+1, n_sparse_idx)`. The module will automatically handle the mapping
                of these parameters to the appropriate inputs for the acados solver.
            p_global: Learnable parameters, i.e., allowing backpropagation, shape
                `(B, p_global_dim)`. Correspond to learnable acados parameters. Deprecated, use
                `params` instead.
            p_stagewise: Acados stagewise parameters, i.e., not allowing backpropagation, shape
                `(B, N_horizon+1, p_stagewise_dim)`. Correspond to `"non-learnable"` acados
                parameters. Deprecated, use `params` instead.
                If `p_stagewise_sparse_idx` is provided, this also has to be provided.
                If `p_stagewise_sparse_idx` is `None`, shape is `(B, N_horizon+1, p_stagewise_dim)`.
                If `p_stagewise_sparse_idx` is provided, shape is
                `(B, N_horizon+1, len(p_stagewise_sparse_idx))`.
            p_stagewise_sparse_idx: Indices for sparsely setting stagewise parameters. Shape is
                `(B, N_horizon+1, n_p_stagewise_sparse_idx)`. Deprecated, use `params` instead.

        Returns:
            ctx: A new context object from solving the problems.
            u0: Solution of initial control input.
            x: The solution of the whole state trajectory.
            u: The solution of the whole control trajectory.
            value: The cost value of the computed trajectory.
        """
        assert (params is None) or (
            p_global is None and p_stagewise is None and p_stagewise_sparse_idx is None
        ), "You cannot provide both `params` and individual parameter tensors. Please use `params`."
        if params is None and (p_global is not None or p_stagewise is not None):
            warnings.warn(
                DeprecationWarning(
                    "Providing individual parameter tensors is deprecated. Please use `params`."
                )
            )
        if params is not None:
            global_params = self.parameter_manager.global_parameter_names
            stagewise_params = self.parameter_manager.stagewise_parameter_names
            assert set(global_params) <= set(params.keys()) and set(stagewise_params) <= set(
                params.keys()
            ), (
                "When providing `params`, it must contain all parameters defined in the ocp object."
                f" Expected global parameters: {global_params},"
                f" stagewise parameters: {stagewise_params}."
            )
            p_global = [params[name] for name in global_params if name in params]
            p_global = torch.cat(p_global, dim=-1) if p_global else None
            p_stagewise = [params[name] for name in stagewise_params if name in params]
            p_stagewise = torch.cat(p_stagewise, dim=-1) if p_stagewise else None
            p_stagewise_sparse_idx = [
                params[name + "_sparse_idx"]
                for name in stagewise_params
                if name + "_sparse_idx" in params
            ]
            p_stagewise_sparse_idx = (
                torch.cat(p_stagewise_sparse_idx, dim=-1) if p_stagewise_sparse_idx else None
            )

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

    def sensitivity(self, ctx, field_name: AcadosDiffMpcSensitivityOptions) -> np.ndarray:
        """Retrieves a specific sensitivity field from the context object.

        Args:
            ctx: The ctx object generated by the forward pass.
            field_name: The name of the sensitivity field to retrieve.
        """
        return self.diff_mpc_fun.sensitivity(ctx, field_name)
