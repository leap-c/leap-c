"""Central interface to use acados in PyTorch."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from acados_template import AcadosOcp

from leap_c.autograd.torch import create_autograd_function
from leap_c.ocp.acados.diff_mpc import (
    AcadosDiffMpcCtx,
    AcadosDiffMpcFunction,
    AcadosDiffMpcSensitivityOptions,
)
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer


class AcadosDiffMpc(nn.Module):
    """
    PyTorch module for differentiable MPC based on acados.

    This module wraps acados solvers to enable their use in differentiable
    machine learning pipelines. It provides an autograd compatible forward
    method and supports sensitivity computation with respect to various inputs
    (see `AcadosDiffMpcCtx`).


    Attributes:
        ocp: The acados optimal control problem.
        diff_mpc_fun: The differentiable MPC function wrapper for acados.
        autograd_fun: A PyTorch autograd function created from `diff_mpc_fun`.
    """

    ocp: AcadosOcp
    diff_mpc_fun: AcadosDiffMpcFunction
    autograd_fun: type[torch.autograd.Function]

    def __init__(
        self,
        ocp: AcadosOcp,
        initializer: AcadosDiffMpcInitializer | None = None,
        sensitivity_ocp: AcadosOcp | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
        *,
        n_batch_max: int | None = None,
        num_threads_batch_solver: int | None = None,
    ) -> None:
        """
        Initializes the AcadosDiffMpc module.

        Args:
            ocp: The acados ocp object defining the optimal control problem structure.
            initializer: The initializer used to provide initial guesses for the solver,
                if none are provided explicitly or on a retry. Uses a zero iterate by default.
            sensitivity_ocp: An optional acados ocp object for obtaining the sensitivities.
                If none provided, the sensitivity ocp will be derived from the given "normal" `ocp`.
            discount_factor: An optional discount factor for the sensitivity problem.
                If none provided, the default acados weighting will be used, i.e.,
                1/N_horizon on the stage cost and 1 on the terminal cost.
            export_directory: An optional directory to which the generated C code will be exported.
                If none provided, a unique temporary directory will be created used.
            n_batch_max: Maximum batch size supported by the batch ocp solver. If
                `None`, the default value is used.
            num_threads_batch_solver: Number of parallel threads to use for the batch
                ocp solver. If `None`, the default value is used.
        """
        super().__init__()

        self.ocp = ocp

        self.diff_mpc_fun = AcadosDiffMpcFunction(
            ocp=ocp,
            initializer=initializer,
            sensitivity_ocp=sensitivity_ocp,
            discount_factor=discount_factor,
            export_directory=export_directory,
            n_batch_max=n_batch_max,
            num_threads_batch_solver=num_threads_batch_solver,
        )
        self.autograd_fun = create_autograd_function(self.diff_mpc_fun)

    def forward(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor | None = None,
        p_learnable: torch.Tensor | None = None,
        p_non_learnable: torch.Tensor | None = None,
        p_non_learnable_sparse_idx: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ):
        """
        Performs the forward pass by solving the provided problem instances.

        In the background, PyTorch builds a computational graph that can be
        used for backpropagation. The context `ctx` is used to store
        intermediate values required for the backward pass, warmstart the solver
        for subsequent calls and to compute specific sensitivities.

        Args:
            ctx: An object for storing context. If provided,
                it will be used to warmstart the solve (e.g., by using the saved iterate).
                Defaults to `None`.
            x0: Initial states with shape `(B, x_dim)`.
            u0: Initial actions with shape `(B, u_dim)`. Defaults to `None`.
            p_learnable: Learnable parameters, i.e., allowing backpropagation,
                shape `(B, p_learnable_dim)`. Correspond to acados p_global. If none provided,
                the default values set in the acados ocp object are used.
            p_non_learnable: Non-learnable parameters, i.e., not allowing backpropagation,
                shape `(B, N_horizon+1, p_non_learnable_dim)`.
                Correspond to "normal" acados parameters.
                If none provided, the default values set in the acados ocp object are used.
                If p_stagewise_sparse_idx is provided, this also has to be provided.
                If `p_stagewise_sparse_idx` is `None`, shape is
                `(B, N_horizon+1, p_stagewise_dim)`.
                If `p_stagewise_sparse_idx` is provided, shape is
                `(B, N_horizon+1, len(p_stagewise_sparse_idx))`.
            p_non_learnable_sparse_idx: Indices for sparsely setting stagewise
                parameters. Shape is `(B, N_horizon+1, n_p_stagewise_sparse_idx)`.

        Returns:
            ctx: A new context object from solving the problems.
            u0: Solution of initial control input.
            x: The solution of the whole state trajectory.
            u: The solution of the whole control trajectory.
            value: The cost value of the computed trajectory.
        """
        ctx, u0, x, u, value = self.autograd_fun.apply(
            ctx, x0, u0, p_learnable, p_non_learnable, p_non_learnable_sparse_idx
        )  # type: ignore

        return ctx, u0, x, u, value

    def sensitivity(self, ctx, field_name: AcadosDiffMpcSensitivityOptions) -> np.ndarray:
        """
        Retrieves a specific sensitivity field from the
        context object, or recalculates it if not already present.

        Args:
            ctx: The ctx object generated by the forward pass.
            field_name: The name of the sensitivity field to retrieve.
        """
        return self.diff_mpc_fun.sensitivity(ctx, field_name)
