"""Central interface to use acados in PyTorch."""

from pathlib import Path

from acados_template import AcadosOcp
import numpy as np
import torch
import torch.nn as nn

from leap_c.autograd.torch import create_autograd_function
from leap_c.ocp.acados.diff_mpc import (
    AcadosDiffMpcCtx,
    AcadosDiffMpcFunction,
    AcadosDiffMpcSensitivityOptions,
)
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer


class AcadosDiffMpc(nn.Module):
    """
    Acados differentiable MPC interface for PyTorch.

    This module wraps acados solvers to enable their use in differentiable
    machine learning pipelines. It provides an autograd compatible forward
    method and supports sensitivity computation with respect to various inputs.

    Attributes:
        ocp: The acados optimal control problem.
        diff_mpc_fun: The differentiable MPC function wrapper for acados.
        autograd_fun: A PyTorch autograd function created from the MPC function.
    """

    def __init__(
        self,
        ocp: AcadosOcp,
        initializer: AcadosDiffMpcInitializer | None = None,
        sensitivity_ocp: AcadosOcp | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
    ):
        """
        Initializes the AcadosDiffMpc module.

        Args:
            ocp: Optimal control problem formulation used for solving the OCP.
            initializer: Initializer for the OCP solver. If None, solvers
                are initialized with zeros.
            sensitivity_ocp: The optimal control problem formulation to use
                for sensitivities. If None, the sensitivity problem is derived
                from the `ocp` object.
            discount_factor: Discount factor. If None, acados default cost
                scaling is used, i.e., dt for intermediate stages and 1 for the
                terminal stage.
            export_directory: Directory to export the generated code.
        """
        super().__init__()

        self.ocp = ocp

        self.diff_mpc_fun = AcadosDiffMpcFunction(
            ocp=ocp,
            initializer=initializer,
            sensitivity_ocp=sensitivity_ocp,
            discount_factor=discount_factor,
            export_directory=export_directory,
        )
        self.autograd_fun = create_autograd_function(self.diff_mpc_fun)

    def forward(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor | None = None,
        p_global: torch.Tensor | None = None,
        p_stagewise: torch.Tensor | None = None,
        p_stagewise_sparse_idx: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ):
        """
        Performs the forward pass.

        In the background, PyTorch builds a computational graph that can be
        used for backpropagation. The context `ctx` is used to store
        intermediate values required for the backward pass, warmstart the solver
        for subsequent calls and to compute specific sensitivities.

        Args:
            x0: Initial state.
            u0: Initial control input.
            p_global: Global parameters.
            p_stagewise: Stagewise parameters.
            p_stagewise_sparse_idx: Sparse index for stagewise parameters.
            ctx: The context for the forward pass. If None, a new context is created.

        Returns:
            ctx: The updated context after solving the OCP.
            u0: Initial control input.
            x: The computed state trajectory.
            u: The computed control trajectory.
            value: The cost value of the computed trajectory.
        """
        ctx, u0, x, u, value = self.autograd_fun.apply(
            ctx, x0, u0, p_global, p_stagewise, p_stagewise_sparse_idx
        )  # type: ignore

        return ctx, u0, x, u, value

    def sensitivity(
        self, ctx, field_name: AcadosDiffMpcSensitivityOptions
    ) -> np.ndarray:
        """
        Compute the sensitivity of the implicit function with respect to a field.

        Args:
            ctx: Context from the forward pass.
            field_name: The field to compute the sensitivity for.

        Returns:
            The sensitivity of the implicit function with respect to the specified field.
        """
        return self.diff_mpc_fun.sensitivity(ctx, field_name)
