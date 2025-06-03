"""Central interface to use Acados in PyTorch."""

from pathlib import Path

from acados_template import AcadosOcp
import torch
import torch.nn as nn

from leap_c.autograd.torch import create_autograd_function
from leap_c.ocp.acados.implicit import (
    AcadosImplicitCtx,
    AcadosImplicitFunction,
    SensitivityField,
)
from leap_c.ocp.acados.initializer import AcadosInitializer


N_BATCH_MAX = 256
NUM_THREADS_BATCH_SOLVER = 4


class AcadosImplicitLayer(nn.Module):
    def __init__(
        self,
        ocp: AcadosOcp,
        initializer: AcadosInitializer,
        ocp_sensitivity: AcadosOcp | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
    ):
        # TODO: This is the central interface for people to create stuff.
        super().__init__()

        self.ocp = ocp

        self.implicit_fun = AcadosImplicitFunction(
            batch_solver=batch_solver,
            sensitivity_batch_solver=sensitivity_batch_solver,
            initializer=initializer,
        )
        self.autograd_function = create_autograd_function()

    def forward(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor | None = None,
        p_global: torch.Tensor | None = None,
        p_stagewise: torch.Tensor | None = None,
        p_stagewise_sparse_idx: torch.Tensor | None = None,
        ctx: AcadosImplicitCtx | None = None,
    ):
        """Performs the forward pass of the implicit function.

        Args:
            x0: Initial state.
            u0: Initial control input.
            p_global: Global parameters.
            p_stagewise: Stagewise parameters.
            p_stagewise_sparse_idx: Sparse index for stagewise parameters.
            ctx: Context for the implicit function.

        Returns:
            A tuple containing the context and the output of the implicit function.
        """
        return self.autograd_function.apply(
            x0, u0, p_global, p_stagewise, p_stagewise_sparse_idx, ctx
        )

    def sensitivity(self, ctx, field_name: SensitivityField):
        """Computes the sensitivity of the implicit function with respect to a field.

        Args:
            ctx: Context from the forward pass.
            field_name: The field to compute sensitivity for.

        Returns:
            The sensitivity of the implicit function with respect to the specified field.
        """
        return self.implicit_fun.sensitivity(ctx, field_name)
