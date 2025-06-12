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


class AcadosImplicitLayer(nn.Module):
    def __init__(
        self,
        ocp: AcadosOcp,
        initializer: AcadosInitializer | None = None,
        sensitivity_ocp: AcadosOcp | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
    ):
        """
        Initialize the Acados implicit layer.

        Args:
            ocp: Optimal control problem formulation used for solving the OCP.
            initializer: Initializer for the OCP solver. If None, a zero initializer is used.
            sensitivity_ocp: The optimal control problem formulation to use for sensitivities.
                If None, the sensitivity problem is derived from the ocp object, however only the EXTERNAL cost type is allowed then.
                For an example of how to set up other cost types refer, e.g., to examples/pendulum_on_cart.py .
            discount_factor: Discount factor. If None, acados default cost scaling is used, i.e. dt for intermediate stages, 1 for terminal stage.
            export_directory: Directory to export the generated code.
        """
        super().__init__()

        self.ocp = ocp

        self.implicit_fun = AcadosImplicitFunction(
            ocp=ocp,
            initializer=initializer,
            sensitivity_ocp=sensitivity_ocp,
            discount_factor=discount_factor,
            export_directory=export_directory,
        )
        self.autograd_function = create_autograd_function(self.implicit_fun)

    def forward(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor | None = None,
        p_global: torch.Tensor | None = None,
        p_stagewise: torch.Tensor | None = None,
        p_stagewise_sparse_idx: torch.Tensor | None = None,
        ctx: AcadosImplicitCtx | None = None,
    ):
        """
        Perform the forward pass of the implicit function.

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
        ctx, u0, x, u, value = self.autograd_function.apply(
            ctx, x0, u0, p_global, p_stagewise, p_stagewise_sparse_idx
        )

        return ctx, u0, x, u, value

    def sensitivity(self, ctx, field_name: SensitivityField):
        """
        Compute the sensitivity of the implicit function with respect to a field.

        Args:
            ctx: Context from the forward pass.
            field_name: The field to compute sensitivity for.

        Returns:
            The sensitivity of the implicit function with respect to the specified field.
        """
        return self.implicit_fun.sensitivity(ctx, field_name)
