import math
from typing import Any

import casadi as ca
import torch
import torch.nn as nn
from acados_template import AcadosSimSolver

from leap_c.mpc import Mpc, MpcBatchedState, MpcInput, MpcOutput

from .autograd import AutogradCasadiFunction, DynamicsSimFunction, MPCSolutionFunction


class CasadiExprModule(nn.Module):
    def __init__(self, expr: ca.SX, inputs: list[ca.SX]):
        super().__init__()
        self.expr = expr
        self.inputs = inputs
        self.f = ca.Function("f", inputs, [expr])

        # setup backward AD
        inputs_cat = ca.vertcat(*inputs)
        grad_output = ca.SX.sym("output", expr.shape[0])
        jtimes = ca.jtimes(expr, inputs_cat, grad_output, True)
        self.jtimes = ca.Function("jtimes", [inputs_cat, grad_output], [jtimes])

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        assert len(inputs) == len(self.inputs)
        return AutogradCasadiFunction.apply(self.f, self.jtimes, *inputs)


class AcadosSimModule(nn.Module):
    """A PyTorch module to wrap around AcadosSimSolver."""

    def __init__(self, sim: AcadosSimSolver):
        super().__init__()
        self.sim = sim

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, p: torch.Tensor | None = None
    ) -> torch.Tensor:
        return DynamicsSimFunction.apply(self.sim, x, u, p)


class MpcSolutionModule(nn.Module):
    """A PyTorch module to allow differentiating the solution given by an MPC planner,
    with respect to some inputs.

    Backpropagation works by using the sensitivities
    given by the MPC. If differentiation with respect to parameters is desired, they must
    be declared as global over the horizon (contrary to stagewise parameters).

    NOTE: Make sure that you follow the documentation of AcadosOcpSolver.eval_solution_sensitivity
        and AcadosOcpSolver.eval_and_get_optimal_value_gradient
        or else the gradients used in the backpropagation might be erroneous!

    NOTE: The status output can be used to rid yourself from non-converged solutions, e.g., by using the
        CleanseAndReducePerSampleLoss module.

    Attributes:
        mpc: The MPC object to use.
    """

    def __init__(self, mpc: Mpc):
        super().__init__()
        self.mpc = mpc

    def forward(
        self,
        mpc_input: MpcInput,
        mpc_state: MpcBatchedState | None = None,
    ) -> tuple[MpcOutput, MpcBatchedState, dict[str, Any]]:
        """Differentiation is only allowed with respect to x0, u0 and p_global.

        Args:
            mpc_input: The MPCInput object containing the input that should be set.
                NOTE x0, u0 and p_global must be tensors, if not None.
            mpc_state: The MPCBatchedState object containing the initializations for the solver.

        Returns:
            mpc_output: An MPCOutput object containing tensors of u0, value (or Q, if u0 was given) and status of the solution.
            mpc_state: The MPCBatchedState containing the iterates of the solution.
            stats: A dictionary containing statistics from the MPC evaluation.
        """
        if mpc_input.parameters is None:
            p_glob = None
            p_rest = None
        else:
            p_glob = mpc_input.parameters.p_global
            p_rest = mpc_input.parameters._replace(p_global=None)

        u0, value, status, state = MPCSolutionFunction.apply(  # type:ignore
            self.mpc,
            mpc_input.x0,
            mpc_input.u0,
            p_glob,
            p_rest,
            mpc_state,
        )

        if mpc_input.u0 is None:
            V = value
            Q = None
        else:
            Q = value
            V = None

        return (
            MpcOutput(u0=u0, Q=Q, V=V, status=status),
            state,
            self.mpc.last_call_stats,
        )


class CleanseAndReducePerSampleLoss(nn.Module):
    """A module that is ment to substitute the last part of a loss,
    taking over the reduction to a scalar and, in particular,
    cleansing the per-sample-loss of the samples whose MPC-part did not converge.
    This basically works by removing all samples that correspond to a status unequal to 0,
    hence effectively training with a varying batch_size.
    """

    def __init__(
        self,
        reduction: str,
        num_batch_dimensions: int,
        n_nonconvergences_allowed: int,
        throw_exception_if_exceeded: bool = False,
    ):
        """
        Parameters:
            reduction: Either "mean", "sum" or "none" as in native pytorch modules.
            num_batch_dimensions: Determines how many of the first dimensions should be treated as batch_dimension.
            n_nonconvergences_allowed: The number of nonconvergences allowed in a batch. If this number is exceeded, the loss returned will just be 0 (and hence do nothing when backward is called).
            throw_exception_if_exceeded: If True, an exception will be thrown if the number of nonconvergences is exceeded.
        """
        super().__init__()
        self.reduction = reduction
        self.n_nonconvergences_allowed = n_nonconvergences_allowed
        self.throw_exception_if_exceeded = throw_exception_if_exceeded
        self.num_batch_dimensions = num_batch_dimensions

    def forward(
        self, per_sample_loss: torch.Tensor, status: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Parameters:
            per_sample_loss: The per_sample_loss.
            status: The status of the MPC solution, of same shape as the per_sample loss for the first how_many_batch_dimensions and afterwards one integer dimension, containing whether the solution converged (0 means converged, all other integers count as not converged).

        Returns:
            The cleansed loss and a dictionary containing

                nonconvergent_samples: The number of nonconvergent samples

                invalid_loss: Whether the number of nonconvergences exceeded the allowed number.
        """
        if (
            per_sample_loss.shape[: self.num_batch_dimensions]
            != status.shape[: self.num_batch_dimensions]
        ):
            raise ValueError(
                "The per_sample_loss and status must have the same batch dimensions."
            )
        if len(status.shape) != self.num_batch_dimensions + 1:
            raise ValueError(
                "The status must have a shape corresponding to the number of batch dimensions, followed by one dimension containing the status."
            )
        if status.shape[-1] != 1:
            raise ValueError(
                "The last dimension of status must be of size 1, containing the status of each sample."
            )

        stats = dict()
        error_mask = status.to(dtype=torch.bool)
        nonconvergent_samples = torch.sum(error_mask).item()
        stats["nonconvergent_samples"] = nonconvergent_samples

        if nonconvergent_samples > self.n_nonconvergences_allowed:
            if self.throw_exception_if_exceeded:
                raise ValueError(
                    f"Number of nonconvergences exceeded the allowed number of {self.n_nonconvergences_allowed}."
                )
            stats["invalid_loss"] = False
            return torch.zeros(1), stats
        stats["invalid_loss"] = True

        cleansed_loss = per_sample_loss.clone()
        cleansed_loss[error_mask.squeeze()] = 0
        if self.reduction == "mean":
            cleansed_loss = torch.mean(cleansed_loss)
            batch_size = math.prod(per_sample_loss.shape[: self.num_batch_dimensions])
            # We have to adjust the meaned loss by the cleansed samples
            result = cleansed_loss * batch_size / (batch_size - nonconvergent_samples)
        elif self.reduction == "sum":
            result = torch.sum(cleansed_loss)
        elif self.reduction == "none":
            result = cleansed_loss
        else:
            raise ValueError("Reduction must be either 'mean', 'sum' or 'none'.")
        return result, stats
