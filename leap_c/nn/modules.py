import math
from typing import Any

import casadi as ca
import torch
import torch.nn as nn
from acados_template import AcadosSimSolver
from leap_c.mpc import MPC, MPCBatchedState, MPCParameter

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


class MPCSolutionModule(nn.Module):
    """A pytorch module to allow differentiating the solution given by an MPC controller,
    with respect to some inputs. Backpropagation works by using the sensitivities
    given by the MPC. If differentiation with respect to parameters is desired, they must
    be global over the horizon (contrary to stagewise parameters).

        NOTE: This solves every sample in the batch sequentially, so it is not efficient for large batch sizes.

        NOTE: Make sure that you follow the documentation of AcadosOcpSolver.eval_solution_sensitivity
        and AcadosOcpSolver.eval_and_get_optimal_value_gradient
        or else the gradients used in the backpropagation might be erroneous!

        NOTE: Make sure you also read the documentation of the forward method!

        NOTE: The status output can be used to rid yourself from non-converged solutions, e.g., by using the
        CleanseAndReducePerSampleLoss module.

        NOTE: Currently, acados guards calculating sensitivities for parameters in constraints with an Exception.
        Still, it might be that you want to have such parameters,
        and still use the sensitivities of OTHER parameters.
        For this, you can, e.g., in acados github commit 95c341c6,
        disable the Exception in acados_ocp.py, line 800 to still calculate all the sensitivities
        and then only use the sensitivities of the parameters not belonging to constraints
        (because as acados states in eval_solution_sensitivity, they are erroneous).

    Attributes:
        mpc: The MPC object to use.
    """

    def __init__(self, mpc: MPC):
        super().__init__()
        self.mpc = mpc

    def forward(
        self,
        x0: torch.Tensor,
        u0: torch.Tensor | None = None,
        p_global: torch.Tensor | None = None,
        p_stagewise: MPCParameter | None = None,
        initializations: MPCBatchedState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Differentiation is only allowed with respect to x0, u0 and p_global.


        Parameters:
            x0: The initial state of the MPC, shape (batch_size, xdim).
            u0: The initial action of the MPC, shape (batch_size, udim).
                If it is not given, the initial action will be variable in the MPC.
            p_global: The parameters of the MPC, shape (batch_size, p_global_dim).
            p_stagewise: The remaining parameter information for the MPC, i.e., the stagewise parameters (batched according to the other input).
                NOTE that it should not contain p_global, since this will be overwritten by p_global!
            initializations: The batched MPCState used for initialization in the mpc solve.
        Returns: A tuple (u_star, value, status, stats).
            u_star is the first optimal action of the MPC solution, given the initial state and parameters.
                NOTE that this is a tensor of shape (1, ) containing NaN if u0 was given in the forward pass.
            value is the value of the MPC solution (the cost of the objective function in the solution).
                Corresponds to the Value function if u0 is not given, and the Q function if u0 is given.
            status is the status of the MPC solution, where 0 means converged and all other integers count as not converged,
                (useful for e.g., logging, debugging or cleaning the backward pass from non-converged solutions).
            stats is a dictionary containing additional statistics about the solution, such as the number of iterations the solver took, etc.

        NOTE: An extension to allow for outputting and differentiating also with respect to other stages
            (meaning stages of the action and state trajectories) than the first one is possible, but not implemented yet.
        NOTE: Using a multiphase MPC formulation allows differentiation with respect to parameters that are not "truly" global,
            but this is not implemented yet.
        """
        u_star, value, status = MPCSolutionFunction.apply(  # type:ignore
            self.mpc,
            x0,
            u0,
            p_global,
            p_stagewise,
            initializations,
        )
        stats = self.mpc.last_call_stats
        return u_star, value, status, stats


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
