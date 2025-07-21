# OUTDATED: This file is deprecated and will be removed in a couple of PRs.
from dataclasses import dataclass
from typing import Any
import warnings

import casadi as ca
import numpy as np
import torch
from torch import autograd as autograd, nn as nn

from acados_template import AcadosSimSolver
from leap_c.ocp.acados.mpc import Mpc, MpcParameter, MpcBatchedState, MpcInput, MpcOutput


@dataclass
class ModuleCtx:
    dudp_global: torch.Tensor | None = None


class MPCSolutionFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mpc: Mpc,
        x0: torch.Tensor,
        u0: torch.Tensor | None,
        p_global: torch.Tensor | None,
        p_the_rest: MpcParameter | None,
        initializations: MpcBatchedState | None,
        module_ctx: ModuleCtx | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x0.device
        dtype = x0.dtype

        need_dx0 = ctx.needs_input_grad[1]
        need_du0 = ctx.needs_input_grad[2]
        need_dp_global = ctx.needs_input_grad[3]
        need_dudp_global = need_dp_global and u0 is None
        need_dudx0 = need_dx0 and u0 is None

        if p_the_rest.p_global is not None:
            raise ValueError("p_global should not be set in p_the_rest")

        p_global = p_global.detach().cpu().numpy().astype(np.float64)  # type: ignore
        if p_the_rest is None:
            p_whole = MpcParameter(p_global=p_global)  # type: ignore
        else:
            p_the_rest = p_the_rest.ensure_float64()
            p_whole = p_the_rest._replace(p_global=p_global)

        x0_np = tensor_to_numpy(x0)
        u0_np = None if u0 is None else tensor_to_numpy(u0)
        mpc_input = MpcInput(x0_np, u0_np, parameters=p_whole)

        mpc_output, mpc_state = mpc(
            mpc_input=mpc_input,
            mpc_state=initializations,
            dudx=need_dudx0,
            dudp=need_dudp_global,
            dvdx=need_dx0,
            dvdu=need_du0,
            dvdp=need_dp_global,
            use_adj_sens=True,
        )
        u_star = mpc_output.u0 if u0 is None else None  # type:ignore
        dudp_global = mpc_output.du0_dp_global if need_dudp_global else None  # type:ignore
        dudx0 = mpc_output.du0_dx0 if need_dudx0 else None  # type:ignore
        dvaluedp_global = mpc_output.dvalue_dp_global if need_dp_global else None  # type:ignore
        dvaluedu0 = mpc_output.dvalue_du0 if need_du0 else None  # type:ignore
        dvaluedx0 = mpc_output.dvalue_dx0 if need_dx0 else None  # type:ignore
        value = mpc_output.Q if u0 is not None else mpc_output.V
        status = mpc_output.status

        ctx.dudp_global = dudp_global
        ctx.dudx0 = dudx0
        ctx.dvaluedp_global = dvaluedp_global
        ctx.dvaluedu0 = dvaluedu0
        ctx.dvaluedx0 = dvaluedx0
        ctx.u0_was_none = u0 is None

        if module_ctx is not None:
            module_ctx.dudp_global = dudp_global

        value = mpc_output.Q if u0 is not None else mpc_output.V
        if u0 is None:
            # TODO (Jasper): Why do we have an utils dim here?
            # u_star = u_star[..., 0]  # type: ignore
            u_star = torch.tensor(u_star, device=device, dtype=dtype)
        else:
            u_star = torch.empty(1)
            u_star[0] = float("nan")
            ctx.mark_non_differentiable(u_star)
        value = torch.tensor(value, device=device, dtype=dtype)
        status = torch.tensor(status, device=device, dtype=torch.int8)

        ctx.mark_non_differentiable(status)

        return u_star, value, status, mpc_state

    @staticmethod
    @autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        dLossdu, dLossdvalue, _, _ = grad_outputs

        device = dLossdvalue.device
        dtype = dLossdvalue.dtype

        need_dx0 = ctx.needs_input_grad[1]
        need_du0 = ctx.needs_input_grad[2]
        need_dp_global = ctx.needs_input_grad[3]
        u0_was_none = ctx.u0_was_none
        need_dudx0 = need_dx0 and u0_was_none
        need_dudp_global = need_dp_global and u0_was_none

        if need_dx0:
            dvaluedx0 = ctx.dvaluedx0
            if dvaluedx0 is None:
                raise ValueError(
                    "Something went wrong: The necessary sensitivities dvaluedx0 from the forward pass do not exist."
                )
            dvaluedx0 = torch.tensor(dvaluedx0, device=device, dtype=dtype)
            grad_x = torch.einsum("bj,b->bj", dvaluedx0, dLossdvalue)
            if need_dudx0:
                dudx0 = ctx.dudx0
                if dudx0 is None:
                    raise ValueError(
                        "Something went wrong: The necessary sensitivities dudx0 from the forward pass do not exist."
                    )
                dudx0 = torch.tensor(dudx0, device=device, dtype=dtype)
                grad_x = grad_x + torch.einsum("bkj,bk->bj", dudx0, dLossdu)
        else:
            grad_x = None

        if need_dp_global:
            dvaluedp_global = ctx.dvaluedp_global
            if dvaluedp_global is None:
                raise ValueError(
                    "Something went wrong: The necessary sensitivities dvaluedp_global from the forward pass do not exist."
                )
            dvaluedp_global = torch.tensor(dvaluedp_global, device=device, dtype=dtype)
            grad_p = torch.einsum("bj,b->bj", dvaluedp_global, dLossdvalue)
            if need_dudp_global:
                dudp_global = ctx.dudp_global
                if dudp_global is None:
                    raise ValueError(
                        "ctx.needs_input_grad wrt. p was not True in forward pass, it is not working as we expected."
                    )
                dudp_global = torch.tensor(dudp_global, device=device, dtype=dtype)
                grad_p = grad_p + torch.einsum("bkj,bk->bj", dudp_global, dLossdu)
        else:
            grad_p = None

        if need_du0:
            dvaluedu0 = ctx.dvaluedu0
            if dvaluedu0 is None:
                raise ValueError(
                    "Something went wrong: The necessary sensitivities dvaluedu0 from the forward pass do not exist."
                )
            dvaluedu0 = torch.tensor(dvaluedu0, device=device, dtype=dtype)
            grad_u = torch.einsum("bj,b->bj", dvaluedu0, dLossdvalue)
        else:
            grad_u = None

        # print("Before: Grad p min", grad_p.min(), "Grad p max", grad_p.max())

        # # gradient clipping
        # norm_grad_p = torch.norm(grad_p, p=2, dim=-1)
        # max_norm = 5e-3
        # ratios = (norm_grad_p / max_norm).clamp(min=1.0)

        # grad_p = grad_p / ratios.unsqueeze(-1)


        # print("After: Grad p min", grad_p.min(), "Grad p max", grad_p.max())

        return (None, grad_x, grad_u, grad_p, None, None, None)


def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


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
        # raise deprecation warning if mpc class is used
        warning_msg = (
            "The MpcSolutionModule class is deprecated and will be removed in a future version. "
            "Please use the AcadosDiffMpc class instead."
        )
        warnings.warn(warning_msg, DeprecationWarning)
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

        ctx = ModuleCtx()
        u0, value, status, state = MPCSolutionFunction.apply(  # type:ignore
            self.mpc,
            mpc_input.x0,
            mpc_input.u0,
            p_glob,
            p_rest,
            mpc_state,
            ctx,
        )

        if ctx.dudp_global is not None:
            dudp_global = torch.tensor(ctx.dudp_global, device=u0.device, dtype=u0.dtype)
        else:
            dudp_global = None

        if mpc_input.u0 is None:
            V = value
            Q = None
        else:
            Q = value
            V = None

        return (
            MpcOutput(u0=u0, Q=Q, V=V, status=status, du0_dp_global=dudp_global),
            state,
            self.mpc.last_call_stats,
        )
