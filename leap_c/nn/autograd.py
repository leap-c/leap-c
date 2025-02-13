"""Provides a PyTorch module that wraps a CasADi expression.

TODO (Jasper): Does JIT compilation speed up the module?
"""

import casadi as ca
import numpy as np
import torch
import torch.autograd as autograd
from acados_template import AcadosSimSolver

from leap_c.mpc import MPC, MPCBatchedState, MPCInput, MPCParameter
from leap_c.util import tensor_to_numpy


class AutogradCasadiFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx, f: ca.Function, jtimes: ca.Function, *inputs: torch.Tensor
    ) -> torch.Tensor:
        assert inputs[0].ndim == 2
        device = inputs[0].device
        dtype = inputs[0].dtype

        # assert that the inputs are 1D and batched assuming first dimension is batch size
        inputs = [input.detach().cpu().numpy().T for input in inputs]

        # save the jacobian and inputs for the backward pass
        ctx.inputs = inputs
        ctx.jtimes = jtimes

        result = f(*inputs).full()

        # convert numpy to tensor and transpose
        return torch.tensor(result, device=device, dtype=dtype).T

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # convert grad_output to numpy
        device = grad_output.device
        dtype = grad_output.dtype
        grad_output = grad_output.detach().cpu().numpy().T

        # prepare inputs and remember the splits
        inputs = ctx.inputs
        sizes = [input.shape[0] for input in inputs]
        splits = np.cumsum(sizes)[:-1]
        inputs = np.concatenate(inputs, axis=0)

        # compute the jacobian
        grad_input = ctx.jtimes(inputs, grad_output).T
        # split the grads for the individual inputs
        grad_input = np.split(grad_input, splits, axis=1)
        # convert to tensor
        grad_input = [
            torch.tensor(grad, device=device, dtype=dtype) for grad in grad_input
        ]

        return (None, None, *grad_input)


class DynamicsSimFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sim: AcadosSimSolver,
        x: torch.Tensor,
        u: torch.Tensor,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        xdim = x.shape[1]
        udim = u.shape[1]

        # assert that the inputs are 1D and batched assuming first dimension is batch size
        x = x.detach().cpu().numpy()
        u = u.detach().cpu().numpy()
        if p is not None:
            p = p.detach().cpu().numpy()

        x_next = np.zeros_like(x)
        Sx = np.zeros((batch_size, xdim, xdim))
        Su = np.zeros((batch_size, xdim, udim))

        for i in range(x.shape[0]):
            # TODO (Jasper): Replace with method simulate!
            sim.set("x", x[i, :])
            sim.set("u", u[i, :])
            if p is not None:
                sim.set("p", p[i, :])
            sim.solve()
            x_next[i, :] = sim.get("x")
            Sx[i, ...] = sim.get("Sx")
            Su[i, ...] = sim.get("Su")

        # save the jacobian for the backward pass
        ctx.Sx = Sx
        ctx.Su = Su

        return torch.tensor(x_next, device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # convert grad_output to numpy
        device = grad_output.device
        dtype = grad_output.dtype

        # compute gradients of the inputs
        Sx = torch.tensor(ctx.Sx, device=device, dtype=dtype)
        Su = torch.tensor(ctx.Su, device=device, dtype=dtype)
        grad_x = torch.einsum("bkj,bk->bj", Sx, grad_output)
        grad_u = torch.einsum("bkj,bk->bj", Su, grad_output)

        # print(Sx[0])
        # print(grad_x[0])

        return (None, grad_x, grad_u, None)


class MPCSolutionFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mpc: MPC,
        x0: torch.Tensor,
        u0: torch.Tensor | None,
        p_global: torch.Tensor | None,
        p_the_rest: MPCParameter | None,
        initializations: MPCBatchedState | None,
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
            p_whole = MPCParameter(p_global=p_global)  # type: ignore
        else:
            p_the_rest = p_the_rest.ensure_float64()
            p_whole = p_the_rest._replace(p_global=p_global)


        x0_np = tensor_to_numpy(x0)
        u0_np = None if u0 is None else tensor_to_numpy(u0)
        mpc_input = MPCInput(x0_np, u0_np, parameters=p_whole)

        mpc_output = mpc(
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

        value = mpc_output.Q if u0 is not None else mpc_output.V
        if u0 is None:
            # TODO (Jasper): Why do we have an extra dim here?
            # u_star = u_star[..., 0]  # type: ignore
            u_star = torch.tensor(u_star, device=device, dtype=dtype)
        else:
            u_star = torch.empty(1)
            u_star[0] = float("nan")
            ctx.mark_non_differentiable(u_star)
        value = torch.tensor(value, device=device, dtype=dtype)
        status = torch.tensor(status, device=device, dtype=torch.int8)

        ctx.mark_non_differentiable(status)

        return u_star, value, status

    @staticmethod
    @autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        dLossdu, dLossdvalue, _ = grad_outputs

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

        return (None, grad_x, grad_u, grad_p, None, None)
