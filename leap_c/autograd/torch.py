"""This module creates PyTorch autograd functions"""

import torch

from leap_c.autograd.function import DiffFunction


def create_autograd_function(fun: DiffFunction) -> type[torch.autograd.Function]:
    """Creates a PyTorch autograd function from an object implementing
    NumPy-based forward and backward methods.

    The `fun` object must implement the `forward` and `backward` methods
    as described in the `Function` class.

    Args:
        fun: An object implementing `forward(custom_ctx, *args)` and

    Returns:
        A PyTorch autograd function, wrapping the object.

    Usage:
        fn = create_autograd_function(obj)
        y = fn(*inputs)
    """

    class AutogradFunction(torch.autograd.Function):
        @staticmethod
        def forward(torch_ctx, *args, custom_ctx=None):
            device = args[0].device
            np_args = _to_np(args)

            custom_ctx, *outputs = fun.forward(*np_args, ctx=custom_ctx)  # type: ignore
            torch_ctx.custom_ctx = custom_ctx

            if len(outputs) == 1:
                return custom_ctx, _to_tensor(outputs[0], device)

            return custom_ctx, *_to_tensor(outputs, device)

        @staticmethod
        def backward(torch_ctx, _, *grad_outputs):  # type: ignore
            device = grad_outputs[0].device
            custom_ctx = torch_ctx.custom_ctx
            np_grad_outputs = _to_np(grad_outputs)

            grad_inputs = fun.backward(custom_ctx, *np_grad_outputs)

            torch_grad_inputs = _to_tensor(grad_inputs, device)

            return torch_grad_inputs

    return AutogradFunction


def _to_np(data):
    return tuple(e.detach().cpu().numpy() for e in data)


def _to_tensor(data, device, dtype=torch.float32):
    if isinstance(data, (tuple, list)):
        return tuple(_to_tensor(item, device, dtype) for item in data)
    return torch.tensor(data, device=device, dtype=dtype)
