"""This module creates PyTorch autograd functions"""

import torch

from leap_c.autograd.function import DiffFunction


def create_autograd_function(fun: DiffFunction) -> type[torch.autograd.Function]:
    """Creates a PyTorch autograd function from an object implementing
    NumPy-based forward and backward methods.

    The `fun` object must implement the `forward` and `backward` methods
    as described in the `Function` class. During the backward pass, the
    custom context objects gets the information about which inputs need
    gradients via `torch_ctx.needs_input_grad`.

    Args:
        fun: An object implementing `forward(custom_ctx, *args)` and
            `backward(custom_ctx, *grad_outputs)` methods, where
            `custom_ctx` is a context object that can be used to store
            intermediate values for the backward pass.

    Returns:
        A PyTorch autograd function, wrapping the object.

    Usage:
        fn = create_autograd_function(obj)
        ctx, y = fn(*inputs)
    """

    class AutogradFunction(torch.autograd.Function):
        @staticmethod
        def forward(torch_ctx, *args):
            custom_ctx, *non_ctx_args = args
            device = non_ctx_args[0].device
            np_args = _to_np(non_ctx_args)

            custom_ctx, *outputs = fun.forward(custom_ctx, *np_args)  # type: ignore
            torch_ctx.custom_ctx = custom_ctx

            if len(outputs) == 1:
                return custom_ctx, _to_tensor(outputs[0], device)

            return custom_ctx, *_to_tensor(outputs, device)

        @staticmethod
        def backward(torch_ctx, grad_ctx, *grad_outputs):  # type: ignore
            device = grad_outputs[0].device
            custom_ctx = torch_ctx.custom_ctx
            custom_ctx.needs_input_grad = torch_ctx.needs_input_grad
            np_grad_outputs = _to_np(grad_outputs)

            grad_inputs = fun.backward(custom_ctx, *np_grad_outputs)  # type: ignore

            torch_grad_inputs = _to_tensor(grad_inputs, device)

            if isinstance(torch_grad_inputs, torch.Tensor):
                return None, torch_grad_inputs

            return None, *torch_grad_inputs

    return AutogradFunction


def _to_np(data):
    if data is None:
        return None
    if isinstance(data, (tuple, list)):
        return tuple(_to_np(item) for item in data)
    try:
        return data.detach().cpu().numpy()
    except AttributeError:
        return data


def _to_tensor(data, device, dtype=torch.float32):
    if data is None:
        return None
    if isinstance(data, (tuple, list)):
        return tuple(_to_tensor(item, device, dtype) for item in data)
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    return torch.tensor(data, device=device, dtype=dtype)
