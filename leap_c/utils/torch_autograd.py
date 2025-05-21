"""Utility function to create a PyTorch autograd function"""

import torch


def _to_np(data):
    return tuple(e.detach().cpu().numpy() for e in data)


def _to_tensor(data, device, dtype=torch.float32):
    if isinstance(data, tuple):
        return tuple(_to_tensor(item, device, dtype) for item in data)
    return torch.tensor(data, device=device, dtype=dtype)

def _add_none(data):
    if isinstance(data, tuple):
        return None, *data
    return None, data


def create_autograd_function(obj) -> type[torch.autograd.Function]:
    """
    Creates a PyTorch autograd function from an object implementing
    NumPy-based forward and backward methods.

    An example:
        class MyAutogradFunction:
            def forward(
                self, custom_ctx, *args
            ) -> np.ndarray | tuple[np.ndarray]:
                # Perform forward computation using NumPy arrays
                return outputs

            def backward(
                self, custom_ctx, *grad_outputs
            ) -> np.ndarray | tuple[np.ndarray]:
                # Compute gradients using NumPy arrays
                return grad_inputs  # A single gradient or a tuple

    Args:
        obj: An object implementing `forward(custom_ctx, *args)` and
             `backward(custom_ctx, *grad_outputs)` using NumPy.

    Returns:
        A PyTorch autograd function, wrapping the object.

    Usage:
        ctx = SimpleNamespace()
        fn = create_autograd_function(obj)
        y = fn(ctx, *inputs)
    """

    class AutogradFunction(torch.autograd.Function):
        @staticmethod
        def forward(torch_ctx, custom_ctx, *args):
            device = args[0].device
            np_args = _to_np(args)

            outputs = obj.forward(custom_ctx, *np_args)
            torch_ctx.custom_ctx = custom_ctx

            return _to_tensor(outputs, device)

        @staticmethod
        def backward(torch_ctx, *grad_outputs):  # type: ignore
            device = grad_outputs[0].device
            custom_ctx = torch_ctx.custom_ctx
            np_grad_outputs = _to_np(grad_outputs)

            grad_inputs = obj.backward(custom_ctx, *np_grad_outputs)

            torch_grad_inputs = _to_tensor(grad_inputs, device)

            return _add_none(torch_grad_inputs)

    return AutogradFunction
