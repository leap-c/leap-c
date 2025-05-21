import torch
from types import SimpleNamespace

from leap_c.utils.torch_autograd import create_autograd_function


class DummyFunction:
    def forward(self, ctx, x_np):
        ctx.saved = x_np.copy()
        y_np = x_np**2 + 1
        return y_np

    def backward(self, ctx, grad_y_np):
        x_np = ctx.saved
        grad_x_np = 2 * x_np * grad_y_np
        return grad_x_np


def test_create_autograd_function():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    ctx = SimpleNamespace()

    autograd_fn = create_autograd_function(DummyFunction())
    y = autograd_fn.apply(ctx, x)

    expected_y = x.detach() ** 2 + 1
    assert torch.allclose(y, expected_y)  # type: ignore

    y.sum().backward()  # type: ignore
    expected_grad = 2 * x.detach()
    assert torch.allclose(x.grad, expected_grad)  # type: ignore


class DummyTupleFunction:
    def forward(self, ctx, x_np, y_np):
        ctx.saved = (x_np.copy(), y_np.copy())
        out1 = x_np + y_np
        out2 = x_np * y_np
        return out1, out2

    def backward(self, ctx, grad_out1_np, grad_out2_np):
        x_np, y_np = ctx.saved
        grad_x = grad_out1_np + grad_out2_np * y_np
        grad_y = grad_out1_np + grad_out2_np * x_np
        return grad_x, grad_y


def test_create_autograd_function_with_tuples():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
    ctx = SimpleNamespace()

    autograd_fn = create_autograd_function(DummyTupleFunction())
    out1, out2 = autograd_fn.apply(ctx, x, y)  # type: ignore

    expected_out1 = x + y
    expected_out2 = x * y
    assert torch.allclose(out1, expected_out1)
    assert torch.allclose(out2, expected_out2)

    loss = out1.sum() + out2.sum()
    loss.backward()

    expected_grad_x = torch.ones_like(x) + y.detach()
    expected_grad_y = torch.ones_like(y) + x.detach()
    assert torch.allclose(x.grad, expected_grad_x)  # type: ignore
    assert torch.allclose(y.grad, expected_grad_y)  # type: ignore
