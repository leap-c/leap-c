import torch
import torch.nn as nn
import torch.autograd as autograd

from gymnasium import spaces


def string_to_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Activation function {activation} not recognized.")


class NormalizeStraightThroughFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, space: spaces.Box) -> torch.Tensor:
        low = torch.tensor(space.low, dtype=x.dtype, device=x.device)
        high = torch.tensor(space.high, dtype=x.dtype, device=x.device)

        return (x - low) / (high - low)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:  # type: ignore
        return grad_output, None


def normalize(
    x: torch.Tensor, space: spaces.Box, straight_through: bool = False
) -> torch.Tensor:
    """Normalize a tensor x to the range [0, 1] given a Box space.

    Args:
        x: The tensor to normalize.
        space: The Box space to normalize the tensor to.
        straight_through: If True, the gradient will be passed through the normalization.

    Returns:
        The normalized tensor.
    """
    if straight_through:
        return NormalizeStraightThroughFunction.apply(x, space)  # type: ignore

    low = torch.tensor(space.low, dtype=x.dtype, device=x.device)
    high = torch.tensor(space.high, dtype=x.dtype, device=x.device)

    return (x - low) / (high - low)
