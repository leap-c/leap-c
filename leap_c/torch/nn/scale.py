import torch
import torch.autograd as autograd

from gymnasium import spaces


class MinMaxStraightThroughFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, space: spaces.Box) -> torch.Tensor:
        low = torch.tensor(space.low, dtype=x.dtype, device=x.device)
        high = torch.tensor(space.high, dtype=x.dtype, device=x.device)

        return (x - low) / (high - low)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:  # type: ignore
        return grad_output, None


def min_max_scaling(
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
        return MinMaxStraightThroughFunction.apply(x, space)  # type: ignore

    low = torch.tensor(space.low, dtype=x.dtype, device=x.device)
    high = torch.tensor(space.high, dtype=x.dtype, device=x.device)

    # check if low and high are correctly set
    if not (low < high).all():
        raise ValueError("The low bound must be less than the high bound.")
    if torch.isinf(low).any() or torch.isinf(high).any():
        raise ValueError("The low and high bounds must not be infinite.")

    return (x - low) / (high - low)
