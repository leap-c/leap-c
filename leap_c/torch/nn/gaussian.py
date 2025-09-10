"""Provides a simple Gaussian layer that allows policies to respect action bounds."""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces


class BoundedTransform(nn.Module):
    """A bounded transform.

    The input is squashed with a tanh function and then scaled and shifted to match the space.

    Attributes:
        scale: The scale of the transform.
        loc: The location of the transform (for shifting).
    """

    scale: torch.Tensor
    loc: torch.Tensor

    def __init__(
        self,
        space: spaces.Box,
    ):
        """Initializes the Bounded Transform module.

        Args:
            space: The space that the transform is bounded to.
        """
        super().__init__()
        loc = (space.high + space.low) / 2.0
        scale = (space.high - space.low) / 2.0

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the transformation to the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The squashed tensor, scaled and shifted to match the action space.
        """
        x = torch.tanh(x)
        return x * self.scale[None, :] + self.loc[None, :]

    def inverse(self, x: torch.Tensor, padding: float = 0.001) -> torch.Tensor:
        """Applies the inverse transformation to the input tensor, i.e., descale, then arctanh.
        For numerical stability, the input is slightly padded away from the bounds
        before applying arctanh.

        Args:
            x: The input tensor.
            padding: The amount of padding to distance the action of the bounds.

        Returns:
            The inverse squashed tensor, scaled and shifted to match the action space.
        """
        abs_padding = self.scale[None, :] * padding
        x = (x - self.loc[None, :]) / (self.scale[None, :] + 2 * abs_padding)
        return torch.arctanh(x)


class SquashedGaussian(nn.Module):
    """A squashed Gaussian.
    Samples the output from a Gaussian distribution specified by the input,
    and then squashes the result with a tanh function.
    Finally, the output of the tanh function is scaled and shifted to match the space.

    Can for example be used to enforce certain action bounds of a stochastic policy.

    Attributes:
        scale: The scale of the space-fitting transform.
        loc: The location of the space-fitting transform (for shifting).
    """

    scale: torch.Tensor
    loc: torch.Tensor
    log_std_min: float
    log_std_max: float

    def __init__(
        self,
        space: spaces.Box,
        log_std_min: float = -4,
        log_std_max: float = 2.0,
    ):
        """Initializes the SquashedGaussian module.

        Args:
            space: Space the output should fit to.
            log_std_min: The minimum value for the logarithm of the standard deviation.
            log_std_max: The maximum value for the logarithm of the standard deviation.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        loc = (space.high + space.low) / 2.0
        scale = (space.high - space.low) / 2.0

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """
        Args:
            mean: The mean of the normal distribution.
            log_std: The logarithm of the standard deviation of the normal distribution,
                of the same shape as the mean (i.e., assuming independent dimensions).
                Will be clamped according to the attributes of this class.
            deterministic: If True, the output will just be spacefitting(tanh(mean)),
                no sampling is taking place.

        Returns:
            An output sampled from the SquashedGaussian, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            y = mean
        else:
            # reparameterization trick
            y = mean + std * torch.randn_like(mean)

        log_prob = -0.5 * ((y - mean) / std).pow(2) - log_std - np.log(np.sqrt(2) * np.pi)

        y = torch.tanh(y)

        log_prob -= torch.log(self.scale[None, :] * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        y_scaled = y * self.scale[None, :] + self.loc[None, :]

        stats = {"gaussian_unsquashed_std": std.prod(dim=-1).mean().item()}

        return y_scaled, log_prob, stats
