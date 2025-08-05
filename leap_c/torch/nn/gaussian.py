"""Provides a simple Gaussian layer that allows policies to respect action bounds."""

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn


class BoundedTransform(nn.Module):
    """A bounded transform.

    The output is squashed with a tanh function and then scaled and shifted to match the space.
    """

    scale: torch.tensor
    loc: torch.tensor

    def __init__(
        self,
        space: spaces.Box,
    ):
        """Initializes the Bounded Transform module.

        Args:
            space: The space that the transform is bounded to.
            padding: The amount of padding to subtract to the bounds.
        """
        super().__init__()
        loc = (space.high + space.low) / 2.0
        scale = (space.high - space.low) / 2.0

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the squashing function to the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The squashed tensor, scaled and shifted to match the action space.
        """
        x = torch.tanh(x)
        return x * self.scale[None, :] + self.loc[None, :]

    def inverse(self, x: torch.Tensor, padding: float = 0.001) -> tuple[torch.Tensor]:
        """Applies the inverse squashing function to the input tensor.

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

    The output is sampled from this distribution and then squashed with a tanh function.
    Finally, the output of the tanh function is scaled and shifted to match the space.
    """

    scale: torch.Tensor
    loc: torch.Tensor

    def __init__(
        self,
        space: spaces.Box,
        log_std_min: float = -4,
        log_std_max: float = 2.0,
    ):
        """Initializes the SquashedGaussian module.

        Args:
            space: The action space of the environment. Used for constraints.
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
            mean: The mean of the distribution.
            log_std: The logarithm of the standard deviation of the distribution.
            deterministic: If True, the output will just be tanh(mean), no sampling is taking place.

        Returns:
            An output sampled from the TanhNormal, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            y = mean
        else:
            # reparameterization trick
            y = mean + std * torch.randn_like(mean)

        log_prob = (
            -0.5 * ((y - mean) / std).pow(2) - log_std - np.log(np.sqrt(2) * np.pi)
        )

        y = torch.tanh(y)

        log_prob -= torch.log(self.scale[None, :] * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        y_scaled = y * self.scale[None, :] + self.loc[None, :]

        stats = {"gaussian_unsquashed_std": std.prod(dim=-1).mean().item()}

        return y_scaled, log_prob, stats
