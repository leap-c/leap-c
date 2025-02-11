"""Provides a simple Gaussian layer that optionally allows policies to respect action bounds."""

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gaussian(nn.Module):
    """A Gaussian potentially transformed by a tanh function.

    The output is sampled from this distribution and then squashed with a tanh function.
    # TODO (Jasper): Why are we not using the transformed distr class from torch.
    """

    scale: torch.Tensor
    loc: torch.Tensor

    def __init__(
        self,
        action_space: spaces.Box,
        log_std_min: float = -4,
        log_std_max: float = 2.0,
    ):
        """Initializes the TanhNormal module.

        Args:
            action_space: The action space of the environment. Used for constraints.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        loc = (action_space.high + action_space.low) / 2.0
        scale = (action_space.high - action_space.low) / 2.0

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        normal = torch.distributions.Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = normal.rsample()

        log_prob = normal.log_prob(action)
        action = torch.tanh(action)
        log_prob -= torch.log(self.scale * (1 - action.pow(2)) + 1e-6)

        action = action * self.scale + self.loc

        return action, log_prob
