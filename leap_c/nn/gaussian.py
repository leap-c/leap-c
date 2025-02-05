"""Provides a simple Gaussian layer that optionally allows policies to respect action bounds."""

from gymnasium import spaces
import numpy as np
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import (
    AffineTransform,
    TanhTransform,
)
import torch.nn as nn
import torch.nn.functional as F


class Gaussian(nn.Module):
    """A Gaussian transformed by a Gaussian.

    The output is sampled from this distribution and then squashed with a tanh function.
    # TODO (Jasper): Why are we not using the transformed distr class from torch.
    """

    def __init__(
        self,
        action_space: spaces.Box,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ):
        """Initializes the TanhNormal module.

        Args:
            action_space: The action space of the environment. Used for constraints.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if not np.any(np.isinf(action_space.low)) and not np.any(
            np.isinf(action_space.high)
        ):
            low = torch.tensor(action_space.low, dtype=torch.float32)
            high = torch.tensor(action_space.high, dtype=torch.float32)

            self.transform = [
                AffineTransform(loc=low, scale=(high - low) / 2),
                TanhTransform(),
            ]
        else:
            self.transform = None

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mean: The mean of the distribution.
            log_std: The logarithm of the standard deviation of the distribution.
            deterministic: If True, the output will just be tanh(mean), no sampling is taking place.

        Returns:
            an output sampled from the TanhNormal, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        # TODO (Jasper): Do we need this?
        std = torch.exp(log_std.clamp(self.log_std_min, self.log_std_max))

        distr = Normal(mean, std)

        if self.transform is not None:
            distr = TransformedDistribution(distr, self.transform)

        if deterministic:
            action = mean
            if self.transform is not None:
                for t in self.transform:
                    action = t(action)
        else:
            action = distr.rsample()

        log_prob = distr.log_prob(action)

        return action, log_prob
