"""Provides a simple Gaussian layer that allows policies to respect action bounds."""

from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions.beta import Beta

BoundedDistributionName = Literal["squashed_gaussian", "scaled_beta"]


def get_bounded_distribution(name: BoundedDistributionName, **init_kwargs) -> "BoundedDistribution":
    if name == "squashed_gaussian":
        return SquashedGaussian(**init_kwargs)
    elif name == "scaled_beta":
        return ScaledBeta(**init_kwargs)
    else:
        raise ValueError(f"Unknown bounded distribution: {name}")


class BoundedDistribution(nn.Module):
    """An abstract class for bounded distributions."""

    @abstractmethod
    def forward(
        self, *defining_parameters, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the distribution.

        If `deterministic` is True, the mode of the distribution is used instead of
        sampling.

        Returns:
            A tuple containing the samples, their log_prob and a dictionary of stats.
        """
        ...

    @abstractmethod
    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        """Returns param size required to define the distribution for the given output dim.

        Args:
            output_dim: The dimensionality of the output space (e.g., action space).

        Returns:
            A tuple of integers, each integer specifying the size of one
            parameter required to define the distribution in the forward pass.
        """
        ...


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
        """Apply the inverse transformation to the input tensor.

        The inverse transformation is a descale and then arctanh.
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


class SquashedGaussian(BoundedDistribution):
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
        """Sample from the SquashedGaussian distribution.

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

    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        return (output_dim, output_dim)

    def inverse(self, x: torch.Tensor, padding: float = 0.001) -> torch.Tensor:
        """Apply the inverse transformation to the input tensor.

        The inverse transformation is a descale and then arctanh.
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


class ScaledBeta(BoundedDistribution):
    """A unimodal scaled Beta distribution.

    Samples the output from a Beta distribution specified by the input,
    and then scales and shifts the result to match the space. Unomodality is ensured
    by enforcing alpha, beta > 1.

    Can for example be used to enforce certain action bounds of a stochastic policy.

    Attributes:
        scale: The scale of the space-fitting transform.
        loc: The location of the space-fitting transform (for shifting).
        log_alpha_min: The minimum value for the logarithm of the alpha parameter.
        log_beta_min: The minimum value for the logarithm of the beta parameter.
        log_alpha_max: The maximum value for the logarithm of the alpha parameter.
        log_beta_max: The maximum value for the logarithm of the beta parameter.
    """

    scale: torch.Tensor
    loc: torch.Tensor
    log_alpha_min: float
    log_beta_min: float
    log_alpha_max: float
    log_beta_max: float

    def __init__(
        self,
        space: spaces.Box,
        log_alpha_min: float = -10.0,
        log_beta_min: float = -10.0,
        log_alpha_max: float = 10.0,
        log_beta_max: float = 10.0,
    ):
        """Initializes the ScaledBeta module.

        Args:
            space: Space the output should fit to.
            log_alpha_min: The minimum value for the logarithm of the alpha parameter.
            log_beta_min: The minimum value for the logarithm of the beta parameter.
            log_alpha_max: The maximum value for the logarithm of the alpha parameter.
            log_beta_max: The maximum value for the logarithm of the beta parameter.
        """
        super().__init__()

        self.log_alpha_max = log_alpha_max
        self.log_beta_max = log_beta_max
        self.log_alpha_min = log_alpha_min
        self.log_beta_min = log_beta_min

        loc = space.low
        scale = space.high - space.low

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(
        self, log_alpha: torch.Tensor, log_beta: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the ScaledBeta distribution.

        Note that alpha and beta are enforced to be > 1 to ensure concavity.

        Args:
            log_alpha: The logarithm of the alpha parameter of the Beta distribution.
            log_beta: The logarithm of the beta parameter of the Beta distribution.
            deterministic: If True, the output will just be spacefitting(mode),
                no sampling is taking place.

        Returns:
            An output sampled from the ScaledBeta distribution, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        log_alpha = torch.clamp(log_alpha, self.log_alpha_min, self.log_alpha_max)
        log_beta = torch.clamp(log_beta, self.log_beta_min, self.log_beta_max)
        # Add 1 to ensure concavity
        alpha = torch.exp(log_alpha) + 1.0
        beta = torch.exp(log_beta) + 1.0

        dist = Beta(alpha, beta)

        if deterministic:
            y = dist.mode
        else:
            # reparameterization trick
            y = dist.rsample()
        log_prob = dist.log_prob(y)

        y_scaled = y * self.scale[None, :] + self.loc[None, :]
        log_prob -= torch.log(self.scale[None, :])
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # We could return the mean of alpha and beta as stats,
        # but I think they should at least be investigated for each action
        # dimension independently
        return y_scaled, log_prob, {}

    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        return (output_dim, output_dim)
