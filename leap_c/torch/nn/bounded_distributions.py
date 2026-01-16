"""Provides a simple Gaussian layer that allows policies to respect action bounds."""

from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions.beta import Beta

BoundedDistributionName = Literal["squashed_gaussian", "scaled_beta", "mode_concentration_beta"]


def get_bounded_distribution(name: BoundedDistributionName, **init_kwargs) -> "BoundedDistribution":
    if name == "squashed_gaussian":
        return SquashedGaussian(**init_kwargs)
    elif name == "scaled_beta":
        return ScaledBeta(**init_kwargs)
    elif name == "mode_concentration_beta":
        return ModeConcentrationBeta(**init_kwargs)
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

    @abstractmethod
    def inverse(self, normalized_x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse transformation to the input tensor.

        Args:
            normalized_x: The input tensor.

        Returns:
            The inverse transformed tensor.
        """
        ...


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
        padding: float = 0.0001,
    ):
        """Initializes the SquashedGaussian module.

        Args:
            space: Space the output should fit to.
            log_std_min: The minimum value for the logarithm of the standard deviation.
            log_std_max: The maximum value for the logarithm of the standard deviation.
            padding: The amount of padding to distance the action of the bounds, when
                using the inverse transformation for the anchoring. This improves numerical
                stability.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.padding = padding
        self.space = space

        loc = (space.high + space.low) / 2.0
        scale = (space.high - space.low) / 2.0

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor | None = None,
        deterministic: bool = False,
        anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the SquashedGaussian distribution.

        Args:
            mean: The mean of the normal distribution.
            log_std: The logarithm of the standard deviation of the normal distribution,
                of the same shape as the mean (i.e., assuming independent dimensions).
                Will be clamped according to the attributes of this class.
                If None, the output is deterministic (no noise added to mean).
            deterministic: If True, the output will just be spacefitting(tanh(mean)),
                no sampling is taking place.
            anchor: Anchor point to shift the mean. Used for residual policies.

        Returns:
            An output sampled from the SquashedGaussian, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        if log_std is not None:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
        else:
            std = None

        if anchor is not None:
            # Convert anchor to tensor if it's a numpy array
            if not isinstance(anchor, torch.Tensor):
                anchor = torch.from_numpy(anchor).to(mean.device, dtype=mean.dtype)

            # TODO: Add a check to ensure anchor is within action space bounds

            inv_anchor = self.inverse(anchor)
            mean = mean + inv_anchor  # Use out-of-place operation to avoid modifying view

        if deterministic or std is None:
            y = mean
        else:
            # reparameterization trick
            y = mean + std * torch.randn_like(mean)

        if std is not None:
            log_prob = -0.5 * ((y - mean) / std).pow(2) - log_std - 0.5 * np.log(2 * np.pi)
        else:
            # Deterministic: log_prob is 0 in the unbounded space (delta distribution)
            log_prob = torch.zeros_like(mean)

        y = torch.tanh(y)

        log_prob -= torch.log(self.scale[None, :] * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        y_scaled = y * self.scale[None, :] + self.loc[None, :]

        stats = (
            {"gaussian_unsquashed_std": std.prod(dim=-1).mean().item()} if std is not None else {}
        )

        return y_scaled, log_prob, stats

    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        return (output_dim, output_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse transformation to the input tensor.

        The inverse transformation is a descale and then arctanh.
        For numerical stability, the input is slightly padded away from the bounds
        before applying arctanh.

        Args:
            x: The input tensor.

        Returns:
            The inverse squashed tensor, scaled and shifted to match the action space.
        """
        abs_padding = self.scale[None, :] * self.padding
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
        self,
        log_alpha: torch.Tensor,
        log_beta: torch.Tensor,
        deterministic: bool = False,
        anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the ScaledBeta distribution.

        Note that alpha and beta are enforced to be > 1 to ensure concavity.

        Args:
            log_alpha: The logarithm of the alpha parameter of the Beta distribution.
            log_beta: The logarithm of the beta parameter of the Beta distribution.
            deterministic: If True, the output will just be spacefitting(mode),
                no sampling is taking place.
            anchor: If provided, the Beta distribution's mode is centered around this anchor point.
                This is useful for action noise where the MPC output serves as the anchor.

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

        # If anchor is provided, center the distribution around it
        if anchor is not None:
            # TODO (Jasper): Check whether we want to do it differently?
            raise NotImplementedError("Anchor functionality not implemented for ScaledBeta yet.")

        log_prob -= torch.log(self.scale[None, :])
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # We could return the mean of alpha and beta as stats,
        # but I think they should at least be investigated for each action
        # dimension independently
        return y_scaled, log_prob, {}

    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        return (output_dim, output_dim)


class ModeConcentrationBeta(BoundedDistribution):
    """Beta distribution parameterized by mode and total concentration.

    This distribution is parameterized similarly to SquashedGaussian:
    - mode: The mode of the distribution in [0, 1] space
    - log_conc: The logarithm of the concentration parameter

    Attributes:
        scale: The scale of the space-fitting transform.
        loc: The location of the space-fitting transform (for shifting).
        log_conc_min: The minimum value for the logarithm of the concentration.
        log_conc_max: The maximum value for the logarithm of the concentration.

    NOTE: The mode in the forward pass is assumed to be in the interval [0, 1],
    and the concentration is clamped to be > 2 to ensure unimodality. Use
    the BetaParameterNetwork to output mode and concentration parameters from a neural network.
    """

    lb: torch.Tensor
    ub: torch.Tensor
    scale: torch.Tensor
    _beta_dist: Beta
    log_conc_min: float
    log_conc_max: float
    padding: float

    def __init__(
        self,
        space: spaces.Box,
        log_conc_min: float = np.log(2.0),
        log_conc_max: float = np.log(100.0),
        padding: float = 0.0001,
    ) -> None:
        """Initialize ModeConcentrationBeta distribution.

        Args:
            space: Space the output should fit to.
            log_conc_min: The minimum value for the logarithm of the concentration.
            log_conc_max: The maximum value for the logarithm of the concentration.
            padding: The amount of padding to distance the action of the bounds, when
                using the inverse transformation for the anchoring. This improves numerical
                stability.
        """
        super().__init__()
        self.log_conc_min = log_conc_min
        self.log_conc_max = log_conc_max
        self.padding = padding

        lb = torch.tensor(space.low, dtype=torch.float32)
        scale = torch.tensor(space.high - space.low, dtype=torch.float32)

        self.register_buffer("lb", lb)
        self.register_buffer("ub", lb + scale)
        self.register_buffer("scale", scale)

    def forward(
        self,
        mode: torch.Tensor,
        log_conc: torch.Tensor,
        deterministic: bool = False,
        anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the ModeConcentrationBeta distribution.

        Args:
            mode: The mode of the Beta distribution in [0, 1] space.
            log_conc: The logarithm of the concentration parameter,
                of the same shape as the mode (i.e., assuming independent dimensions).
                Will be clamped according to the attributes of this class.
            deterministic: If True, the output will just be spacefitting(mode),
                no sampling is taking place.
            anchor: Anchor point to shift the mode. Used for residual policies.

        Returns:
            An output sampled from the ModeConcentrationBeta, the log probability of this output
            and a statistics dict.
        """
        # Mode must be in [padding, 1-padding]
        mode = self.padding + (1.0 - 2 * self.padding) * torch.sigmoid(mode)

        # Concentration must be > 2 for valid parameterization
        log_conc = self.log_conc_min + (self.log_conc_max - self.log_conc_min) * torch.sigmoid(
            log_conc
        )
        concentration = torch.exp(log_conc)

        if anchor is not None:
            # Convert anchor to tensor if it's a numpy array
            if not isinstance(anchor, torch.Tensor):
                anchor = torch.from_numpy(anchor).to(mode.device, dtype=mode.dtype)

            # TODO: Add a check to ensure anchor is within action space bounds

            inv_anchor = self.inverse(anchor)
            mode = mode + inv_anchor  # Use out-of-place operation to avoid modifying view

        if deterministic:
            # Deterministic: just use the mode scaled to action space
            y = mode
            log_prob = torch.zeros_like(mode)
        else:
            # Update distribution and sample
            self._update_distribution(mode=mode, concentration=concentration)
            y = self._beta_dist.rsample()
            log_prob = self._beta_dist.log_prob(y)

        # Scale from [0, 1] to [lb, ub]
        y_scaled = y * self.scale[None, :] + self.lb[None, :]

        # Adjust log_prob for scaling transformation
        log_prob -= torch.log(self.scale[None, :])
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        stats = (
            {"beta_concentration": concentration.prod(dim=-1).mean().item()}
            if concentration is not None
            else {}
        )

        return y_scaled, log_prob, stats

    def _update_distribution(
        self,
        mode: torch.Tensor | float,
        concentration: torch.Tensor | float,
    ) -> None:
        """Update the mode and concentration parameters of the distribution.

        Args:
            mode: New mode parameter in [0, 1] space
            concentration: New concentration parameter
        """
        mode = torch.as_tensor(mode)
        concentration = torch.as_tensor(concentration)

        # Compute alpha, beta from mode and total concentration c
        alpha = 1.0 + mode * (concentration - 2.0)
        beta = 1.0 + (1.0 - mode) * (concentration - 2.0)

        # Update the internal Beta distribution with new parameters
        self._beta_dist = Beta(concentration1=alpha, concentration0=beta, validate_args=None)

    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        return (output_dim, output_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse transformation from [lb, ub] to [0, 1].

        Args:
            x: The input tensor.

        Returns:
            The inverse scaled tensor.
        """
        x = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x

        return (x - self.lb[None, :]) / self.scale[None, :]
