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
            log_prob = -0.5 * ((y - mean) / std).pow(2) - log_std - np.log(np.sqrt(2) * np.pi)
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


class ModeConcentrationBeta(Beta):
    """Beta distribution parameterized by mode and total concentration."""

    @staticmethod
    def compute_alpha_beta(
        mode: torch.Tensor | float,
        concentration: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute alpha and beta parameters from mode and concentration.

        Args:
            mode: Mode parameter (must be in [0, 1])
            concentration: Total concentration parameter (must be > 2)

        Returns:
            Tuple of (alpha, beta) parameters for Beta distribution
        """
        mode = torch.as_tensor(mode)
        concentration = torch.as_tensor(concentration)

        # Compute alpha, beta from mode and total concentration c
        alpha = 1.0 + mode * (concentration - 2.0)
        beta = 1.0 + (1.0 - mode) * (concentration - 2.0)

        return alpha, beta

    def __init__(
        self,
        space: spaces.Box,
        mode: torch.Tensor | float | None = None,
        concentration: torch.Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Initialize ModeConcentrationBeta distribution.

        Args:
            space: Space the output should fit to.
            mode: Mode parameter (must be in [0, 1])
            concentration: Total concentration parameter (must be > 2)
            validate_args: Whether to validate input arguments
        """
        mode = torch.tensor(0.5) if mode is None else torch.as_tensor(mode).clone()
        concentration = (
            torch.tensor(5.0) if concentration is None else torch.as_tensor(concentration).clone()
        )

        self._lb = torch.tensor(space.low, dtype=torch.float32)
        self._ub = torch.tensor(space.high, dtype=torch.float32)
        self._scale = self._ub - self._lb
        self._mode = mode
        self._concentration = concentration

        # Compute alpha, beta from mode in [0, 1] space and total concentration
        alpha, beta = ModeConcentrationBeta.compute_alpha_beta(
            mode=mode, concentration=concentration
        )

        # Initialize parent Beta distribution first
        super().__init__(
            concentration1=alpha,
            concentration0=beta,
            validate_args=validate_args,
        )

    @property
    def mode(self) -> torch.Tensor:
        """The mode of the distribution in [0, 1] space."""
        return self._mode

    @property
    def concentration(self) -> torch.Tensor:
        """The concentration of the distribution (c = alpha + beta)."""
        return self._concentration

    @property
    def lb(self) -> torch.Tensor:
        """Lower bound for sample rescaling."""
        return self._lb

    @property
    def ub(self) -> torch.Tensor:
        """Upper bound for sample rescaling."""
        return self._ub

    def set_mode_from_nominal_parameters(self, mode: torch.Tensor | float) -> None:
        self._mode = self.inverse(mode)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample using reparameterization trick in [0, 1] and rescale to [lb, ub].

        Args:
            sample_shape: Shape of samples to draw

        Returns:
            Samples rescaled to [lb, ub] range with gradient support
        """
        # Get reparameterized samples from parent Beta distribution (in [0, 1])
        samples_01 = super().rsample(sample_shape=sample_shape)

        # Rescale samples to [lb, ub]
        return self._lb + samples_01 * (self._scale)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of samples in [lb, ub] range.

        Args:
            value: Sample values in [lb, ub] range

        Returns:
            Log probabilities adjusted for the transformation
        """
        # Compute log_prob in [0, 1] space
        log_prob_01 = super().log_prob(
            value=torch.clamp(
                input=self.inverse(value),
                min=0.0,
                max=1.0,
            )
        )

        # Adjust for Jacobian of transformation: dx/dy = 1 / (ub - lb)
        # log_prob_rescaled = log_prob_01 - log|ub - lb|
        log_jacobian = -torch.log(self._scale).sum()
        log_prob_rescaled = log_prob_01 + log_jacobian

        return log_prob_rescaled

    def update_parameters(
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

        # Compute new alpha, beta from mode in [0, 1] space and concentration > 2
        alpha, beta = ModeConcentrationBeta.compute_alpha_beta(
            mode=mode,
            concentration=concentration,
        )

        # Reinitialize the parent Beta distribution with new parameters
        Beta.__init__(self, concentration1=alpha, concentration0=beta)

        # Update stored mode and concentration (in [0, 1] space)
        self._mode = mode
        self._concentration = concentration

    def __call__(
        self,
        mode: torch.Tensor,
        concentration: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the distribution.

        If `deterministic` is True, the mode of the distribution is used instead of
        sampling.

        Args:
            mode: Mode parameter in [0, 1] space
            concentration: Concentration parameter > 2
            deterministic: If True, the output will just be mode, no sampling is taking place.

        Returns:
            A tuple containing the samples, their log_prob and a dictionary of stats.
        """
        return self.forward(
            mode=mode,
            concentration=concentration,
            deterministic=deterministic,
        )

    def forward(
        self,
        mode: torch.Tensor,
        concentration: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the distribution.

        If `deterministic` is True, the mode of the distribution is used instead of
        sampling.

        Returns:
            A tuple containing the samples, their log_prob and a dictionary of stats.
        """
        self.update_parameters(mode=mode, concentration=concentration)

        sample = self.rsample() if not deterministic else self._mode

        log_prob = self.log_prob(sample)
        return sample, log_prob, {}

    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        """Returns param size required to define the distribution for the given output dim.

        Args:
            output_dim: The dimensionality of the output space (e.g., action space).

        Returns:
            A tuple of integers, each integer specifying the size of one
            parameter required to define the distribution in the forward pass.
        """
        return (output_dim, output_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse transformation from [lb, ub] to [0, 1].

        Args:
            x: The input tensor.

        Returns:
            The inverse scaled tensor.
        """
        x = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x

        return (x - self._lb[None, :]) / self._scale[None, :]
