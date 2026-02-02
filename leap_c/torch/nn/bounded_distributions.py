"""Provides a simple distributional layer that allows policies to respect action bounds."""

from abc import abstractmethod
from math import log, pi
from typing import Any, Literal

import torch
from gymnasium.spaces import Box
from numpy import ndarray
from numpy.typing import ArrayLike
from torch.distributions.beta import Beta

BoundedDistributionName = Literal["squashed_gaussian", "scaled_beta", "mode_concentration_beta"]


class BoundedDistribution(torch.nn.Module):
    """An abstract class for bounded distributions."""

    def __init__(self, space: Box) -> None:
        """Initializes the bounded distribution.

        Args:
            space: The space the output is bounded to.

        Raises:
            ValueError: If the provided space is not bounded.
        """
        if not space.is_bounded():
            raise ValueError("The provided space must be bounded to support scaling and shifting.")
        super().__init__()

    @abstractmethod
    def forward(
        self, *defining_parameters, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the distribution.

        If `deterministic` is `True`, the mode of the distribution is used instead of sampling.

        Returns:
            A tuple containing the samples, their log probabilities and a dictionary of stats.
        """
        ...

    @abstractmethod
    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        """Returns param size required to define the distribution for the given output dim.

        Args:
            output_dim: The dimensionality of the output space (e.g., action space).

        Returns:
            A tuple of integers, each integer specifying the size of one parameter required to
            define the distribution in the forward pass.
        """
        ...

    @abstractmethod
    def inverse(self, normalized_x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse transformation to the input tensor.

        The inverse transform is meant to translate the input from the bounded space to the squashed
        interval (e.g., `[0,1]` or `[-1,1]`) .

        Args:
            normalized_x: The input tensor.

        Returns:
            The inverse transformed tensor.
        """
        ...


def get_bounded_distribution(name: BoundedDistributionName, **kwargs: Any) -> BoundedDistribution:
    """Create an instance of a bounded distribution by name.

    Args:
        name ({"squashed_gaussian", "scaled_beta", "mode_concentration_beta"}): The name of the
            bounded distribution
        kwargs: Additional keyword arguments passed to the constructor of the bounded distribution.

    Returns:
        BoundedDistribution: The newly created instance.

    Raises:
        ValueError: If the provided name does not correspond to a known bounded distribution.
    """
    match name:
        case "squashed_gaussian":
            return SquashedGaussian(**kwargs)
        case "scaled_beta":
            return ScaledBeta(**kwargs)
        case "mode_concentration_beta":
            return ModeConcentrationBeta(**kwargs)
        case _:
            raise ValueError(f"Unknown bounded distribution: {name}")


class SquashedGaussian(BoundedDistribution):
    """A squashed Gaussian.

    Samples the output from a Gaussian distribution specified by the input, and then squashes the
    result with a `tanh` function. Finally, the output of the `tanh` function is scaled and shifted
    to match the space.

    Can for example be used to enforce certain action bounds of a stochastic policy.

    Attributes:
        scale: The scale of the space-fitting transform.
        loc: The location of the space-fitting transform (for shifting).
        log_std_min: The minimum value for the logarithm of the standard deviation.
        log_std_max: The maximum value for the logarithm of the standard deviation.
        padding: The amount of padding to distance the action of the bounds, when using the inverse
            transformation for the anchoring. This improves numerical stability.
    """

    scale: torch.Tensor
    loc: torch.Tensor
    log_std_min: torch.Tensor
    log_std_max: torch.Tensor
    padding: torch.Tensor

    def __init__(
        self,
        space: Box,
        log_std_min: ArrayLike | float = -4.0,
        log_std_max: ArrayLike | float = 2.0,
        padding: ArrayLike | float = 1e-4,
    ) -> None:
        """Initializes the `SquashedGaussian` module.

        Args:
            space: Space the output should fit to.
            log_std_min: The minimum value for the logarithm of the standard deviation.
            log_std_max: The maximum value for the logarithm of the standard deviation.
            padding: The amount of padding to distance the action of the bounds, when using the
                inverse transformation for the anchoring. This improves numerical stability.

        Raises:
            ValueError: If the provided space is not bounded.
        """
        super().__init__(space)
        dt = torch.get_default_dtype()
        dev = torch.get_default_device()
        loc = (space.high + space.low) / 2.0
        scale = (space.high - space.low) / 2.0
        self.register_buffer("loc", torch.as_tensor(loc, dtype=dt, device=dev))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=dt, device=dev))
        self.register_buffer("log_std_min", torch.as_tensor(log_std_min, dtype=dt, device=dev))
        self.register_buffer("log_std_max", torch.as_tensor(log_std_max, dtype=dt, device=dev))
        self.register_buffer("padding", torch.as_tensor(padding, dtype=dt, device=dev))

    def forward(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor | None = None,
        deterministic: bool = False,
        anchor: torch.Tensor | ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the `SquashedGaussian` distribution.

        Args:
            mean: The mean of the normal distribution.
            log_std: The logarithm of the standard deviation of the normal distribution, of the same
                shape as the mean (i.e., assuming independent dimensions).
                Will be clamped according to the attributes of this class.
                If `None`, the output is deterministic (no noise added to `mean`; same as for
                `deterministic == True`).
            deterministic: If `True`, the output will just be `spacefitting(tanh(mean))`, with no
                sampling taking place.
            anchor: Anchor point to shift the mean. Used for residual policies.

        Returns:
            An output sampled from the `SquashedGaussian`, the log probability of this output
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
                anchor = torch.as_tensor(anchor, dtype=mean.dtype, device=mean.device)

            # ensure anchor is within action space bounds
            assert (anchor >= self.loc - self.scale).all() and (
                anchor <= self.loc + self.scale
            ).all(), "Anchor point must be within action space bounds."

            inv_anchor = self.inverse(anchor)
            mean = mean + inv_anchor  # Use out-of-place operation to avoid modifying view

        if deterministic or std is None:
            y = mean
        else:
            # reparameterization trick
            y = mean + std * torch.randn_like(mean)

        if std is not None:
            log_prob = -0.5 * ((y - mean) / std).square() - log_std - 0.5 * log(2 * pi)
        else:
            # Deterministic: log_prob is 0 in the unbounded space (delta distribution)
            log_prob = torch.zeros_like(mean)

        y = torch.tanh(y)

        log_prob -= torch.log(self.scale * (1 - y.square()) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        y_scaled = y * self.scale + self.loc

        stats = (
            {"gaussian_unsquashed_std": std.prod(dim=-1).mean().item()} if std is not None else {}
        )

        return y_scaled, log_prob, stats

    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        return output_dim, output_dim

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse transformation to the input tensor.

        The inverse transformation is a descale and then `arctanh`. For numerical stability, the
        input is slightly padded away (by `padding`) from the bounds before applying `arctanh`.

        Args:
            x: The input tensor.

        Returns:
            The inverse squashed tensor, scaled and shifted to match the action space.
        """
        abs_padding = self.scale * self.padding
        y = (x - self.loc) / (self.scale + 2 * abs_padding)
        return torch.arctanh(y)


class ScaledBeta(BoundedDistribution):
    """A unimodal scaled Beta distribution.

    Samples the output from a Beta distribution specified by the input, and then scales and shifts
    the result to match the space. Unomodality is ensured by enforcing `alpha, beta > 1`.

    Can for example be used to enforce certain action bounds of a stochastic policy.

    Attributes:
        scale: The scale of the space-fitting transform (for scaling).
        loc: The location of the space-fitting transform (for shifting).
        log_alpha_min: The minimum value for the logarithm of the alpha parameter.
        log_beta_min: The minimum value for the logarithm of the beta parameter.
        log_alpha_max: The maximum value for the logarithm of the alpha parameter.
        log_beta_max: The maximum value for the logarithm of the beta parameter.
        padding: The amount of padding to distance the action of the bounds, when using the inverse
            transformation for the anchoring. This improves numerical stability.
    """

    scale: torch.Tensor
    loc: torch.Tensor
    log_alpha_min: torch.Tensor
    log_alpha_max: torch.Tensor
    log_beta_min: torch.Tensor
    log_beta_max: torch.Tensor
    padding: torch.Tensor

    def __init__(
        self,
        space: Box,
        log_alpha_min: ArrayLike | float = -10.0,
        log_beta_min: ArrayLike | float = -10.0,
        log_alpha_max: ArrayLike | float = 10.0,
        log_beta_max: ArrayLike | float = 10.0,
        padding: ArrayLike | float = 1e-4,
    ) -> None:
        """Initializes the `ScaledBeta` module.

        Args:
            space: Space the output should fit to.
            log_alpha_min: The minimum value for the logarithm of the alpha parameter.
            log_beta_min: The minimum value for the logarithm of the beta parameter.
            log_alpha_max: The maximum value for the logarithm of the alpha parameter.
            log_beta_max: The maximum value for the logarithm of the beta parameter.
            padding: The amount of padding to distance the action of the bounds, when using the
                inverse transformation for the anchoring. This improves numerical stability.

        Raises:
            ValueError: If the provided space is not bounded.
        """
        super().__init__(space)
        dt = torch.get_default_dtype()
        dev = torch.get_default_device()
        self.register_buffer("scale", torch.as_tensor(space.high - space.low, dtype=dt, device=dev))
        self.register_buffer("loc", torch.as_tensor(space.low, dtype=dt, device=dev))
        self.register_buffer("log_alpha_min", torch.as_tensor(log_alpha_min, dtype=dt, device=dev))
        self.register_buffer("log_beta_min", torch.as_tensor(log_beta_min, dtype=dt, device=dev))
        self.register_buffer("log_alpha_max", torch.as_tensor(log_alpha_max, dtype=dt, device=dev))
        self.register_buffer("log_beta_max", torch.as_tensor(log_beta_max, dtype=dt, device=dev))
        self.register_buffer("padding", torch.as_tensor(padding, dtype=dt, device=dev))

    def forward(
        self,
        log_alpha: torch.Tensor,
        log_beta: torch.Tensor,
        deterministic: bool = False,
        anchor: torch.Tensor | ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the `ScaledBeta` distribution.

        Note that `alpha` and `beta` are enforced to be `> 1` to ensure concavity.

        Args:
            log_alpha: The logarithm of the alpha parameter of the Beta distribution.
            log_beta: The logarithm of the beta parameter of the Beta distribution.
            deterministic: If `True`, the output will just be `spacefitting(mode)`, with no
                sampling taking place.
            anchor: If provided, the Beta distribution's mode is centered around this anchor point.
                This is useful for action noise where the MPC output serves as the anchor.

        Returns:
            An output sampled from the `ScaledBeta` distribution, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        # add 1 to ensure concavity
        alpha = 1.0 + torch.clamp(log_alpha, self.log_alpha_min, self.log_alpha_max).exp()
        beta = 1.0 + torch.clamp(log_beta, self.log_beta_min, self.log_beta_max).exp()

        if anchor is not None:
            # get current mode and translate from [0, 1] to [-inf, inf] logit space
            logit_mode = torch.special.logit((alpha - 1) / (alpha + beta - 2))

            # translate the anchor from [lb, ub] to [0, 1] to [-inf, inf] logit space
            logit_inv_anchor = torch.special.logit(self.inverse(anchor))

            # sum the modes in logit space, and then translate back to [0, 1] space (with padding)
            logit_mode = logit_mode + logit_inv_anchor
            mode = self.padding + (1.0 - 2.0 * self.padding) * torch.sigmoid(logit_mode)

            # update alpha and beta to reflect the new mode while keeping concentration constant
            concentration_m2 = alpha + beta - 2.0
            alpha = 1.0 + mode * concentration_m2
            beta = 1.0 + (1.0 - mode) * concentration_m2

        # create Beta distribution and sample from it or use its mode (deterministic case)
        dist = Beta(alpha, beta)
        y = dist.mode if deterministic else dist.rsample()  # reparameterization trick
        y_scaled = y * self.scale + self.loc

        # update log probability to reflect scaling
        log_prob = dist.log_prob(y)
        log_prob -= torch.log(self.scale)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # We could return the mean of alpha and beta as stats, but I think they should at least be
        # investigated for each action dimension independently
        return y_scaled, log_prob, {}

    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        return output_dim, output_dim

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse transformation from `[lb, ub]` to `[0, 1]`.

        Args:
            x: The input tensor.

        Returns:
            The inverse scaled tensor.
        """
        return (torch.as_tensor(x) - self.loc) / self.scale


class ModeConcentrationBeta(BoundedDistribution):
    """Beta distribution parameterized by mode and total concentration.

    This distribution is parameterized similarly to `SquashedGaussian`:
    - logit_mode: The logit of the mode of the distribution
    - log_conc: The logarithm of the concentration parameter

    Attributes:
        scale: The scale of the space-fitting transform (for scaling).
        loc: The location of the space-fitting transform (for shifting).
        log_conc_min: The minimum value for the logarithm of the concentration.
        log_conc_max: The maximum value for the logarithm of the concentration.
        padding: The amount of padding to distance the action of the bounds, when using the inverse
            transformation for the anchoring. This improves numerical stability.
    """

    scale: torch.Tensor
    loc: torch.Tensor
    log_conc_min: torch.Tensor
    log_conc_max: torch.Tensor
    padding: torch.Tensor

    _beta_dist: Beta

    def __init__(
        self,
        space: Box,
        log_conc_min: float = log(2.0),
        log_conc_max: float = log(100.0),
        padding: float = 1e-4,
    ) -> None:
        """Initialize `ModeConcentrationBeta` distribution.

        Args:
            space: Space the output should fit to.
            log_conc_min: The minimum value for the logarithm of the concentration.
            log_conc_max: The maximum value for the logarithm of the concentration.
            padding: The amount of padding to distance the action of the bounds, when using the
                inverse transformation for the anchoring. This improves numerical stability.

        Raises:
            ValueError: If the provided space is not bounded.
        """
        super().__init__(space)
        dt = torch.get_default_dtype()
        dev = torch.get_default_device()
        self.register_buffer("scale", torch.as_tensor(space.high - space.low, dtype=dt, device=dev))
        self.register_buffer("loc", torch.as_tensor(space.low, dtype=dt, device=dev))
        self.register_buffer("log_conc_min", torch.as_tensor(log_conc_min, dtype=dt, device=dev))
        self.register_buffer("log_conc_max", torch.as_tensor(log_conc_max, dtype=dt, device=dev))
        self.register_buffer("padding", torch.as_tensor(padding, dtype=dt, device=dev))

    def forward(
        self,
        logit_mode: torch.Tensor,
        logit_log_conc: torch.Tensor,
        deterministic: bool = False,
        anchor: torch.Tensor | ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the `ModeConcentrationBeta` distribution.

        Args:
            logit_mode: The logit of the mode of the Beta distribution (unbounded). This number is
                later squashed through a sigmoid to ensure it lies in the valid `[0, 1]` interval.
            logit_log_conc: The logit of the logarithm of the concentration parameter, of the same
                shape as the mode (i.e., assuming independent dimensions). The logit is forced to
                the bounded interval `[log_conc_min, log_conc_max]` via a sigmoid; then, it is
                exponentiated.
            deterministic: If `True`, the output will just be `spacefitting(mode)`, with no sampling
                taking place.
            anchor: Anchor point to shift the mode. Used for residual policies.

        Returns:
            An output sampled from the `ModeConcentrationBeta`, the log probability of this output
            and a statistics dict.
        """
        if anchor is not None:
            # Convert anchor to tensor if it's a numpy array
            if not isinstance(anchor, torch.Tensor):
                anchor = torch.as_tensor(anchor, dtype=logit_mode.dtype, device=logit_mode.device)

            # ensure anchor is within action space bounds
            assert (anchor >= self.loc).all() and (anchor <= self.loc + self.scale).all(), (
                "Anchor point must be within action space bounds."
            )

            # Use out-of-place operation to avoid modifying view
            logit_inv_anchor = torch.special.logit(self.inverse(anchor))
            logit_mode = logit_mode + logit_inv_anchor

        # Mode must be in [padding, 1-padding]
        mode = self.padding + (1.0 - 2.0 * self.padding) * torch.sigmoid(logit_mode)

        # Concentration must be > 2 for valid parameterization
        log_conc = self.log_conc_min + (self.log_conc_max - self.log_conc_min) * torch.sigmoid(
            logit_log_conc
        )
        concentration = torch.exp(log_conc)

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
        y_scaled = y * self.scale + self.loc

        # Adjust log_prob for scaling transformation
        log_prob -= torch.log(self.scale)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        stats = (
            {"beta_concentration": concentration.prod(dim=-1).mean().item()}
            if concentration is not None
            else {}
        )

        return y_scaled, log_prob, stats

    def _update_distribution(
        self, mode: torch.Tensor | float, concentration: torch.Tensor | float
    ) -> None:
        """Update the mode and concentration parameters of the distribution.

        Args:
            mode: New mode parameter in `[0, 1]` space.
            concentration: New concentration parameter.
        """
        mode = torch.as_tensor(mode)
        concentration = torch.as_tensor(concentration)

        # Compute alpha, beta from mode and total concentration c
        alpha = 1.0 + mode * (concentration - 2.0)
        beta = 1.0 + (1.0 - mode) * (concentration - 2.0)

        # Update the internal Beta distribution with new parameters
        self._beta_dist = Beta(concentration1=alpha, concentration0=beta)

    def parameter_size(self, output_dim: int) -> tuple[int, ...]:
        return output_dim, output_dim

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse transformation from `[lb, ub]` to `[0, 1]`.

        Args:
            x: The input tensor.

        Returns:
            The inverse scaled tensor.
        """
        return (torch.as_tensor(x) - self.loc) / self.scale
