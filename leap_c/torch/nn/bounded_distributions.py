"""Provides a simple distributional layer that allows policies to respect action bounds."""

from abc import abstractmethod
from math import log, pi
from typing import Any, Literal

import torch
from gymnasium.spaces import Box
from numpy import ndarray
from numpy.typing import ArrayLike
from torch.distributions.beta import Beta
from torch.nn.functional import softplus

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
        if space.low.ndim > 1 or not space.is_bounded():
            raise ValueError(
                "The provided space must be a 1d bounded space to support scaling and shifting."
            )
        super().__init__()

    @abstractmethod
    def forward(
        self,
        *defining_parameters: torch.Tensor,
        deterministic: bool = False,
        anchor: torch.Tensor | ndarray | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the distribution.

        Args:
            *defining_parameters: The parameters defining the distribution, such as mean and log
                standard deviation for a Gaussian distribution, or alpha and beta parameters for a
                Beta distribution.
                These parameters are expected to be broadcastable with the bounds of the space
                provided at construction, and have shape `(*, event_dim)` with `event_dim` the
                dimension of said space and `*` any shape for the batched distributions.
            deterministic: If `deterministic` is `True`, the mode of the distribution is used
                instead of sampling.
            anchor: An optional tensor/array to shift the distribution's mean/mode towards. Used in
                residual policies where the output distribution models an offset to the anchor.
                Must be broadcastable to `(*, event_dim)`.
            sample_shape: The shape of the samples to be drawn from the distribution. This allows
                for drawing multiple samples at once from each distribution in the batch.

        Returns:
            A tuple containing the samples with shape `(*sample_shape, *, event_dim)`, the
            corresponding log-probabilities, and a dictionary of stats.
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
        log_std_min: ArrayLike = -4.0,
        log_std_max: ArrayLike = 2.0,
        padding: ArrayLike = 1e-4,
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
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the `SquashedGaussian` distribution.

        Args:
            mean: The mean of the normal distribution with shape `(*, event_dim)`, where `event_dim`
                is the dimension of space provided at construction.
            log_std: The logarithm of the standard deviation of the normal distribution, of the same
                shape as the mean (i.e., assuming independent dimensions). Will be clamped according
                to the attributes of this class. Must be broadcastable to `(*, event_dim)`.
                If `None`, the output is deterministic (no noise added to `mean`; same as for
                `deterministic == True`).
            deterministic: If `True`, the output will just be `spacefitting(tanh(mean))`, with no
                sampling taking place.
            anchor: An optional tensor/array to shift the distribution's mean towards. Used in
                residual policies where the output distribution models an offset to the anchor.
                Must be broadcastable to `(*, event_dim)`.
            sample_shape: The shape of the samples to be drawn from the distribution. This allows
                for drawing multiple samples at once from each distribution in the batch.

        Returns:
            Output sampled from the `SquashedGaussian` with shape `(*sample_shape, *, event_dim)`,
            the log probability of this output, and a statistics dict containing the standard
            deviation.
        """
        if anchor is not None:
            # Convert anchor to tensor if it's a numpy array
            if not isinstance(anchor, torch.Tensor):
                anchor = torch.as_tensor(anchor, dtype=mean.dtype, device=mean.device)

            # ensure anchor is within action space bounds
            assert (anchor >= self.loc - self.scale).all() and (
                anchor <= self.loc + self.scale
            ).all(), "Anchor point must be within action space bounds."

            # Use out-of-place operation to avoid modifying view
            inv_anchor = self.inverse(anchor)
            mean = mean + inv_anchor

        # sample and compute log probability - if deterministic, use the mean and assign 0 log_prob
        stats = {}
        if deterministic or log_std is None:
            out_shape = sample_shape + torch.broadcast_shapes(mean.shape, self.loc.shape)
            y = mean.broadcast_to(out_shape)
            log_prob = mean.new_zeros(()).broadcast_to(out_shape)
        else:
            std = log_std.clamp(self.log_std_min, self.log_std_max).exp()
            out_shape = sample_shape + torch.broadcast_shapes(mean.shape, std.shape, self.loc.shape)
            y = torch.addcmul(mean, std, mean.new_empty(out_shape).normal_())
            log_prob = -0.5 * ((y - mean) / std).square() - log_std - 0.5 * log(2.0 * pi)
            stats["gaussian_unsquashed_std"] = std.prod(dim=-1).mean().item()

        # adjust log_prob with tanh correction (reformulated as softplus for numerical stability;
        # see `torch.distributions.transforms.TanhTransform.log_abs_det_jacobian` for reference) and
        # scaling correction
        tanh_correction = 2.0 * (log(2.0) - y - softplus(-2.0 * y))
        log_prob = (log_prob - tanh_correction).sum(-1, keepdim=True) - self.scale.log().sum()

        # map to desired bounds - pad the scale slightly to avoid rare bound violations
        padded_scale = self.scale * (1.0 - 2.0 * self.padding)
        y_scaled = torch.addcmul(self.loc, padded_scale, y.tanh())
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
        padded_scale = self.scale * (1.0 + 2.0 * self.padding)
        y = (x - self.loc) / padded_scale
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
        log_alpha_min: ArrayLike = -10.0,
        log_beta_min: ArrayLike = -10.0,
        log_alpha_max: ArrayLike = 10.0,
        log_beta_max: ArrayLike = 10.0,
        padding: ArrayLike = 1e-4,
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
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the `ScaledBeta` distribution.

        Note that `alpha` and `beta` are enforced to be `> 1` to ensure concavity.

        Args:
            log_alpha: The logarithm of the alpha parameter of the Beta distribution, with shape
                `(*, event_dim)`, where `event_dim` is the shape of space provided at construction.
            log_beta: The logarithm of the beta parameter of the Beta distribution. Must be
                broadcastable to `(*, event_dim)`.
            deterministic: If `True`, the output will just be `spacefitting(mode)`, with no
                sampling taking place.
            anchor: An optional tensor/array to shift the distribution's mode towards. Used in
                residual policies where the output distribution models an offset to the anchor.
                Must be broadcastable to `(*, event_dim)`.
            sample_shape: The shape of the samples to be drawn from the distribution. This allows
                for drawing multiple samples at once from each distribution in the batch.

        Returns:
            An output sampled from the `ScaledBeta` distribution with shape
            `(*sample_shape, *, event_dim)`, the log probability of this output, and an empty stats
            dict.
        """
        # add 1+eps to ensure concavity and existance of mode (alpha, beta > 1)
        offset = 1.0 + torch.finfo(log_alpha.dtype).eps
        alpha = offset + torch.clamp(log_alpha, self.log_alpha_min, self.log_alpha_max).exp()
        beta = offset + torch.clamp(log_beta, self.log_beta_min, self.log_beta_max).exp()

        if anchor is not None:
            # convert anchor to tensor if it's a numpy array
            if not isinstance(anchor, torch.Tensor):
                anchor = torch.as_tensor(anchor, dtype=alpha.dtype, device=alpha.device)

            # get current mode and translate from [0, 1] to [-inf, inf] logit space
            logit_mode = torch.special.logit((alpha - 1) / (alpha + beta - 2))

            # translate the anchor from [lb, ub] to [0, 1] to [-inf, inf] logit space
            logit_inv_anchor = torch.special.logit(self.inverse(anchor))

            # sum the modes in logit space, and then translate back to [0, 1] space (with padding)
            logit_mode = logit_mode + logit_inv_anchor
            mode = torch.addcmul(self.padding, 1.0 - 2.0 * self.padding, logit_mode.sigmoid())

            # update alpha and beta to reflect the new mode while keeping concentration constant
            concentration = alpha + beta
            alpha = 1.0 + mode * (concentration - 2.0)
            beta = concentration - alpha

        # if deterministic, use the mode as sample; otherwise, create a Beta distribution and sample
        # from it via the reparameterization trick to preserve differentiability
        dist = Beta(alpha, beta)
        y = (
            dist.mode.broadcast_to(sample_shape + alpha.shape)
            if deterministic
            else dist.rsample(sample_shape)
        )
        y_scaled = torch.addcmul(self.loc, self.scale, y)

        # update log probability to reflect scaling
        log_prob = dist.log_prob(y).sum(-1, keepdim=True) - self.scale.log().sum()

        # NOTE: we could return the mean of alpha and beta as stats, but I think they should at
        # least be investigated for each action dimension independently
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
        return (x - self.loc) / self.scale


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

    def __init__(
        self,
        space: Box,
        log_conc_min: ArrayLike = log(2.0),
        log_conc_max: ArrayLike = log(100.0),
        padding: ArrayLike = 1e-4,
    ) -> None:
        """Initialize `ModeConcentrationBeta` distribution.

        Args:
            space: Space the output should fit to.
            log_conc_min: The minimum value for the logarithm of the concentration.
            log_conc_max: The maximum value for the logarithm of the concentration.
            padding: The amount of padding to distance the action of the bounds, when using the
                inverse transformation for the anchoring. This improves numerical stability.

        Raises:
            ValueError: If the provided space is not bounded, or if `log_conc_min` is less than
                `log(2)`.
        """
        dt = torch.get_default_dtype()
        dev = torch.get_default_device()
        log_conc_min = torch.as_tensor(log_conc_min, dtype=dt, device=dev)
        if (log_conc_min < log(2.0)).any().item():
            raise ValueError("`log_conc_min` must be at least `log(2)` to ensure unimodality.")
        super().__init__(space)

        self.register_buffer("scale", torch.as_tensor(space.high - space.low, dtype=dt, device=dev))
        self.register_buffer("loc", torch.as_tensor(space.low, dtype=dt, device=dev))
        self.register_buffer("log_conc_min", log_conc_min)
        self.register_buffer("log_conc_max", torch.as_tensor(log_conc_max, dtype=dt, device=dev))
        self.register_buffer("padding", torch.as_tensor(padding, dtype=dt, device=dev))

    def forward(
        self,
        logit_mode: torch.Tensor,
        logit_log_conc: torch.Tensor,
        deterministic: bool = False,
        anchor: torch.Tensor | ndarray | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from the `ModeConcentrationBeta` distribution.

        Args:
            logit_mode: The logit of the mode of the Beta distribution (unbounded). This number is
                later squashed through a sigmoid to ensure it lies in the valid `[0, 1]` interval.
                Must be broadcastable to `(*, event_dim)`, where `event_dim` is the dimension of
                space provided at construction.
            logit_log_conc: The logit of the logarithm of the concentration parameter, of the same
                shape as the mode (i.e., assuming independent dimensions). The logit is forced to
                the bounded interval `[log_conc_min, log_conc_max]` via a sigmoid; then, it is
                exponentiated.
                Must be also broadcastable to `(*, event_dim)`.
            deterministic: If `True`, the output will just be `spacefitting(mode)`, with no sampling
                taking place.
            anchor: An optional tensor/array to shift the distribution's mode towards. Used in
                residual policies where the output distribution models an offset to the anchor.
                Must be broadcastable to `(*, event_dim)`.
            sample_shape: The shape of the samples to be drawn from the distribution. This allows
                for drawing multiple samples at once from each distribution in the batch.

        Returns:
            An output sampled from the `ModeConcentrationBeta` distribution with shape
            `(*sample_shape, *, event_dim)`, the log probability of this output, and an empty stats
            dict.
        """
        if anchor is not None:
            # convert anchor to tensor if it's a numpy array
            if not isinstance(anchor, torch.Tensor):
                anchor = torch.as_tensor(anchor, dtype=logit_mode.dtype, device=logit_mode.device)

            # ensure anchor is within action space bounds
            assert (anchor >= self.loc).all() and (anchor <= self.loc + self.scale).all(), (
                "Anchor point must be within action space bounds."
            )

            # use out-of-place operation to avoid modifying view
            logit_inv_anchor = torch.special.logit(self.inverse(anchor))
            logit_mode = logit_mode + logit_inv_anchor

        # translate mode from [0, 1] to [padding, 1-padding]
        mode = torch.addcmul(self.padding, 1.0 - 2.0 * self.padding, logit_mode.sigmoid())

        # translate log_conc from [log_conc_min+eps, log_conc_max] and then exponentiate
        # NOTE: concentration must be > 2 to ensure unimodality
        log_conc_min = self.log_conc_min + torch.finfo(logit_log_conc.dtype).eps
        log_conc = torch.addcmul(
            log_conc_min, self.log_conc_max - log_conc_min, logit_log_conc.sigmoid()
        )
        concentration = log_conc.exp()

        # generate sample - if deterministic, use mode directly
        if deterministic:
            out_shape = sample_shape + torch.broadcast_shapes(mode.shape, self.loc.shape)
            y = mode.broadcast_to(out_shape)
            log_prob = mode.new_zeros(()).broadcast_to(out_shape)
        else:
            # sample from Beta distribution
            distr = Beta(alpha := (1.0 + mode * (concentration - 2.0)), concentration - alpha)
            y = distr.rsample(sample_shape)
            log_prob = distr.log_prob(y)

        # scale from [0, 1] to [lb, ub]
        y_scaled = torch.addcmul(self.loc, self.scale, y)

        # update log probability to reflect scaling
        log_prob = log_prob.sum(-1, keepdim=True) - self.scale.log().sum()

        # NOTE: we could return the mean of alpha and beta as stats, but I think they should at
        # least be investigated for each action dimension independently
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
        return (x - self.loc) / self.scale
