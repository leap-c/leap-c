import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from torch.distributions import (
    AffineTransform,
    Beta,
    Normal,
    TanhTransform,
    TransformedDistribution,
)

from leap_c.torch.nn.bounded_distributions import (
    ModeConcentrationBeta,
    ScaledBeta,
    SquashedGaussian,
)


def _setup_test(
    single_sample: bool, seed: int = None
) -> tuple[np.random.Generator, Box, int, tuple[int, ...], tuple[int, ...]]:
    """Helper method to create necessary components for the tests below.

    In particular, this method creates
     - a 1d `Box` space (i.e., lower and upper bounds have one dimension) with random size
       `event_dim` (i.e., the number of independent dimensions in the distribution support)
     - a random `batch_shape`, a tuple of integers describing how many independent distributions to
        test in a single batch
     - a random `sample_shape`, a tuple of integers describing how many i.i.d. samples to draw from
       each distribution in the batch.
    Drawn samples are expected to have shape `(*sample_shape, *batch_shape, event_dim)`.

    Args:
        single_sample (bool): Whether to test with a single sample or multiple samples. If `True`,
            `sample_shape` is forced to be `()`; otherwise, it will be a tuple of random integers.
        seed (int, optional): RNG seed for reproducibility. If `None`, a random seed will be used.

    Returns:
        tuple[np.random.Generator, Box, int, tuple[int, ...], tuple[int, ...]]: A tuple containing
        the RNG, the created `Box` space, the event dimension, the batch shape, and the sample
        shape.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(1 << 31)))

    event_dim, ndim_batch = map(int, rng.integers(2, 5, size=2))
    batch_shape = tuple(map(int, rng.integers(1, 10, size=ndim_batch)))
    sample_shape = (
        () if single_sample else tuple(map(int, rng.integers(1, 10, size=rng.integers(2, 5))))
    )

    low = -5 - np.abs(rng.normal(scale=5, size=event_dim))
    high = 5 + np.abs(rng.normal(scale=5, size=event_dim))
    space = Box(low, high, dtype=np.float64)

    return rng, space, event_dim, batch_shape, sample_shape


@pytest.mark.parametrize("deterministic", (False, True), ids=["stoch", "deter"])
@pytest.mark.parametrize("single_sample", (False, True), ids=["many", "one"])
def test_squashed_gaussian_anchor(deterministic: bool, single_sample: bool) -> None:
    """Test anchor functionality for `SquashedGaussian` distribution."""
    # NOTE: set seed=4 and remove padding to produce failure
    rng, space, event_dim, batch_shape, sample_shape = _setup_test(single_sample)
    distribution = SquashedGaussian(space, padding=0.0)  # remove paddings to avoid distorsion
    samples: torch.Tensor
    log_prob: torch.Tensor

    # check shapes and within bounds - if deterministic, test that with mean=0 mode is on anchor
    shape = batch_shape + (event_dim,)
    mean = (
        torch.zeros(shape, requires_grad=True)
        if deterministic
        else torch.from_numpy(rng.normal(size=shape)).requires_grad_()
    )
    log_std = (
        torch.from_numpy(rng.normal(size=shape))
        .clamp(distribution.log_std_min, distribution.log_std_max)
        .requires_grad_()
    )
    anchor = torch.from_numpy(rng.uniform(space.low, space.high, size=shape))
    samples, log_prob, _ = distribution(mean, log_std, deterministic, anchor, sample_shape)
    assert samples.shape == sample_shape + shape
    assert log_prob.shape == sample_shape + batch_shape + (1,)
    # NOTE: assert within bounds or close to them, since padding=0
    samples_np = samples.numpy(force=True)
    assert ((samples_np >= space.low) | np.isclose(samples_np, space.low)).all()
    assert ((samples_np <= space.high) | np.isclose(samples_np, space.high)).all()
    if deterministic:
        torch.testing.assert_close(samples, anchor.broadcast_to(samples.shape))

    # test gradients work with anchor
    if deterministic:
        samples.sum().backward(retain_graph=True)
        assert mean.grad is not None and not mean.grad.isnan().any().item()
        assert log_std.grad is None
        mean.grad = None  # reset for next test
        log_prob.sum().backward()
        assert mean.grad is not None and not mean.grad.isnan().any().item()
        assert log_std.grad is None
    else:
        samples.sum().backward(retain_graph=True)
        for t in (mean, log_std):
            assert t.grad is not None and not t.grad.isnan().any().item()
            t.grad = None  # reset for next test
        log_prob.sum().backward()
        for t in (mean, log_std):
            assert t.grad is not None and not t.grad.isnan().any().item()


@pytest.mark.parametrize("single_sample", (False, True), ids=["many", "one"])
def test_squashed_gaussian_log_prob(single_sample: bool) -> None:
    """Test that log_prob computation for `SquashedGaussian` is correct."""
    # NOTE: 267 is a nasty seed for computations
    rng, space, event_dim, batch_shape, sample_shape = _setup_test(single_sample)
    distribution = SquashedGaussian(space, padding=0.0)  # remove padding to avoid distorsion
    samples: torch.Tensor
    log_prob: torch.Tensor

    # generate random Gaussian parameters and samples with associated log probs
    shape = batch_shape + (event_dim,)
    mean = torch.from_numpy(rng.normal(size=shape))
    log_std = torch.from_numpy(rng.normal(size=shape)).clamp(
        distribution.log_std_min, distribution.log_std_max
    )
    samples, log_prob, _ = distribution(mean, log_std, sample_shape=sample_shape)
    log_prob = log_prob.squeeze(-1)

    # create the same distribution with `torch.distributions`
    expected_distribution = TransformedDistribution(
        Normal(mean, log_std.exp()),
        [TanhTransform(), AffineTransform(distribution.loc, distribution.scale)],
    )
    expected_log_prob = expected_distribution.log_prob(samples).sum(-1)
    # NOTE: full expression for debugging purposes
    # std = log_std.exp()
    # samples_bounded = (samples - distribution.loc) / distribution.scale
    # samples_inversed = samples_bounded.arctanh()
    # expected_log_prob = -(
    #     0.5 * log(2 * pi)
    #     + log_std
    #     + 0.5 * ((samples_inversed - mean) / std).square()
    #     + (distribution.scale * (1 - samples_bounded.square()) + 1e-6).log()
    # )

    # assert the log probs match where finite
    # NOTE: `expected_log_prob` can be nonfinite when samples are too close to the boundary due to
    # `padding=0`; but padding > 0 would distort the rest of the log-probabilities.
    finite = expected_log_prob.isfinite()
    torch.testing.assert_close(log_prob[finite], expected_log_prob[finite], atol=1e-5, rtol=1e-1)


@pytest.mark.parametrize("deterministic", (False, True), ids=["stoch", "deter"])
@pytest.mark.parametrize("single_sample", (False, True), ids=["many", "one"])
def test_scaled_beta(deterministic: bool, single_sample: bool) -> None:
    """Sanity checks for the `ScaledBeta` distribution."""
    rng, space, event_dim, batch_shape, sample_shape = _setup_test(single_sample)
    distribution = ScaledBeta(space, padding=0)
    samples: torch.Tensor
    log_prob: torch.Tensor

    # check shapes and within bounds
    shape = batch_shape + (event_dim,)
    log_alpha = torch.from_numpy(rng.normal(size=shape)).requires_grad_()
    log_beta = torch.from_numpy(rng.normal(size=shape)).requires_grad_()
    samples, log_prob, _ = distribution(
        log_alpha, log_beta, deterministic, sample_shape=sample_shape
    )
    assert samples.shape == sample_shape + shape
    assert log_prob.shape == sample_shape + batch_shape + (1,)
    samples_np = samples.numpy(force=True)
    assert all(samples_np[i] in space for i in np.ndindex(sample_shape + batch_shape))

    # test backward of samples and log_prob works
    samples.sum().backward(retain_graph=True)
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()
        t.grad = None  # reset for next test
    log_prob.sum().backward()
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()


@pytest.mark.parametrize("deterministic", (False, True), ids=["stoch", "deter"])
@pytest.mark.parametrize("single_sample", (False, True), ids=["many", "one"])
def test_scaled_beta_anchor(deterministic: bool, single_sample: bool) -> None:
    """Test anchor functionality for `ScaledBeta` distribution."""
    rng, space, event_dim, batch_shape, sample_shape = _setup_test(single_sample)
    distribution = ScaledBeta(space, padding=0)
    samples: torch.Tensor
    log_prob: torch.Tensor

    # check shapes and within bounds - if deterministic, test that with alpha=bet, mode is on anchor
    shape = batch_shape + (event_dim,)
    log_alpha = torch.from_numpy(rng.normal(size=shape)).requires_grad_()
    log_beta = (
        log_alpha.detach().clone().requires_grad_()
        if deterministic
        else torch.from_numpy(rng.normal(size=shape)).requires_grad_()
    )
    anchor = torch.from_numpy(rng.uniform(space.low, space.high, size=shape))
    samples, log_prob, _ = distribution(log_alpha, log_beta, deterministic, anchor, sample_shape)
    assert samples.shape == sample_shape + shape
    assert log_prob.shape == sample_shape + batch_shape + (1,)
    samples_np = samples.numpy(force=True)
    assert all(samples_np[i] in space for i in np.ndindex(sample_shape + batch_shape))
    if deterministic:
        torch.testing.assert_close(samples, anchor.broadcast_to(samples.shape))

    # test gradients work with anchor
    samples.sum().backward(retain_graph=True)
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()
        t.grad = None  # reset for next test
    log_prob.sum().backward()
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()


@pytest.mark.parametrize("single_sample", (False, True), ids=["many", "one"])
def test_scaled_beta_log_prob(single_sample: bool) -> None:
    """Test that log_prob computation for `ScaledBeta` is correct."""
    rng, space, event_dim, batch_shape, sample_shape = _setup_test(single_sample)
    distribution = ScaledBeta(space, padding=0)
    samples: torch.Tensor
    log_prob: torch.Tensor

    # generate random Beta parameters and samples with associated log probs
    shape = batch_shape + (event_dim,)
    log_alpha = torch.from_numpy(rng.normal(size=shape))
    log_beta = torch.from_numpy(rng.normal(size=shape))
    samples, log_prob, _ = distribution(log_alpha, log_beta, sample_shape=sample_shape)

    # create the same distribution with `torch.distributions`
    alpha = 1.0 + log_alpha.clamp(distribution.log_alpha_min, distribution.log_alpha_max).exp()
    beta = 1.0 + log_beta.clamp(distribution.log_beta_min, distribution.log_beta_max).exp()
    expected_distribution = TransformedDistribution(
        Beta(alpha, beta), AffineTransform(distribution.loc, distribution.scale)
    )
    expected_log_prob = expected_distribution.log_prob(samples).sum(-1)

    # assert the log probs match
    torch.testing.assert_close(log_prob.squeeze(-1), expected_log_prob, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("deterministic", (False, True), ids=["stoch", "deter"])
@pytest.mark.parametrize("single_sample", (False, True), ids=["many", "one"])
def test_mode_concentration_beta(deterministic: bool, single_sample: bool) -> None:
    """Sanity checks for the `ModeConcentrationBeta` distribution."""
    rng, space, event_dim, batch_shape, sample_shape = _setup_test(single_sample)
    distribution = ModeConcentrationBeta(space, padding=0)
    samples: torch.Tensor
    log_prob: torch.Tensor

    # check samples lie in the space
    shape = batch_shape + (event_dim,)
    logit_mode = torch.from_numpy(rng.normal(size=shape)).requires_grad_()
    logit_log_conc = torch.from_numpy(rng.normal(size=shape)).requires_grad_()
    samples, log_prob, _ = distribution(
        logit_mode, logit_log_conc, deterministic, sample_shape=sample_shape
    )
    assert samples.shape == sample_shape + shape
    assert log_prob.shape == sample_shape + batch_shape + (1,)
    samples_np = samples.numpy(force=True)
    assert all(samples_np[i] in space for i in np.ndindex(sample_shape + batch_shape))
    if deterministic:  # check that samples and mode are equal if deterministic
        with torch.no_grad():
            eps = distribution.padding
            mode_01 = eps + (1.0 - 2.0 * eps) * torch.sigmoid(logit_mode)
            expected_samples = distribution.loc + distribution.scale * mode_01
            torch.testing.assert_close(samples, expected_samples.broadcast_to(samples.shape))

    # test backward of samples and log_prob works
    if deterministic:
        samples.sum().backward(retain_graph=True)
        assert logit_mode.grad is not None and not logit_mode.grad.isnan().any().item()
        assert logit_log_conc.grad is None
        assert not log_prob.requires_grad
    else:
        samples.sum().backward(retain_graph=True)
        for t in (logit_mode, logit_log_conc):
            assert t.grad is not None and not t.grad.isnan().any().item()
            t.grad = None  # reset for next test
        log_prob.sum().backward()
        for t in (logit_mode, logit_log_conc):
            assert t.grad is not None and not t.grad.isnan().any().item()


@pytest.mark.parametrize("deterministic", (False, True), ids=["stoch", "deter"])
@pytest.mark.parametrize("single_sample", (False, True), ids=["many", "one"])
def test_mode_concentration_beta_anchor(deterministic: bool, single_sample: bool) -> None:
    """Test anchor functionality for `ModeConcentrationBeta` distribution."""
    rng, space, event_dim, batch_shape, sample_shape = _setup_test(single_sample)
    distribution = ModeConcentrationBeta(space, padding=0)
    samples: torch.Tensor
    log_prob: torch.Tensor

    # check shapes and within bounds - if deterministic,  when logit_mode=0, mode is on anchor
    shape = batch_shape + (event_dim,)
    logit_mode = (
        torch.zeros(shape, dtype=torch.float64, requires_grad=True)
        if deterministic
        else torch.from_numpy(rng.normal(size=shape)).requires_grad_()
    )
    logit_log_conc = torch.from_numpy(rng.normal(size=shape)).requires_grad_()
    anchor = torch.from_numpy(rng.uniform(space.low, space.high, size=shape))
    samples, log_prob, _ = distribution(
        logit_mode, logit_log_conc, deterministic, anchor, sample_shape
    )
    assert samples.shape == sample_shape + shape
    assert log_prob.shape == sample_shape + batch_shape + (1,)
    samples_np = samples.numpy(force=True)
    assert all(samples_np[i] in space for i in np.ndindex(sample_shape + batch_shape))
    if deterministic:
        torch.testing.assert_close(samples, anchor.broadcast_to(samples.shape))

    # test backward of samples and log_prob works
    if deterministic:
        samples.sum().backward(retain_graph=True)
        assert logit_mode.grad is not None and not logit_mode.grad.isnan().any().item()
        assert logit_log_conc.grad is None
        assert not log_prob.requires_grad
    else:
        samples.sum().backward(retain_graph=True)
        for t in (logit_mode, logit_log_conc):
            assert t.grad is not None and not t.grad.isnan().any().item()
            t.grad = None  # reset for next test
        log_prob.sum().backward()
        for t in (logit_mode, logit_log_conc):
            assert t.grad is not None and not t.grad.isnan().any().item()


@pytest.mark.parametrize("single_sample", (False, True), ids=["many", "one"])
def test_mode_concentration_beta_log_prob(single_sample: bool) -> None:
    """Test that log_prob computation for `ModeConcentrationBeta` is correct."""
    rng, space, event_dim, batch_shape, sample_shape = _setup_test(single_sample)
    distribution = ModeConcentrationBeta(space, padding=0)
    samples: torch.Tensor
    log_prob: torch.Tensor

    # generate random Beta parameters and samples with associated log probs
    shape = batch_shape + (event_dim,)
    logit_mode = torch.from_numpy(rng.normal(size=shape))
    logit_log_conc = torch.from_numpy(rng.normal(size=shape))
    samples, log_prob, _ = distribution(logit_mode, logit_log_conc, sample_shape=sample_shape)

    # create the same distribution with `torch.distributions`
    mode = distribution.padding + (1.0 - 2.0 * distribution.padding) * logit_mode.sigmoid()
    concentration = (
        distribution.log_conc_min
        + (distribution.log_conc_max - distribution.log_conc_min) * logit_log_conc.sigmoid()
    ).exp()
    alpha = 1.0 + mode * (concentration - 2.0)
    beta = concentration - alpha
    expected_distribution = TransformedDistribution(
        Beta(alpha, beta), AffineTransform(distribution.loc, distribution.scale)
    )
    expected_log_prob = expected_distribution.log_prob(samples).sum(-1)

    # assert the log probs match
    torch.testing.assert_close(log_prob.squeeze(-1), expected_log_prob, atol=1e-6, rtol=1e-6)
