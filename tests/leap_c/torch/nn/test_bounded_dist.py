import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from torch.distributions import AffineTransform, Beta, TransformedDistribution

from leap_c.torch.nn.bounded_distributions import (
    ModeConcentrationBeta,
    ScaledBeta,
    SquashedGaussian,
)


@pytest.mark.parametrize("deterministic", (False, True))
def test_squashed_gaussian_anchor(deterministic: bool) -> None:
    """Test anchor functionality for `SquashedGaussian` distribution."""
    rng = np.random.default_rng()
    torch.manual_seed(int(rng.integers(0, 1 << 31)))

    # generate random space and associated distribution
    ndim, n_samples = map(int, rng.integers(2, 10, size=2))
    low = -5 - np.abs(rng.normal(scale=5, size=ndim))
    high = 5 + np.abs(rng.normal(scale=5, size=ndim))
    space = Box(low, high, dtype=np.float64)
    distribution = SquashedGaussian(space, padding=0.0)
    samples: torch.Tensor
    log_prob: torch.Tensor

    # check shapes and within bounds - if deterministic, test that with mean=0 mode is on anchor
    mean = (
        torch.zeros((n_samples, ndim), requires_grad=True)
        if deterministic
        else torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    )
    log_std = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    anchor = torch.from_numpy(rng.uniform(low, high, size=(n_samples, ndim)))
    samples, log_prob, _ = distribution(mean, log_std, deterministic, anchor)
    assert all(s in space for s in samples.numpy(force=True))
    assert log_prob.shape == (n_samples, 1)
    if deterministic:
        torch.testing.assert_close(samples, anchor)

    # test gradients work with anchor
    if deterministic:
        samples.sum().backward(retain_graph=True)
        assert mean.grad is not None and not mean.grad.isnan().any().item()
        assert log_std.grad is None
        mean.grad = None  # reset for next test
        log_prob.sum().backward()
        for t in (mean, log_std):
            assert t.grad is not None and not t.grad.isnan().any().item()
    else:
        samples.sum().backward(retain_graph=True)
        for t in (mean, log_std):
            assert t.grad is not None and not t.grad.isnan().any().item()
            t.grad = None  # reset for next test
        log_prob.sum().backward()
        for t in (mean, log_std):
            assert t.grad is not None and not t.grad.isnan().any().item()


@pytest.mark.parametrize("deterministic", (False, True))
def test_scaled_beta(deterministic: bool) -> None:
    """Sanity checks for the `ScaledBeta` distribution."""
    rng = np.random.default_rng()
    torch.manual_seed(int(rng.integers(0, 1 << 31)))

    # generate random space and associated distribution
    ndim, n_samples = map(int, rng.integers(2, 10, size=2))
    low = -5 - np.abs(rng.normal(scale=5, size=ndim))
    high = 5 + np.abs(rng.normal(scale=5, size=ndim))
    space = Box(low, high, dtype=np.float64)
    distribution = ScaledBeta(space, padding=0)
    samples: torch.Tensor
    log_prob: torch.Tensor

    # check shapes and within bounds
    log_alpha = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    log_beta = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    samples, log_prob, _ = distribution(log_alpha, log_beta, deterministic=deterministic)
    assert all(s in space for s in samples.numpy(force=True))
    assert log_prob.shape == (n_samples, 1)

    # test backward of samples and log_prob works
    samples.sum().backward(retain_graph=True)
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()
        t.grad = None  # reset for next test
    log_prob.sum().backward()
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()


@pytest.mark.parametrize("deterministic", (False, True))
def test_scaled_beta_anchor(deterministic: bool) -> None:
    """Test anchor functionality for `ScaledBeta` distribution."""
    rng = np.random.default_rng()
    torch.manual_seed(int(rng.integers(0, 1 << 31)))

    # generate random space and associated distribution
    ndim, n_samples = map(int, rng.integers(2, 10, size=2))
    low = -5 - np.abs(rng.normal(scale=5, size=ndim))
    high = 5 + np.abs(rng.normal(scale=5, size=ndim))
    space = Box(low, high, dtype=np.float64)
    distribution = ScaledBeta(space, padding=0)  # remove paddings to avoid distorsion
    samples: torch.Tensor
    log_prob: torch.Tensor

    # check shapes and within bounds - if deterministic, test that with alpha=bet, mode is on anchor
    log_alpha = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    log_beta = (
        log_alpha.detach().clone().requires_grad_()
        if deterministic
        else torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    )
    anchor = torch.from_numpy(rng.uniform(low, high, size=(n_samples, ndim)))
    samples, log_prob, _ = distribution(log_alpha, log_beta, deterministic, anchor)
    assert all(s in space for s in samples.numpy(force=True))
    assert log_prob.shape == (n_samples, 1)
    if deterministic:
        torch.testing.assert_close(samples, anchor)

    # test gradients work with anchor
    samples.sum().backward(retain_graph=True)
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()
        t.grad = None  # reset for next test
    log_prob.sum().backward()
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()


def test_scaled_beta_log_prob() -> None:
    """Test that log_prob computation for `ScaledBeta` is correct."""
    rng = np.random.default_rng()
    torch.manual_seed(int(rng.integers(0, 1 << 31)))

    # generate random space and associated distribution
    ndim, n_samples = map(int, rng.integers(2, 10, size=2))
    low = -5 - np.abs(rng.normal(scale=5, size=ndim))
    high = 5 + np.abs(rng.normal(scale=5, size=ndim))
    space = Box(low, high, dtype=np.float64)
    distribution = ScaledBeta(space, padding=0)  # remove paddings to avoid distorsion

    # generate random Gaussian parameters and samples with associated log probs
    log_alpha = torch.from_numpy(rng.normal(size=(n_samples, ndim)))
    log_beta = torch.from_numpy(rng.normal(size=(n_samples, ndim)))
    samples, log_prob, _ = distribution(log_alpha, log_beta)

    # create the same distribution with `torch.distributions`
    alpha = 1.0 + log_alpha.clamp(distribution.log_alpha_min, distribution.log_alpha_max).exp()
    beta = 1.0 + log_beta.clamp(distribution.log_beta_min, distribution.log_beta_max).exp()
    expected_distribution = TransformedDistribution(
        Beta(alpha, beta), AffineTransform(distribution.loc, distribution.scale)
    )
    expected_log_prob = expected_distribution.log_prob(samples).sum(-1)

    # assert the log probs match
    torch.testing.assert_close(log_prob.squeeze(-1), expected_log_prob, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("deterministic", (False, True))
def test_mode_concentration_beta(deterministic: bool) -> None:
    """Sanity checks for the `ModeConcentrationBeta` distribution."""
    rng = np.random.default_rng()
    torch.manual_seed(int(rng.integers(0, 1 << 31)))

    # generate random space and associated distribution
    ndim, n_samples = map(int, rng.integers(2, 10, size=2))
    low = -5 - np.abs(rng.normal(scale=5, size=ndim))
    high = 5 + np.abs(rng.normal(scale=5, size=ndim))
    space = Box(low, high, dtype=np.float64)
    distribution = ModeConcentrationBeta(space)
    samples: torch.Tensor
    log_prob: torch.Tensor

    # check samples lie in the space
    logit_mode = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    logit_log_conc = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    samples, log_prob, _ = distribution(logit_mode, logit_log_conc, deterministic)
    assert all(s in space for s in samples.numpy(force=True))
    assert log_prob.shape == (n_samples, 1)
    if deterministic:
        # check that samples and mode are equal if deterministic
        with torch.no_grad():
            eps = distribution.padding
            mode_01 = eps + (1.0 - 2.0 * eps) * torch.sigmoid(logit_mode)
            expected_samples = distribution.loc + distribution.scale * mode_01
            torch.testing.assert_close(samples, expected_samples)

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


@pytest.mark.parametrize("deterministic", (False, True))
def test_mode_concentration_beta_anchor(deterministic: bool) -> None:
    """Test anchor functionality for `ModeConcentrationBeta` distribution."""
    rng = np.random.default_rng()
    torch.manual_seed(int(rng.integers(0, 1 << 31)))

    # generate random space and associated distribution
    ndim, n_samples = map(int, rng.integers(2, 10, size=2))
    low = -5 - np.abs(rng.normal(scale=5, size=ndim))
    high = 5 + np.abs(rng.normal(scale=5, size=ndim))
    space = Box(low, high, dtype=np.float64)
    distribution = ModeConcentrationBeta(space, padding=0)  # remove paddings to avoid distorsion
    samples: torch.Tensor
    log_prob: torch.Tensor

    # check shapes and within bounds - if deterministic,  when logit_mode=0, mode is on anchor
    logit_mode = (
        torch.zeros((n_samples, ndim), dtype=torch.float64, requires_grad=True)
        if deterministic
        else torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    )
    logit_log_conc = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    anchor = torch.from_numpy(rng.uniform(low, high, size=(n_samples, ndim)))
    samples, log_prob, _ = distribution(logit_mode, logit_log_conc, deterministic, anchor)
    assert all(s in space for s in samples.numpy(force=True))
    assert log_prob.shape == (n_samples, 1)
    if deterministic:
        torch.testing.assert_close(samples, anchor)

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


def test_mode_concentration_beta_log_prob() -> None:
    """Test that log_prob computation for `ModeConcentrationBeta` is correct."""
    rng = np.random.default_rng()
    torch.manual_seed(int(rng.integers(0, 1 << 31)))

    # generate random space and associated distribution
    ndim, n_samples = map(int, rng.integers(2, 10, size=2))
    low = -5 - np.abs(rng.normal(scale=5, size=ndim))
    high = 5 + np.abs(rng.normal(scale=5, size=ndim))
    space = Box(low, high, dtype=np.float64)
    distribution = ModeConcentrationBeta(space, padding=0)  # remove paddings to avoid distorsion

    # generate random Gaussian parameters and samples with associated log probs
    logit_mode = torch.from_numpy(rng.normal(size=(n_samples, ndim)))
    logit_log_conc = torch.from_numpy(rng.normal(size=(n_samples, ndim)))
    samples, log_prob, _ = distribution(logit_mode, logit_log_conc)

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
