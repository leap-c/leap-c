import numpy as np
import torch
from gymnasium.spaces import Box

from leap_c.torch.nn.bounded_distributions import (
    ModeConcentrationBeta,
    ScaledBeta,
    SquashedGaussian,
)


def test_squashed_gaussian_anchor():
    """Test anchor functionality for SquashedGaussian distribution."""
    test_space = Box(np.array([-1.0, -2.0], np.float32), np.array([1.0, 2.0], np.float32))
    dist = SquashedGaussian(test_space)

    # Test deterministic sampling with anchor
    mean = torch.tensor([[0.0, 0.0]], requires_grad=True)
    log_std = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    anchor = torch.tensor([0.5, 1.0])

    samples, log_prob, _ = dist(mean, log_std, deterministic=True, anchor=anchor)

    # With anchor and deterministic, mean=0 should result in anchor value
    assert samples.shape == (1, 2)
    assert log_prob.shape == (1, 1)
    assert torch.allclose(samples[0], anchor, atol=1e-3)

    # Test gradients work with anchor in deterministic mode
    samples.sum().backward()
    assert mean.grad is not None and not torch.any(torch.isnan(mean.grad))

    # Test stochastic sampling with anchor
    mean = torch.tensor([[0.0, 0.0]], requires_grad=True)
    log_std = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    torch.manual_seed(42)
    samples_stochastic, _, _ = dist(mean, log_std, deterministic=False, anchor=anchor)

    # Verify gradients work in stochastic mode
    samples_stochastic.sum().backward()
    assert mean.grad is not None and not torch.any(torch.isnan(mean.grad))
    assert log_std.grad is not None and not torch.any(torch.isnan(log_std.grad))

    # Samples should be in valid range
    assert samples_stochastic.shape == (1, 2)
    assert torch.all(samples_stochastic >= torch.from_numpy(test_space.low))
    assert torch.all(samples_stochastic <= torch.from_numpy(test_space.high))


def test_scaled_beta():
    """Sanity checks for the ScaledBeta distribution."""
    test_space = Box(
        np.array([-10.0, -15.0, 31.0, 3.0], np.float32),
        np.array([-5.0, 20.0, 42.0, 4.0], np.float32),
    )
    dist = ScaledBeta(test_space)

    # Define parameters
    def create_alpha_beta_tensors():
        alpha = torch.tensor([[1.0, -2.0, -3.0, -100.0], [3.0, 4.0, -5.0, 100.0]])
        beta = torch.tensor([[4.0, 3.0, 2.0, -100.0], [2.0, -1.0, 0.0, 100.0]])
        alpha.requires_grad = True
        beta.requires_grad = True
        return alpha, beta

    alpha, beta = create_alpha_beta_tensors()
    samples, log_prob, _ = dist(alpha, beta, deterministic=False)

    # Check shapes
    assert samples.shape == (2, 4)
    assert log_prob.shape == (2, 1)

    # Check that samples are within bounds
    samples_npy = samples.detach().numpy()
    for s in samples_npy:
        assert s in test_space

    # test backward of log_prob works
    log_prob.sum().backward()
    assert alpha.grad is not None and not torch.any(torch.isnan(alpha.grad))
    assert beta.grad is not None and not torch.any(torch.isnan(beta.grad))

    alpha, beta = create_alpha_beta_tensors()
    samples, log_prob, _ = dist(alpha, beta, deterministic=False)
    # test backward of samples works
    samples.sum().backward()
    assert alpha.grad is not None and not torch.any(torch.isnan(alpha.grad))
    assert beta.grad is not None and not torch.any(torch.isnan(beta.grad))

    # Test deterministic sampling (mode)
    alpha, beta = create_alpha_beta_tensors()
    mode_samples, mode_log_prob, _ = dist(alpha, beta, deterministic=True)

    # Check that mode samples are within bounds and their shapes
    assert mode_samples.shape == (2, 4)
    assert mode_log_prob.shape == (2, 1)
    mode_samples_npy = mode_samples.detach().numpy()
    for s in mode_samples_npy:
        assert s in test_space

    # Test mode_sample backward works
    mode_samples.sum().backward()
    assert alpha.grad is not None and not torch.any(torch.isnan(alpha.grad))
    assert beta.grad is not None and not torch.any(torch.isnan(beta.grad))

    alpha, beta = create_alpha_beta_tensors()
    mode_samples, mode_log_prob, _ = dist(alpha, beta, deterministic=True)

    # Test mode_log_prob backward works
    mode_log_prob.sum().backward()
    assert alpha.grad is not None and not torch.any(torch.isnan(alpha.grad))
    assert beta.grad is not None and not torch.any(torch.isnan(beta.grad))


def test_scaled_beta_anchor():
    """Test anchor functionality for `ScaledBeta` distribution."""
    rng = np.random.default_rng()

    # generate random space and associated distribution
    ndim, n_samples = map(int, rng.integers(2, 10, size=2))
    low = -5 - np.abs(rng.normal(scale=5, size=ndim))
    high = 5 + np.abs(rng.normal(scale=5, size=ndim))
    space = Box(low, high, dtype=np.float64)
    distribution = ScaledBeta(space, padding=0)  # remove paddings to avoid distorsion

    # test deterministic sampling with anchor - when alpha=beta, mode is on anchor
    log_alpha = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    log_beta = log_alpha.detach().clone().requires_grad_()
    anchor = torch.from_numpy(rng.uniform(low, high, size=(n_samples, ndim)))
    samples: torch.Tensor
    log_prob: torch.Tensor
    samples, log_prob, _ = distribution(log_alpha, log_beta, True, anchor)
    assert all(s in space for s in samples.numpy(force=True))
    assert log_prob.shape == (n_samples, 1)
    torch.testing.assert_close(samples, anchor)

    # test gradients work with anchor
    samples.sum().backward(retain_graph=True)
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()
        t.grad = None  # reset for next test
    log_prob.sum().backward()
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()

    # test stochastic sampling with anchor
    log_alpha = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    log_beta = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    anchor = torch.from_numpy(rng.uniform(low, high, size=(n_samples, ndim)))
    samples, log_prob, _ = distribution(log_alpha, log_beta, anchor=anchor)
    assert all(s in space for s in samples.numpy(force=True))
    assert log_prob.shape == (n_samples, 1)

    # test gradients work with anchor in stochastic mode
    samples.sum().backward(retain_graph=True)
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()
        t.grad = None  # reset for next test
    log_prob.sum().backward()
    for t in (log_alpha, log_beta):
        assert t.grad is not None and not t.grad.isnan().any().item()


def test_mode_concentration_beta():
    """Sanity checks for the `ModeConcentrationBeta` distribution."""
    rng = np.random.default_rng()

    # generate random space and associated distribution
    ndim, n_samples = map(int, rng.integers(2, 10, size=2))
    low = -5 - np.abs(rng.normal(scale=5, size=ndim))
    high = 5 + np.abs(rng.normal(scale=5, size=ndim))
    space = Box(low, high, dtype=np.float64)
    distribution = ModeConcentrationBeta(space)

    def _create_random_params() -> tuple[torch.Tensor, torch.Tensor]:
        logit_mode = torch.from_numpy(rng.normal(size=(n_samples, ndim)))
        logit_log_conc = torch.from_numpy(rng.normal(size=(n_samples, ndim)))
        return logit_mode.requires_grad_(), logit_log_conc.requires_grad_()

    # check samples lie in the space
    samples: torch.Tensor
    log_prob: torch.Tensor
    logit_mode, logit_log_conc = _create_random_params()
    samples, log_prob, _ = distribution(logit_mode, logit_log_conc, deterministic=False)
    assert all(s in space for s in samples.numpy(force=True))
    assert log_prob.shape == (n_samples, 1)

    # test backward of samples and  log_prob works
    samples.sum().backward(retain_graph=True)
    for t in (logit_mode, logit_log_conc):
        assert t.grad is not None and not t.grad.isnan().any().item()
        t.grad = None  # reset for next test
    log_prob.sum().backward()
    for t in (logit_mode, logit_log_conc):
        assert t.grad is not None and not t.grad.isnan().any().item()

    # check that deterministic samples (modes themselves) lie in the space and are equal to the
    # requested modes
    logit_mode, logit_log_conc = _create_random_params()
    samples, log_prob, _ = distribution(logit_mode, logit_log_conc, deterministic=True)
    assert all(s in space for s in samples.numpy(force=True))
    assert samples.shape == (n_samples, ndim)
    with torch.no_grad():
        mode_01 = distribution.padding + (1.0 - 2.0 * distribution.padding) * torch.sigmoid(
            logit_mode
        )
        expected_samples = distribution.loc + distribution.scale * mode_01
        torch.testing.assert_close(samples, expected_samples)

    # test mode backward works but does not include log_conc, and that log_prob doesn't require grad
    samples.sum().backward()
    assert logit_mode.grad is not None and not logit_mode.grad.isnan().any().item()
    assert logit_log_conc.grad is None
    assert not log_prob.requires_grad


def test_mode_concentration_beta_anchor():
    """Test anchor functionality for `ModeConcentrationBeta` distribution."""
    rng = np.random.default_rng()

    # generate random space and associated distribution
    ndim, n_samples = map(int, rng.integers(2, 10, size=2))
    low = -5 - np.abs(rng.normal(scale=5, size=ndim))
    high = 5 + np.abs(rng.normal(scale=5, size=ndim))
    space = Box(low, high, dtype=np.float64)
    distribution = ModeConcentrationBeta(space, padding=0)  # remove paddings to avoid distorsion

    # test deterministic sampling with anchor - when logit_mode=0, mode is on anchor
    logit_mode = torch.zeros((n_samples, ndim), dtype=torch.float64, requires_grad=True)
    logit_log_conc = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    anchor = torch.from_numpy(rng.uniform(low, high, size=(n_samples, ndim)))
    samples: torch.Tensor
    log_prob: torch.Tensor
    samples, log_prob, _ = distribution(logit_mode, logit_log_conc, True, anchor)
    assert all(s in space for s in samples.numpy(force=True))
    assert log_prob.shape == (n_samples, 1)
    torch.testing.assert_close(samples, anchor)

    # test gradients work with anchor
    samples.sum().backward()
    assert logit_mode.grad is not None and not logit_mode.grad.isnan().any().item()
    assert logit_log_conc.grad is None
    assert not log_prob.requires_grad

    # test stochastic sampling with anchor
    logit_mode = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    logit_log_conc = torch.from_numpy(rng.normal(size=(n_samples, ndim))).requires_grad_()
    anchor = torch.from_numpy(rng.uniform(low, high, size=(n_samples, ndim)))
    samples, log_prob, _ = distribution(logit_mode, logit_log_conc, anchor=anchor)
    assert all(s in space for s in samples.numpy(force=True))
    assert log_prob.shape == (n_samples, 1)

    # test gradients work with anchor in stochastic mode
    samples.sum().backward(retain_graph=True)
    for t in (logit_mode, logit_log_conc):
        assert t.grad is not None and not t.grad.isnan().any().item()
        t.grad = None  # reset for next test
    log_prob.sum().backward()
    for t in (logit_mode, logit_log_conc):
        assert t.grad is not None and not t.grad.isnan().any().item()
