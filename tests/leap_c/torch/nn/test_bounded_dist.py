import numpy as np
import torch
from gymnasium import spaces

from leap_c.torch.nn.bounded_distributions import ScaledBeta, SquashedGaussian


def test_scaled_beta():
    """Sanity checks for the ScaledBeta distribution."""
    test_space = spaces.Box(
        low=np.array([-10.0, -15.0, 31.0, 3.0]), high=np.array([-5.0, 20.0, 42.0, 4.0])
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


def test_squashed_gaussian_anchor():
    """Test anchor functionality for SquashedGaussian distribution."""
    test_space = spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([1.0, 2.0]))
    dist = SquashedGaussian(test_space)

    # Test deterministic sampling with anchor
    mean = torch.tensor([[0.0, 0.0]], requires_grad=True)
    log_std = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    anchor = torch.tensor([0.5, 1.0])

    samples, log_prob, _ = dist(mean, log_std, deterministic=True, anchor=anchor)

    # With anchor and deterministic, mean=0 should result in anchor value
    assert samples.shape == (1, 2)
    assert log_prob.shape == (1, 1)
    assert torch.allclose(samples[0], anchor, atol=1e-5)

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


def test_scaled_beta_anchor():
    """Test anchor functionality for ScaledBeta distribution."""
    test_space = spaces.Box(low=np.array([0.0, -5.0]), high=np.array([10.0, 5.0]))
    dist = ScaledBeta(test_space)

    # Test deterministic sampling with anchor - when alpha=beta, mode is at center
    # This gives us a predictable mode to test anchoring
    log_alpha = torch.tensor([[0.0, 0.0]], requires_grad=True)
    log_beta = torch.tensor([[0.0, 0.0]], requires_grad=True)
    anchor = torch.tensor([5.0, 0.0])

    samples, log_prob, _ = dist(log_alpha, log_beta, deterministic=True, anchor=anchor)

    # Check shapes
    assert samples.shape == (1, 2)
    assert log_prob.shape == (1, 1)

    # With equal alpha and beta, the mode is at 0.5 in [0,1] space
    # After shifting to align mode with anchor, the output should equal the anchor
    # (assuming no clamping is needed)
    assert torch.allclose(samples[0], anchor, atol=1e-5)

    # Samples should be within bounds after anchoring and clamping
    assert torch.all(samples >= torch.from_numpy(test_space.low))
    assert torch.all(samples <= torch.from_numpy(test_space.high))

    # Test gradients work with anchor
    samples.sum().backward()
    assert log_alpha.grad is not None and not torch.any(torch.isnan(log_alpha.grad))
    assert log_beta.grad is not None and not torch.any(torch.isnan(log_beta.grad))

    # Test stochastic sampling with anchor
    log_alpha = torch.tensor([[0.0, 0.0]], requires_grad=True)
    log_beta = torch.tensor([[0.0, 0.0]], requires_grad=True)
    torch.manual_seed(42)
    samples_stochastic, _, _ = dist(log_alpha, log_beta, deterministic=False, anchor=anchor)

    # Verify gradients work in stochastic mode
    samples_stochastic.sum().backward()
    assert log_alpha.grad is not None and not torch.any(torch.isnan(log_alpha.grad))
    assert log_beta.grad is not None and not torch.any(torch.isnan(log_beta.grad))

    # Samples should be in valid range
    assert samples_stochastic.shape == (1, 2)
    assert torch.all(samples_stochastic >= torch.from_numpy(test_space.low))
    assert torch.all(samples_stochastic <= torch.from_numpy(test_space.high))
