import numpy as np
import torch
from gymnasium import spaces

from leap_c.torch.nn.bounded_distributions import ScaledBeta


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
