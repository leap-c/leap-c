import numpy as np
import pytest
import torch
from gymnasium import spaces

from leap_c.examples.hvac.acados_ocp import make_default_hvac_params
from leap_c.examples.hvac.planner import HvacPlanner, HvacPlannerConfig
from leap_c.ocp.acados.parameters import AcadosParameter
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


# def test_scaled_beta_anchor():
#     """Test anchor functionality for ScaledBeta distribution."""
#     test_space = spaces.Box(low=np.array([0.0, -5.0]), high=np.array([10.0, 5.0]))
#     dist = ScaledBeta(test_space)
#
#     # Test deterministic sampling with anchor - when alpha=beta, mode is at center
#     # This gives us a predictable mode to test anchoring
#     log_alpha = torch.tensor([[0.0, 0.0]], requires_grad=True)
#     log_beta = torch.tensor([[0.0, 0.0]], requires_grad=True)
#     anchor = torch.tensor([5.0, 0.0])
#
#     samples, log_prob, _ = dist(log_alpha, log_beta, deterministic=True, anchor=anchor)
#
#     # Check shapes
#     assert samples.shape == (1, 2)
#     assert log_prob.shape == (1, 1)
#
#     # With equal alpha and beta, the mode is at 0.5 in [0,1] space
#     # After shifting to align mode with anchor, the output should equal the anchor
#     # (assuming no clamping is needed)
#     assert torch.allclose(samples[0], anchor, atol=1e-3)
#
#     # Samples should be within bounds after anchoring and clamping
#     assert torch.all(samples >= torch.from_numpy(test_space.low))
#     assert torch.all(samples <= torch.from_numpy(test_space.high))
#
#     # Test gradients work with anchor
#     samples.sum().backward()
#     assert log_alpha.grad is not None and not torch.any(torch.isnan(log_alpha.grad))
#     assert log_beta.grad is not None and not torch.any(torch.isnan(log_beta.grad))
#
#     # Test stochastic sampling with anchor
#     log_alpha = torch.tensor([[0.0, 0.0]], requires_grad=True)
#     log_beta = torch.tensor([[0.0, 0.0]], requires_grad=True)
#     torch.manual_seed(42)
#     samples_stochastic, _, _ = dist(log_alpha, log_beta, deterministic=False, anchor=anchor)
#
#     # Verify gradients work in stochastic mode
#     samples_stochastic.sum().backward()
#     assert log_alpha.grad is not None and not torch.any(torch.isnan(log_alpha.grad))
#     assert log_beta.grad is not None and not torch.any(torch.isnan(log_beta.grad))
#
#     # Samples should be in valid range
#     assert samples_stochastic.shape == (1, 2)
#     assert torch.all(samples_stochastic >= torch.from_numpy(test_space.low))
#     assert torch.all(samples_stochastic <= torch.from_numpy(test_space.high))


def create_hvac_planner_with_q_ti_only(N_horizon: int = 8, stagewise: bool = True) -> HvacPlanner:
    """Create an HVAC planner with only q_Ti as learnable parameter.

    Args:
        N_horizon: Number of forecast steps.
        stagewise: Whether to use stagewise parameters.

    Returns:
        HvacPlanner with only q_Ti as learnable parameter.
    """
    # Get default params
    params = list(make_default_hvac_params(stagewise=stagewise, N_horizon=N_horizon))

    # Make all parameters non-learnable except q_Ti
    modified_params = []
    for param in params:
        if param.name == "q_Ti":
            # Keep q_Ti as learnable
            modified_params.append(param)
        else:
            # Make all other parameters non-learnable
            modified_params.append(
                AcadosParameter(
                    name=param.name,
                    default=param.default,
                    space=param.space,
                    interface="non-learnable",
                    end_stages=param.end_stages,
                )
            )

    cfg = HvacPlannerConfig(N_horizon=N_horizon, stagewise=stagewise)
    return HvacPlanner(cfg=cfg, params=tuple(modified_params))


# @pytest.mark.parametrize("distribution_name", ["squashed_gaussian", "scaled_beta"])
@pytest.mark.parametrize("distribution_name", ["squashed_gaussian"])
@pytest.mark.parametrize("num_samples", [100, 1000])
def test_bounded_distribution_with_hvac_controller(
    distribution_name: str, num_samples: int
) -> None:
    """Test that bounded distributions only sample within controller space.

    This test creates an HVAC planner with only q_Ti as learnable parameter,
    then verifies that all samples from the bounded distribution are within
    the parameter space bounds.

    Args:
        distribution_name: Name of the bounded distribution to test.
        num_samples: Number of samples to draw for verification.
    """
    # Create HVAC planner with only q_Ti as learnable
    planner = create_hvac_planner_with_q_ti_only(N_horizon=8, stagewise=True)

    # Get the parameter space from the planner
    param_space: spaces.Box = planner.param_manager.get_param_space()

    # Create the bounded distribution
    if distribution_name == "squashed_gaussian":
        dist = SquashedGaussian(param_space)
    elif distribution_name == "scaled_beta":
        dist = ScaledBeta(param_space)
    else:
        raise ValueError(f"Unknown distribution: {distribution_name}")

    # Sample from the distribution multiple times
    batch_size = num_samples

    if distribution_name == "squashed_gaussian":
        # Generate random means and log_stds for SquashedGaussian
        mean = torch.randn(batch_size, param_space.shape[0])
        log_std = torch.randn(batch_size, param_space.shape[0]) * 0.5 - 1.0
        samples, log_probs, _ = dist(mean, log_std, deterministic=False)
    else:  # scaled_beta
        # Generate random log_alpha and log_beta for ScaledBeta
        log_alpha = torch.randn(batch_size, param_space.shape[0])
        log_beta = torch.randn(batch_size, param_space.shape[0])
        samples, log_probs, _ = dist(log_alpha, log_beta, deterministic=False)

    # Verify shapes
    assert samples.shape == (batch_size, param_space.shape[0])
    assert log_probs.shape == (batch_size, 1)

    # Verify all samples are within bounds
    samples_np = samples.detach().numpy()
    for i, sample in enumerate(samples_np):
        assert sample in param_space, (
            f"Sample {i} out of bounds: {sample} not in [{param_space.low}, {param_space.high}]"
        )

    # Additional verification: check against explicit bounds
    assert torch.all(samples >= torch.from_numpy(param_space.low).to(samples.device))
    assert torch.all(samples <= torch.from_numpy(param_space.high).to(samples.device))

    # Verify log_probs are finite (no NaN or Inf)
    assert torch.all(torch.isfinite(log_probs))


@pytest.mark.parametrize("distribution_name", ["squashed_gaussian", "scaled_beta"])
def test_bounded_distribution_deterministic_mode_with_hvac(distribution_name: str) -> None:
    """Test deterministic sampling mode with HVAC controller parameter space.

    Args:
        distribution_name: Name of the bounded distribution to test.
    """
    # Create HVAC planner with only q_Ti as learnable
    planner = create_hvac_planner_with_q_ti_only(N_horizon=8, stagewise=True)
    param_space = planner.param_manager.get_param_space()

    # Create the bounded distribution
    if distribution_name == "squashed_gaussian":
        dist = SquashedGaussian(param_space)
    elif distribution_name == "scaled_beta":
        dist = ScaledBeta(param_space)
    else:
        raise ValueError(f"Unknown distribution: {distribution_name}")

    batch_size = 50

    if distribution_name == "squashed_gaussian":
        mean = torch.randn(batch_size, param_space.shape[0])
        log_std = torch.randn(batch_size, param_space.shape[0]) * 0.5 - 1.0
        samples, log_probs, _ = dist(mean, log_std, deterministic=True)
    else:  # scaled_beta
        log_alpha = torch.randn(batch_size, param_space.shape[0])
        log_beta = torch.randn(batch_size, param_space.shape[0])
        samples, log_probs, _ = dist(log_alpha, log_beta, deterministic=True)

    # Verify all deterministic samples are within bounds
    samples_np = samples.detach().numpy()
    for i, sample in enumerate(samples_np):
        assert sample in param_space, (
            f"Deterministic sample {i} out of bounds: {sample} not in "
            f"[{param_space.low}, {param_space.high}]"
        )

    # Verify deterministic mode gives same result for same input
    if distribution_name == "squashed_gaussian":
        samples2, _, _ = dist(mean, log_std, deterministic=True)
    else:
        samples2, _, _ = dist(log_alpha, log_beta, deterministic=True)

    assert torch.allclose(samples, samples2, atol=1e-6)


@pytest.mark.parametrize("distribution_name", ["squashed_gaussian", "scaled_beta"])
def test_bounded_distribution_gradients_with_hvac(distribution_name: str) -> None:
    """Test that gradients flow properly through bounded distributions.

    Args:
        distribution_name: Name of the bounded distribution to test.
    """
    # Create HVAC planner with only q_Ti as learnable
    planner = create_hvac_planner_with_q_ti_only(N_horizon=8, stagewise=True)
    param_space = planner.param_manager.get_param_space()

    # Create the bounded distribution
    if distribution_name == "squashed_gaussian":
        dist = SquashedGaussian(param_space)
    elif distribution_name == "scaled_beta":
        dist = ScaledBeta(param_space)
    else:
        raise ValueError(f"Unknown distribution: {distribution_name}")

    batch_size = 10

    if distribution_name == "squashed_gaussian":
        mean = torch.randn(batch_size, param_space.shape[0], requires_grad=True)
        log_std = torch.randn(batch_size, param_space.shape[0], requires_grad=True)
        samples, log_probs, _ = dist(mean, log_std, deterministic=False)

        # Test gradient flow through samples
        loss = samples.sum()
        loss.backward()
        assert mean.grad is not None
        assert not torch.any(torch.isnan(mean.grad))
        assert log_std.grad is not None
        assert not torch.any(torch.isnan(log_std.grad))

        # Reset gradients and test gradient flow through log_probs
        mean.grad = None
        log_std.grad = None
        samples, log_probs, _ = dist(mean, log_std, deterministic=False)
        loss = log_probs.sum()
        loss.backward()
        assert mean.grad is not None
        assert not torch.any(torch.isnan(mean.grad))
        assert log_std.grad is not None
        assert not torch.any(torch.isnan(log_std.grad))

    else:  # scaled_beta
        log_alpha = torch.randn(batch_size, param_space.shape[0], requires_grad=True)
        log_beta = torch.randn(batch_size, param_space.shape[0], requires_grad=True)
        samples, log_probs, _ = dist(log_alpha, log_beta, deterministic=False)

        # Test gradient flow through samples
        loss = samples.sum()
        loss.backward()
        assert log_alpha.grad is not None
        assert not torch.any(torch.isnan(log_alpha.grad))
        assert log_beta.grad is not None
        assert not torch.any(torch.isnan(log_beta.grad))

        # Reset gradients and test gradient flow through log_probs
        log_alpha.grad = None
        log_beta.grad = None
        samples, log_probs, _ = dist(log_alpha, log_beta, deterministic=False)
        loss = log_probs.sum()
        loss.backward()
        assert log_alpha.grad is not None
        assert not torch.any(torch.isnan(log_alpha.grad))
        assert log_beta.grad is not None
        assert not torch.any(torch.isnan(log_beta.grad))
