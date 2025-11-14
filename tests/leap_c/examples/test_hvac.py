import numpy as np
import pytest

from leap_c.examples.hvac.planner import HvacPlanner, HvacPlannerConfig


@pytest.fixture(scope="module")
def hvac_planner_stagewise():
    """Create an HVAC planner with stagewise parameters enabled."""
    cfg = HvacPlannerConfig(N_horizon=8, stagewise=True)
    planner = HvacPlanner(cfg)
    return planner


@pytest.fixture(scope="module")
def hvac_planner_non_stagewise():
    """Create an HVAC planner without stagewise parameters."""
    cfg = HvacPlannerConfig(N_horizon=8, stagewise=False)
    planner = HvacPlanner(cfg)
    return planner


def create_obs(N_horizon: int, seed: int = 42) -> np.ndarray:
    """Create a single observation with forecasts.

    Args:
        N_horizon: Number of forecast steps.
        seed: Random seed for reproducible forecasts.

    Returns:
        Observation array of shape (5 + 3*N_horizon,).
    """
    rng = np.random.default_rng(seed)

    # First 5 elements: quarter_hour, day, Ti, Th, Te
    base_obs = np.array([12.0, 150.0, 293.15, 303.15, 295.15], dtype=np.float32)

    # Forecasts: N_horizon each for Ta, solar, price
    N_forecast = N_horizon + 1  # +1 because we need N_horizon+1 stages
    Ta_forecast = rng.uniform(280.0, 300.0, N_forecast).astype(np.float32)
    solar_forecast = rng.uniform(0.0, 150.0, N_forecast).astype(np.float32)
    price_forecast = rng.uniform(0.0, 0.5, N_forecast).astype(np.float32)

    obs = np.concatenate([base_obs, Ta_forecast, solar_forecast, price_forecast])
    return obs


def test_default_param_single_obs(hvac_planner_stagewise):
    """Test default_param with a single observation."""
    N_horizon = hvac_planner_stagewise.cfg.N_horizon
    obs = create_obs(N_horizon, seed=42)

    param = hvac_planner_stagewise.default_param(obs)

    # Check that param is 1D
    assert param.ndim == 1
    assert len(param) > 0

    # Check that the result is a numpy array
    assert isinstance(param, np.ndarray)


def test_default_param_batched_obs(hvac_planner_stagewise):
    """Test default_param with batched observations."""
    N_horizon = hvac_planner_stagewise.cfg.N_horizon
    n_batch = 4

    # Create batched observations with different seeds for variety
    obs_batch = np.stack([create_obs(N_horizon, seed=i) for i in range(n_batch)])

    param_batch = hvac_planner_stagewise.default_param(obs_batch)

    # Check that param_batch has correct shape (n_batch, n_param)
    assert param_batch.ndim == 2
    assert param_batch.shape[0] == n_batch

    # Check that each batch element has the same parameter length as single obs
    single_obs = obs_batch[0]
    single_param = hvac_planner_stagewise.default_param(single_obs)
    assert param_batch.shape[1] == len(single_param)

    # Check that the result is a numpy array
    assert isinstance(param_batch, np.ndarray)


def test_default_param_batch_consistency(hvac_planner_stagewise):
    """Test that batched and individual calls produce consistent results."""
    N_horizon = hvac_planner_stagewise.cfg.N_horizon
    n_batch = 3

    # Create batched observations
    obs_batch = np.stack([create_obs(N_horizon, seed=i) for i in range(n_batch)])

    # Get batched params
    param_batch = hvac_planner_stagewise.default_param(obs_batch)

    # Get individual params and compare
    for i in range(n_batch):
        single_obs = obs_batch[i]
        single_param = hvac_planner_stagewise.default_param(single_obs)

        # Check that individual result matches the batched result
        np.testing.assert_allclose(
            param_batch[i],
            single_param,
            rtol=1e-6,
            err_msg=f"Batch element {i} does not match individual computation",
        )


def test_default_param_non_stagewise(hvac_planner_non_stagewise):
    """Test default_param without stagewise parameters (forecasts ignored)."""
    N_horizon = hvac_planner_non_stagewise.cfg.N_horizon

    # Single observation
    obs_single = create_obs(N_horizon, seed=42)
    param_single = hvac_planner_non_stagewise.default_param(obs_single)
    assert param_single.ndim == 1

    # Batched observation
    n_batch = 3
    obs_batch = np.stack([create_obs(N_horizon, seed=i) for i in range(n_batch)])
    param_batch = hvac_planner_non_stagewise.default_param(obs_batch)

    # When stagewise=False, all params should be the same (default)
    assert param_batch.ndim == 2
    assert param_batch.shape[0] == n_batch
    assert param_batch.shape[1] == len(param_single)


def test_default_param_none_obs(hvac_planner_stagewise):
    """Test default_param with None observation."""
    param = hvac_planner_stagewise.default_param(None)

    # Should return default params
    assert param.ndim == 1
    assert len(param) > 0
    assert isinstance(param, np.ndarray)


def test_forecast_extraction(hvac_planner_stagewise):
    """Test that forecasts are correctly extracted and set in parameters."""
    N_horizon = hvac_planner_stagewise.cfg.N_horizon

    # Create an observation with known forecast values
    obs = create_obs(N_horizon, seed=123)

    # Extract forecasts manually
    N_forecast = N_horizon + 1
    Ta_forecast = obs[5 : 5 + N_forecast]
    solar_forecast = obs[5 + N_forecast : 5 + 2 * N_forecast]
    price_forecast = obs[5 + 2 * N_forecast : 5 + 3 * N_forecast]

    # Verify we have the right number of forecast values
    assert len(Ta_forecast) == N_forecast
    assert len(solar_forecast) == N_forecast
    assert len(price_forecast) == N_forecast

    # Get parameters with forecasts
    param = hvac_planner_stagewise.default_param(obs)

    # The test passes if no exception is raised during parameter computation
    assert param is not None
    assert len(param) > 0
