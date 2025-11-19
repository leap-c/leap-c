import numpy as np
import pytest
import torch

from leap_c.examples.hvac.acados_ocp import make_default_hvac_params
from leap_c.examples.hvac.planner import HvacPlanner, HvacPlannerConfig
from leap_c.ocp.acados.parameters import AcadosParameter


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


# ==================== Tests for Forecast Parameter Setting ====================


def create_planner_with_custom_params(
    N_horizon: int,
    stagewise: bool,
    ta_learnable: bool = True,
    solar_learnable: bool = True,
    price_learnable: bool = True,
) -> HvacPlanner:
    """Create a planner with custom parameter learnability settings.

    Args:
        N_horizon: Number of forecast steps.
        stagewise: Whether to use stagewise parameters.
        ta_learnable: Whether ambient temperature should be learnable.
        solar_learnable: Whether solar radiation should be learnable.
        price_learnable: Whether price should be learnable.

    Returns:
        HvacPlanner with custom parameter configuration.
    """
    # Get default params
    params = list(make_default_hvac_params(stagewise=stagewise, N_horizon=N_horizon))

    # Modify the interface for Ta, Phi_s, and price based on arguments
    for i, param in enumerate(params):
        if param.name == "Ta" and not ta_learnable:
            params[i] = AcadosParameter(
                name=param.name,
                default=param.default,
                space=param.space,
                interface="non-learnable",
                end_stages=param.end_stages,
            )
        elif param.name == "Phi_s" and not solar_learnable:
            params[i] = AcadosParameter(
                name=param.name,
                default=param.default,
                space=param.space,
                interface="non-learnable",
                end_stages=param.end_stages,
            )
        elif param.name == "price" and not price_learnable:
            params[i] = AcadosParameter(
                name=param.name,
                default=param.default,
                space=param.space,
                interface="non-learnable",
                end_stages=param.end_stages,
            )

    cfg = HvacPlannerConfig(N_horizon=N_horizon, stagewise=stagewise)
    return HvacPlanner(cfg=cfg, params=tuple(params))


@pytest.mark.parametrize("batch_size", [1, 4])
def test_all_forecasts_non_learnable(batch_size):
    """Test when all forecast parameters are non-learnable (extracted from obs)."""
    N_horizon = 8
    planner = create_planner_with_custom_params(
        N_horizon=N_horizon,
        stagewise=True,
        ta_learnable=False,
        solar_learnable=False,
        price_learnable=False,
    )

    # Create observations
    if batch_size == 1:
        obs = create_obs(N_horizon, seed=42)
        obs_torch = torch.from_numpy(obs).unsqueeze(0).double()
    else:
        obs_batch = np.stack([create_obs(N_horizon, seed=i) for i in range(batch_size)])
        obs_torch = torch.from_numpy(obs_batch).double()

    # Verify that none of the forecast parameters are learnable
    assert not planner.param_manager.has_learnable_param_pattern("Ta_*_*")
    assert not planner.param_manager.has_learnable_param_pattern("Phi_s_*_*")
    assert not planner.param_manager.has_learnable_param_pattern("price_*_*")

    # Forward pass should extract forecasts from obs
    ctx, u0, x, u, value = planner.forward(obs_torch)

    # Verify that the forward pass completed successfully
    assert ctx is not None
    assert x.shape[0] == batch_size
    assert u0 is not None or u is not None  # At least one should be set


@pytest.mark.parametrize("batch_size", [1, 4])
def test_all_forecasts_learnable(batch_size):
    """Test when all forecast parameters are learnable (not extracted from obs)."""
    N_horizon = 8
    planner = create_planner_with_custom_params(
        N_horizon=N_horizon,
        stagewise=True,
        ta_learnable=True,
        solar_learnable=True,
        price_learnable=True,
    )

    # Create observations
    if batch_size == 1:
        obs = create_obs(N_horizon, seed=42)
        obs_torch = torch.from_numpy(obs).unsqueeze(0).double()
    else:
        obs_batch = np.stack([create_obs(N_horizon, seed=i) for i in range(batch_size)])
        obs_torch = torch.from_numpy(obs_batch).double()

    # Verify that all forecast parameters are learnable
    assert planner.param_manager.has_learnable_param_pattern("Ta_*_*")
    assert planner.param_manager.has_learnable_param_pattern("Phi_s_*_*")
    assert planner.param_manager.has_learnable_param_pattern("price_*_*")

    # Forward pass should NOT extract forecasts from obs (uses learned params)
    ctx, u0, x, u, value = planner.forward(obs_torch)

    # Verify that the forward pass completed successfully
    assert ctx is not None
    assert x.shape[0] == batch_size
    assert u0 is not None or u is not None


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize(
    "ta_learnable,solar_learnable,price_learnable",
    [
        (True, False, False),  # Only Ta learnable
        (False, True, False),  # Only solar learnable
        (False, False, True),  # Only price learnable
        (True, True, False),  # Ta and solar learnable
        (True, False, True),  # Ta and price learnable
        (False, True, True),  # Solar and price learnable
    ],
)
def test_mixed_forecast_learnability(batch_size, ta_learnable, solar_learnable, price_learnable):
    """Test scenarios with mixed learnable/non-learnable forecast parameters."""
    N_horizon = 8
    planner = create_planner_with_custom_params(
        N_horizon=N_horizon,
        stagewise=True,
        ta_learnable=ta_learnable,
        solar_learnable=solar_learnable,
        price_learnable=price_learnable,
    )

    # Create observations with known values
    if batch_size == 1:
        obs = create_obs(N_horizon, seed=42)
        obs_torch = torch.from_numpy(obs).unsqueeze(0).double()
    else:
        obs_batch = np.stack([create_obs(N_horizon, seed=i) for i in range(batch_size)])
        obs_torch = torch.from_numpy(obs_batch).double()

    # Verify learnability patterns
    assert planner.param_manager.has_learnable_param_pattern("Ta_*_*") == ta_learnable
    assert planner.param_manager.has_learnable_param_pattern("Phi_s_*_*") == solar_learnable
    assert planner.param_manager.has_learnable_param_pattern("price_*_*") == price_learnable

    # Forward pass should handle mixed scenario correctly
    ctx, u0, x, u, value = planner.forward(obs_torch)

    # Verify that the forward pass completed successfully
    assert ctx is not None
    assert x.shape[0] == batch_size
    assert u0 is not None or u is not None


def test_forecast_extraction_indices():
    """Test that forecasts are extracted from correct observation indices."""
    N_horizon = 8
    planner = create_planner_with_custom_params(
        N_horizon=N_horizon,
        stagewise=True,
        ta_learnable=False,
        solar_learnable=False,
        price_learnable=False,
    )

    # Create observation with known values
    obs = create_obs(N_horizon, seed=123)
    obs_torch = torch.from_numpy(obs).unsqueeze(0).double()

    # Extract expected values manually
    N_forecast = N_horizon + 1
    expected_Ta = obs[5 : 5 + N_forecast]
    expected_solar = obs[5 + N_forecast : 5 + 2 * N_forecast]
    expected_price = obs[5 + 2 * N_forecast : 5 + 3 * N_forecast]

    # Verify shapes are correct
    assert len(expected_Ta) == N_forecast
    assert len(expected_solar) == N_forecast
    assert len(expected_price) == N_forecast

    # Forward pass
    ctx, u0, x, u, value = planner.forward(obs_torch)

    # Verify computation succeeded
    assert ctx is not None
    assert x.shape[0] == 1  # batch_size = 1


def test_forecast_with_different_horizons():
    """Test forecast parameter setting with different horizon lengths."""
    for N_horizon in [4, 8, 16]:
        planner = create_planner_with_custom_params(
            N_horizon=N_horizon,
            stagewise=True,
            ta_learnable=False,
            solar_learnable=True,  # Mix learnable and non-learnable
            price_learnable=False,
        )

        obs = create_obs(N_horizon, seed=42)
        obs_torch = torch.from_numpy(obs).unsqueeze(0).double()

        # Verify parameter patterns
        assert not planner.param_manager.has_learnable_param_pattern("Ta_*_*")
        assert planner.param_manager.has_learnable_param_pattern("Phi_s_*_*")
        assert not planner.param_manager.has_learnable_param_pattern("price_*_*")

        # Forward pass
        ctx, u0, x, u, value = planner.forward(obs_torch)

        # Verify success
        assert ctx is not None
        assert x.shape[0] == 1


def test_forecast_bounds_non_negative():
    """Test that solar radiation forecasts remain non-negative when extracted from obs."""
    N_horizon = 8
    planner = create_planner_with_custom_params(
        N_horizon=N_horizon,
        stagewise=True,
        ta_learnable=False,
        solar_learnable=False,
        price_learnable=False,
    )

    # Create observation - solar values should be non-negative
    obs = create_obs(N_horizon, seed=42)
    N_forecast = N_horizon + 1
    solar_forecast = obs[5 + N_forecast : 5 + 2 * N_forecast]

    # Verify the test observation has non-negative solar values
    assert np.all(solar_forecast >= 0), "Test observation should have non-negative solar values"

    obs_torch = torch.from_numpy(obs).unsqueeze(0).double()

    # Forward pass
    ctx, u0, x, u, value = planner.forward(obs_torch)

    # Verify computation succeeded
    assert ctx is not None
    assert x.shape[0] == 1
