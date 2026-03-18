from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from leap_c.examples.hvac.acados_ocp import make_default_hvac_params
from leap_c.examples.hvac.planner import HvacPlanner, HvacPlannerConfig
from leap_c.ocp.acados.parameters import AcadosParameter
from leap_c.torch.rl.buffer import collate_dict_to_tensordict, pytree_tensor_to


def _make_synthetic_weather() -> pd.DataFrame:
    """Return a minimal synthetic weather DataFrame matching Open-Meteo output format."""
    # Two years of 15-min data covering heating-season months (Jan-Apr, Sep-Dec)
    index = pd.date_range("2020-01-01", "2021-12-31 23:45", freq="15min", tz="UTC")
    index.name = "Timestamp"
    rng = np.random.default_rng(0)
    n = len(index)
    return pd.DataFrame(
        {
            "date": index,
            "temperature_2m": rng.uniform(-10, 20, n).astype(np.float32),
            "apparent_temperature": rng.uniform(-15, 18, n).astype(np.float32),
            "shortwave_radiation": rng.uniform(0, 200, n).astype(np.float32),
            "direct_normal_irradiance": rng.uniform(0, 300, n).astype(np.float32),
        },
        index=index,
    )


def _make_synthetic_price() -> pd.DataFrame:
    """Return a minimal synthetic price DataFrame matching energy-charts output format."""
    index = pd.date_range("2020-01-01", "2021-12-31 23:45", freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Timestamp": index,
            "price": rng.uniform(0.0, 0.5, len(index)).astype(np.float32),
        }
    )


@pytest.fixture()
def mock_external_data():
    """Patch both external API calls."""
    with (
        patch(
            "leap_c.examples.hvac.dataset.get_open_meteo_data",
            return_value=_make_synthetic_weather(),
        ),
        patch(
            "leap_c.examples.hvac.dataset.get_energy_charts_data",
            return_value=_make_synthetic_price(),
        ),
    ):
        yield


def _make_synthetic_weather() -> pd.DataFrame:
    """Return a minimal synthetic weather DataFrame matching Open-Meteo output format."""
    # Two years of 15-min data covering heating-season months (Jan-Apr, Sep-Dec)
    index = pd.date_range("2020-01-01", "2021-12-31 23:45", freq="15min", tz="UTC")
    index.name = "Timestamp"
    rng = np.random.default_rng(0)
    n = len(index)
    return pd.DataFrame(
        {
            "date": index,
            "temperature_2m": rng.uniform(-10, 20, n).astype(np.float32),
            "apparent_temperature": rng.uniform(-15, 18, n).astype(np.float32),
            "shortwave_radiation": rng.uniform(0, 200, n).astype(np.float32),
            "direct_normal_irradiance": rng.uniform(0, 300, n).astype(np.float32),
        },
        index=index,
    )


def _make_synthetic_price() -> pd.DataFrame:
    """Return a minimal synthetic price DataFrame matching energy-charts output format."""
    index = pd.date_range("2020-01-01", "2021-12-31 23:45", freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Timestamp": index,
            "price": rng.uniform(0.0, 0.5, len(index)).astype(np.float32),
        }
    )


@pytest.fixture()
def mock_external_data():
    """Patch both external API calls."""
    with (
        patch(
            "leap_c.examples.hvac.dataset.get_open_meteo_data",
            return_value=_make_synthetic_weather(),
        ),
        patch(
            "leap_c.examples.hvac.dataset.get_energy_charts_data",
            return_value=_make_synthetic_price(),
        ),
    ):
        yield


@pytest.fixture(scope="module")
def hvac_planner_stagewise():
    """Create an HVAC planner with stagewise parameters enabled."""
    cfg = HvacPlannerConfig(N_horizon=8, param_granularity="stagewise")
    planner = HvacPlanner(cfg)
    return planner


@pytest.fixture(scope="module")
def hvac_planner_non_stagewise():
    """Create an HVAC planner without stagewise parameters."""
    cfg = HvacPlannerConfig(N_horizon=8, param_granularity="global")
    planner = HvacPlanner(cfg)
    return planner


def create_obs(N_horizon: int, seed: int = 42) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """Create a single dict observation with forecasts.

    Args:
        N_horizon: Number of forecast steps.
        seed: Random seed for reproducible forecasts.

    Returns:
        Dict observation matching the HVAC environment's observation space.
    """
    rng = np.random.default_rng(seed)

    N_forecast = N_horizon + 1  # +1 because we need N_horizon+1 stages
    temperature_forecast = rng.uniform(280.0, 300.0, N_forecast).astype(np.float32)
    solar_forecast = rng.uniform(0.0, 150.0, N_forecast).astype(np.float32)
    price_forecast = rng.uniform(0.0, 0.5, N_forecast).astype(np.float32)

    return {
        "time": {
            "quarter_hour": np.array([12.0], dtype=np.float32),
            "day_of_year": np.array([150.0], dtype=np.float32),
            "day_of_week": np.array([3.0], dtype=np.float32),
        },
        "state": np.array([293.15, 303.15, 295.15], dtype=np.float32),
        "forecast": {
            "temperature": temperature_forecast,
            "solar": solar_forecast,
            "price": price_forecast,
        },
    }


_collate_fn_map = {**default_collate_fn_map, dict: collate_dict_to_tensordict}


def collate_obs(obs_list: list[dict]):
    """Collate a list of dict observations into a batched TensorDict.

    Uses the same collation logic as the replay buffer to stack dict
    observations and convert numpy arrays to torch tensors.

    Args:
        obs_list: List of dict observations from create_obs().

    Returns:
        TensorDict with a leading batch dimension and float64 tensors.
    """
    return pytree_tensor_to(
        collate(obs_list, collate_fn_map=_collate_fn_map),
        device="cpu",
        tensor_dtype=torch.float64,
    )


def test_default_param_single_obs(hvac_planner_stagewise: HvacPlanner) -> None:
    """Test default_param with a single observation."""
    N_horizon = hvac_planner_stagewise.cfg.N_horizon
    obs = create_obs(N_horizon, seed=42)

    param = hvac_planner_stagewise.default_param(obs)

    # Check that param is 1D
    assert param.ndim == 1
    assert len(param) > 0

    # Check that the result is a numpy array
    assert isinstance(param, np.ndarray)


def test_default_param_batched_obs(hvac_planner_stagewise: HvacPlanner) -> None:
    """Test default_param with batched observations."""
    N_horizon = hvac_planner_stagewise.cfg.N_horizon
    n_batch = 4

    # Create batched observations with different seeds for variety
    obs_list = [create_obs(N_horizon, seed=i) for i in range(n_batch)]
    obs_batch = collate_obs(obs_list)

    param_batch = hvac_planner_stagewise.default_param(obs_batch)

    # Check that param_batch has correct shape (n_batch, n_param)
    assert param_batch.ndim == 2
    assert param_batch.shape[0] == n_batch

    # Check that each batch element has the same parameter length as single obs
    single_obs = obs_list[0]
    single_param = hvac_planner_stagewise.default_param(single_obs)
    assert param_batch.shape[1] == len(single_param)

    # Check that the result is a numpy array
    assert isinstance(param_batch, np.ndarray)


def test_default_param_batch_consistency(hvac_planner_stagewise: HvacPlanner) -> None:
    """Test that batched and individual calls produce consistent results."""
    N_horizon = hvac_planner_stagewise.cfg.N_horizon
    n_batch = 3

    # Create batched observations
    obs_list = [create_obs(N_horizon, seed=i) for i in range(n_batch)]
    obs_batch = collate_obs(obs_list)

    # Get batched params
    param_batch = hvac_planner_stagewise.default_param(obs_batch)

    # Get individual params and compare
    for i in range(n_batch):
        single_obs = obs_list[i]
        single_param = hvac_planner_stagewise.default_param(single_obs)

        # Check that individual result matches the batched result
        np.testing.assert_allclose(
            param_batch[i],
            single_param,
            rtol=1e-6,
            err_msg=f"Batch element {i} does not match individual computation",
        )


def test_default_param_non_stagewise(hvac_planner_non_stagewise: HvacPlanner) -> None:
    """Test default_param without stagewise parameters (forecasts ignored)."""
    N_horizon = hvac_planner_non_stagewise.cfg.N_horizon

    # Single observation
    obs_single = collate_obs([create_obs(N_horizon, seed=42)])
    param_single = hvac_planner_non_stagewise.default_param(obs_single)
    assert param_single.ndim == 2

    # Batched observation
    n_batch = 3
    obs_batch = collate_obs([create_obs(N_horizon, seed=i) for i in range(n_batch)])
    param_batch = hvac_planner_non_stagewise.default_param(obs_batch)

    # When stagewise=False, all params should be the same (default)
    assert param_batch.ndim == 2
    assert param_batch.shape[0] == n_batch
    assert param_batch.shape[1] == len(param_single[0])


def test_forecast_extraction(hvac_planner_stagewise: HvacPlanner) -> None:
    """Test that forecasts are correctly extracted and set in parameters."""
    N_horizon = hvac_planner_stagewise.cfg.N_horizon

    # Create an observation with known forecast values
    obs = create_obs(N_horizon, seed=123)

    # Extract forecasts from dict observation
    N_forecast = N_horizon + 1
    temperature_forecast = obs["forecast"]["temperature"]
    solar_forecast = obs["forecast"]["solar"]
    price_forecast = obs["forecast"]["price"]

    # Verify we have the right number of forecast values
    assert len(temperature_forecast) == N_forecast
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
    ta_learnable: bool = False,
    solar_learnable: bool = False,
    price_learnable: bool = False,
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
    # Get default params (now temperature, solar, price are non-learnable by default)
    params = list(
        make_default_hvac_params(
            interface="reference", granularity="stagewise", N_horizon=N_horizon
        )
    )

    # Modify the interface for temperature, solar, and price based on arguments
    for i, param in enumerate(params):
        if param.name == "temperature":
            params[i] = AcadosParameter(
                name=param.name,
                default=param.default,
                space=param.space,
                interface="learnable" if ta_learnable else "non-learnable",
                end_stages=param.end_stages,
            )
        elif param.name == "solar":
            params[i] = AcadosParameter(
                name=param.name,
                default=param.default,
                space=param.space,
                interface="learnable" if solar_learnable else "non-learnable",
                end_stages=param.end_stages,
            )
        elif param.name == "price":
            params[i] = AcadosParameter(
                name=param.name,
                default=param.default,
                space=param.space,
                interface="learnable" if price_learnable else "non-learnable",
                end_stages=param.end_stages,
            )

    cfg = HvacPlannerConfig(
        N_horizon=N_horizon, param_granularity="stagewise" if stagewise else "global"
    )
    return HvacPlanner(cfg=cfg, params=tuple(params))


@pytest.mark.parametrize("batch_size", [1, 4])
def test_all_forecasts_non_learnable(batch_size: int) -> None:
    """Test when all forecast parameters are non-learnable (extracted from obs)."""
    N_horizon = 8
    planner = create_planner_with_custom_params(
        N_horizon=N_horizon,
        stagewise=True,
        ta_learnable=False,  # default, but explicit for clarity
        solar_learnable=False,  # default, but explicit for clarity
        price_learnable=False,  # default, but explicit for clarity
    )

    # Create observations
    obs_list = [create_obs(N_horizon, seed=i) for i in range(batch_size)]
    obs_batch = collate_obs(obs_list)

    # Verify that none of the forecast parameters are learnable
    assert not planner.param_manager.has_learnable_param_pattern("temperature_*_*")
    assert not planner.param_manager.has_learnable_param_pattern("solar_*_*")
    assert not planner.param_manager.has_learnable_param_pattern("price_*_*")

    # Forward pass should extract forecasts from obs
    ctx, u0, x, u, value = planner.forward(obs_batch)

    # Verify that the forward pass completed successfully
    assert ctx is not None
    assert x.shape[0] == batch_size
    assert u0 is not None or u is not None  # At least one should be set


@pytest.mark.parametrize("batch_size", [1, 4])
def test_all_forecasts_learnable(batch_size: int) -> None:
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
    obs_list = [create_obs(N_horizon, seed=i) for i in range(batch_size)]
    obs_batch = collate_obs(obs_list)

    # Verify that all forecast parameters are learnable
    assert planner.param_manager.has_learnable_param_pattern("temperature_*_*")
    assert planner.param_manager.has_learnable_param_pattern("solar_*_*")
    assert planner.param_manager.has_learnable_param_pattern("price_*_*")

    # Forward pass should NOT extract forecasts from obs (uses learned params)
    ctx, u0, x, u, value = planner.forward(obs_batch)

    # Verify that the forward pass completed successfully
    assert ctx is not None
    assert x.shape[0] == batch_size
    assert u0 is not None or u is not None


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize(
    "ta_learnable,solar_learnable,price_learnable",
    [
        (True, False, False),  # Only temperature learnable
        (False, True, False),  # Only solar learnable
        (False, False, True),  # Only price learnable
        (True, True, False),  # temperature and solar learnable
        (True, False, True),  # temperature and price learnable
        (False, True, True),  # Solar and price learnable
    ],
)
def test_mixed_forecast_learnability(
    batch_size: int, ta_learnable: bool, solar_learnable: bool, price_learnable: bool
) -> None:
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
    obs_list = [create_obs(N_horizon, seed=i) for i in range(batch_size)]
    obs_batch = collate_obs(obs_list)

    # Verify learnability patterns
    assert planner.param_manager.has_learnable_param_pattern("temperature_*_*") == ta_learnable
    assert planner.param_manager.has_learnable_param_pattern("solar_*_*") == solar_learnable
    assert planner.param_manager.has_learnable_param_pattern("price_*_*") == price_learnable

    # Forward pass should handle mixed scenario correctly
    ctx, u0, x, u, value = planner.forward(obs_batch)

    # Verify that the forward pass completed successfully
    assert ctx is not None
    assert x.shape[0] == batch_size
    assert u0 is not None or u is not None


def test_forecast_extraction_indices() -> None:
    """Test that forecasts are extracted from correct observation indices."""
    N_horizon = 8
    planner = create_planner_with_custom_params(
        N_horizon=N_horizon,
        stagewise=True,
        ta_learnable=False,  # default
        solar_learnable=False,  # default
        price_learnable=False,  # default
    )

    # Create observation with known values
    obs = create_obs(N_horizon, seed=123)
    obs_batch = collate_obs([obs])

    # Extract expected values from dict observation
    N_forecast = N_horizon + 1
    expected_Ta = obs["forecast"]["temperature"]
    expected_solar = obs["forecast"]["solar"]
    expected_price = obs["forecast"]["price"]

    # Verify shapes are correct
    assert len(expected_Ta) == N_forecast
    assert len(expected_solar) == N_forecast
    assert len(expected_price) == N_forecast

    # Forward pass
    ctx, u0, x, u, value = planner.forward(obs_batch)

    # Verify computation succeeded
    assert ctx is not None
    assert x.shape[0] == 1  # batch_size = 1


def test_forecast_with_different_horizons() -> None:
    """Test forecast parameter setting with different horizon lengths."""
    for N_horizon in [4, 8, 16]:
        planner = create_planner_with_custom_params(
            N_horizon=N_horizon,
            stagewise=True,
            ta_learnable=False,  # default
            solar_learnable=True,  # Make this learnable to test mixed scenario
            price_learnable=False,  # default
        )

        obs = create_obs(N_horizon, seed=42)
        obs_torch = collate_obs([obs])

        # Verify parameter patterns
        assert not planner.param_manager.has_learnable_param_pattern("temperature_*_*")
        assert planner.param_manager.has_learnable_param_pattern("solar_*_*")
        assert not planner.param_manager.has_learnable_param_pattern("price_*_*")

        # Forward pass
        ctx, u0, x, u, value = planner.forward(obs_torch)

        # Verify success
        assert ctx is not None
        assert x.shape[0] == 1


def test_forecast_bounds_non_negative() -> None:
    """Test that solar radiation forecasts remain non-negative when extracted from obs."""
    N_horizon = 8
    planner = create_planner_with_custom_params(
        N_horizon=N_horizon,
        stagewise=True,
        ta_learnable=False,  # default
        solar_learnable=False,  # default
        price_learnable=False,  # default
    )

    # Create observation - solar values should be non-negative
    obs = create_obs(N_horizon, seed=42)
    solar_forecast = obs["forecast"]["solar"]

    # Verify the test observation has non-negative solar values
    assert np.all(solar_forecast >= 0), "Test observation should have non-negative solar values"

    obs_batch = collate_obs([obs])

    # Forward pass
    ctx, u0, x, u, value = planner.forward(obs_batch)

    # Verify computation succeeded
    assert ctx is not None
    assert x.shape[0] == 1


def test_default_param_in_param_space(hvac_planner_stagewise: HvacPlanner) -> None:
    """Test that default_param returns parameters within the defined param_space."""
    N_horizon = hvac_planner_stagewise.cfg.N_horizon
    obs = create_obs(N_horizon, seed=42)

    param = hvac_planner_stagewise.default_param(obs)

    # Check that each parameter is within the defined space
    param_space = hvac_planner_stagewise.param_space

    assert param_space.contains(param), "Default parameters are not within the defined param space"


def test_continual_episodes_only_valid_months(mock_external_data) -> None:
    """Test that continual learning episodes only contain dates from valid months."""
    from leap_c.examples.hvac.dataset import DataConfig, HvacDataset

    # Create dataset with continual mode
    cfg = DataConfig(mode="continual")
    dataset = HvacDataset(cfg=cfg)

    valid_months = cfg.valid_months
    assert valid_months is not None

    horizon = 96  # 24 hours at 15-min intervals

    # Reset continual index
    dataset.reset_continual_index()

    # Sample multiple episodes and check all dates
    n_episodes = 20
    invalid_dates_found = []

    for ep in range(n_episodes):
        start_idx, max_steps = dataset._sample_continual(horizon)

        # Check all steps in the episode
        for step in range(max_steps):
            idx = start_idx + step
            date = dataset.index[idx]

            if date.month not in valid_months:
                invalid_dates_found.append(
                    f"Episode {ep}, step {step}: {date} (month {date.month})"
                )

    assert len(invalid_dates_found) == 0, (
        f"Found {len(invalid_dates_found)} dates in invalid months:\n"
        + "\n".join(invalid_dates_found[:10])  # Show first 10
    )


def test_continual_episodes_sequential_coverage(mock_external_data) -> None:
    """Test that continual episodes cover the dataset sequentially."""
    from leap_c.examples.hvac.dataset import DataConfig, HvacDataset

    cfg = DataConfig(mode="continual")
    dataset = HvacDataset(cfg=cfg)

    horizon = 96
    dataset.reset_continual_index()

    # Collect episode boundaries
    episodes = []
    initial_idx = None

    for i in range(50):  # Sample up to 50 episodes
        start_idx, max_steps = dataset._sample_continual(horizon)

        if initial_idx is None:
            initial_idx = start_idx
        elif start_idx == initial_idx:
            # Wrapped around, stop
            break

        episodes.append((start_idx, max_steps))

    # Check that episodes are sequential (each starts after previous ends)
    for i in range(1, len(episodes)):
        prev_end = episodes[i - 1][0] + episodes[i - 1][1]
        curr_start = episodes[i][0]

        # Current episode should start at or after previous episode ended
        assert curr_start >= prev_end, (
            f"Episode {i} starts at {curr_start} but episode {i - 1} ended at {prev_end}"
        )
