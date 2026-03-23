from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from scipy.constants import convert_temperature

from leap_c.examples.hvac.dataset import HvacDataset
from leap_c.examples.hvac.dynamics import (
    HydronicParameters,
    compute_discrete_matrices,
    compute_steady_state,
)
from leap_c.examples.hvac.forecast import Forecaster
from leap_c.examples.hvac.utils import set_temperature_limits
from leap_c.examples.utils.matplotlib_env import MatplotlibRenderEnv


@dataclass(kw_only=True)
class HvacEnvConfig:
    """Configuration for the HVAC environment.

    Attributes:
        thermal_params: Thermal model parameters.
        step_size: Simulation time step in seconds.
        enable_noise: Whether to include stochastic noise.
        randomize_params: Whether to randomize thermal parameters.
        param_noise_scale: Scale for parameter randomization.
        random_seed: Seed for parameter randomization.
    """

    thermal_params: HydronicParameters | None = None
    step_size: float = 900.0
    enable_noise: bool = True
    randomize_params: bool = False
    param_noise_scale: float = 0.3
    random_seed: int = 0

    def __post_init__(self):
        if self.step_size != 900.0:
            raise NotImplementedError("Only step_size of 900s is currently supported.")


class StochasticThreeStateRcEnv(MatplotlibRenderEnv):
    """Simulator for a three-state RC thermal model with exact discretization of Gaussian noise.

    This environment uses the matrix exponential approach to exactly discretize both the
    deterministic dynamics and the stochastic noise terms.

    Observation Space:
    ------------------

    The observation is a `Dict` space with the following structure:

    - "time": Dict
        - "quarter_hour": Box(0, 95, shape=(1,)) - quarter hour of the day (0=00:00, ..., 95=23:45)
        - "day_of_year": Box(0, 365, shape=(1,)) - day within a year
        - "day_of_week": Box(0, 6, shape=(1,)) - day of week (0=Monday, ..., 6=Sunday)
    - "state": Box(shape=(3,)) - [Ti, Th, Te] temperatures in Kelvin
        - Ti: indoor air temperature (0 to 303.15 K)
        - Th: radiator temperature (0 to 773.15 K)
        - Te: envelope temperature (0 to 303.15 K)
    - "forecast": Dict
        - "temperature": Box(shape=(N,)) - ambient temperature forecast [K]
        - "solar": Box(shape=(N,)) - solar radiation forecast [W/m²]
        - "price": Box(shape=(N,)) - electricity price forecast [EUR/kWh]

    Additional notes:
    - All forecasts are at 15-minute intervals starting from current time
    - N_forecast (or N) is the prediction horizon length (typically 96 for 24h)

    Action Space:
    -------------

    The action is a `ndarray` with shape `(1,)` which can take values in the range (0, 5000)
    representing the electric power to the radiators of the hvac system.

    Reward:
    -------
    The reward is a combination of comfort (r_comfort) and energy costs (r_costs)

    For specified lower and upper bound on comfort temperature, lb, and ub:
        r_comfort =
        +1 if lb <= Ti <= ub
        0  else

    The normalized energy costs reward is computed as
        r_costs = -(action/action_max) * (price/price_max)


    Terminates:
    -----------
    Does not terminate.

    Truncates:
    ----------
    Truncates after dataset's max_hours is reached in simulated time or when the
    historical data ends.

    Attributes:
        cfg: Configuration for the environment.
        dataset: HVAC dataset containing price, weather, and time features.
        N_forecast: Number of forecast steps.
        max_steps: Maximum number of simulation steps.
        params: Thermal parameters dictionary.
        state: Current system state [Ti, Th, Te].
        idx: Current index in the data.
        step_counter: Current step counter.
        Ad, Bd, Ed, Qd: Discrete-time system matrices.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    cfg: HvacEnvConfig
    dataset: HvacDataset
    forecaster: Forecaster
    trajectory_plots: dict[str, plt.Line2D] | None

    def __init__(
        self,
        render_mode: str | None = None,
        cfg: HvacEnvConfig | None = None,
        dataset: HvacDataset | None = None,
        forecaster: Forecaster | None = None,
    ) -> None:
        """Initialize the stochastic environment.

        Args:
            render_mode: Render mode for the environment.
            cfg: Configuration for the environment. If None, default configuration is used.
            dataset: Dataset for weather and price data. If None, default dataset is created.
            forecaster: Forecaster for generating predictions. If None, default forecaster
                is created.
        """
        super().__init__(render_mode=render_mode)

        self.cfg = HvacEnvConfig() if cfg is None else cfg

        # Use default thermal params if not provided
        if self.cfg.thermal_params is None:
            self.cfg.thermal_params = HydronicParameters()

        # Use provided dataset or create default
        self.dataset = HvacDataset() if dataset is None else dataset

        # Use provided forecaster or create default
        self.forecaster = Forecaster() if forecaster is None else forecaster

        # Setup forecast and simulation parameters
        self.N_forecast = 4 * self.forecaster.cfg.horizon_hours
        self.max_steps = -1  #  int(self.dataset.cfg.max_hours * 3600 / self.cfg.step_size)

        print("env N_forecast: ", self.N_forecast)

        # Setup observation space as Dict
        self.observation_space = spaces.Dict(
            {
                "time": spaces.Dict(
                    {
                        "quarter_hour": spaces.Box(
                            low=0.0, high=24 * 4 - 1, shape=(1,), dtype=np.float32
                        ),
                        "day_of_year": spaces.Box(
                            low=0.0, high=365.0, shape=(1,), dtype=np.float32
                        ),
                        "day_of_week": spaces.Box(low=0.0, high=6.0, shape=(1,), dtype=np.float32),
                    }
                ),
                "state": spaces.Box(
                    low=np.array(
                        [0.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [
                            convert_temperature(30.0, "celsius", "kelvin"),  # Ti
                            convert_temperature(500.0, "celsius", "kelvin"),  # Th
                            convert_temperature(30.0, "celsius", "kelvin"),  # Te
                        ],
                        dtype=np.float32,
                    ),
                ),
                "forecast": spaces.Dict(
                    {
                        "temperature": spaces.Box(
                            low=self.dataset.min["temperature_forecast"],
                            high=self.dataset.max["temperature_forecast"],
                            shape=(self.N_forecast,),
                            dtype=np.float32,
                        ),
                        "solar": spaces.Box(
                            low=self.dataset.min["solar_forecast"],
                            high=self.dataset.max["solar_forecast"],
                            shape=(self.N_forecast,),
                            dtype=np.float32,
                        ),
                        "price": spaces.Box(
                            low=self.dataset.min["price"],
                            high=self.dataset.max["price"],
                            shape=(self.N_forecast,),
                            dtype=np.float32,
                        ),
                    }
                ),
            }
        )

        # Action space: electric power to radiators in Watts
        self.max_power = 5000.0  # Maximum power in Watts
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.max_power], dtype=np.float32),
        )

        # Randomize thermal parameters if enabled
        if self.cfg.randomize_params:
            rng = np.random.default_rng(self.cfg.random_seed)
            self.params = self.cfg.thermal_params.randomize(rng, self.cfg.param_noise_scale)
        else:
            self.params = self.cfg.thermal_params

        # Precompute discrete-time system matrices
        self.Ad, self.Bd, self.Ed, self.Qd = compute_discrete_matrices(
            params=self.params,
            dt=self.cfg.step_size,
        )

        # Initialize step counter and data index
        self.step_counter = 0
        self.idx = 0

    def _get_observation(self) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Get the current observation.

        The observation includes time, state, and forecast data.

        Returns:
            dict: Observation dictionary with:
                - "time": dict with quarter_hour, day_of_year, day_of_week
                - "state": flat array [Ti, Th, Te]
                - "forecast": dict with temperature, solar, price arrays

        Notes:
            The observation structure is described in the class docstring.
        """
        quarter_hour, day_of_year = self.dataset.get_time_features(self.idx)
        day_of_week = self.dataset.index[self.idx].dayofweek  # Monday=0, Sunday=6

        forecasts = self.forecaster.get_forecast(
            idx=self.idx,
            dataset=self.dataset,
            N_forecast=self.N_forecast,
            np_random=self.np_random,
        )

        return {
            "time": {
                "quarter_hour": np.array([quarter_hour], dtype=np.float32),
                "day_of_year": np.array([day_of_year], dtype=np.float32),
                "day_of_week": np.array([day_of_week], dtype=np.float32),
            },
            "state": self.state.astype(np.float32),  # [Ti, Th, Te]
            "forecast": {
                "temperature": forecasts["temperature"].astype(np.float32),
                "solar": forecasts["solar"].astype(np.float32),
                "price": forecasts["price"].astype(np.float32),
            },
        }

    def _reward_function(self, state: np.ndarray, action: np.ndarray):
        """Compute the reward based on the current state and action.

        Args:
            state: Current state vector [Ti, Th, Te].
            action: Control input (electric power to radiators).

        Returns:
            Tuple containing:
                - reward: Combined comfort and energy cost reward.
                - reward_info: Dictionary with detailed reward components.
        """
        quarter_hour, _ = self.dataset.get_time_features(self.idx)
        lb, ub = set_temperature_limits(quarter_hours=quarter_hour)

        # Reward for comfort zone compliance
        success = int(lb <= state[0] <= ub)
        constraint_violation = 0.0
        if state[0] < lb:
            constraint_violation += lb - state[0]
        elif state[0] > ub:
            constraint_violation += state[0] - ub

        comfort_reward = -(constraint_violation**2 + abs(constraint_violation)) * 0.1

        # Reward for energy saving
        price = self.dataset.get_price(self.idx)[0]
        energy_consumption_normalized = np.abs(action[0]) / self.max_power

        price_normalized = price / self.dataset.max["price"]

        # True energy reward (for reporting)
        energy_reward = -50 * price_normalized * energy_consumption_normalized

        # Compute actual money spent: energy (kWh) * price (currency/kWh)
        # action is in Watts, step_size in seconds -> energy in kWh = W * s / 3600 / 1000
        energy_kwh = np.abs(action[0]) * self.cfg.step_size / 3600 / 1000
        money_spent = energy_kwh * price

        # Scale energy_reward
        reward = comfort_reward + energy_reward

        reward_info = {
            "price": price,
            "power": np.abs(action[0]),
            "energy_kwh": energy_kwh,
            "money_spent": money_spent,
            "comfort_reward": comfort_reward,
            "energy_reward": energy_reward,
            "success": success,
            "constraint_violation": constraint_violation,
        }

        return reward, reward_info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step within the environment.

        Args:
            action: Control input (electric power to radiators in Watts).

        Returns:
            Tuple containing:
                - observation: Next observation.
                - reward: Reward for this step.
                - terminated: Whether the episode terminated (always False).
                - truncated: Whether the episode was truncated.
                - info: Dictionary with 'time_forecast' and 'task' info.
        """
        # Get exogenous inputs
        exog = np.array(
            [
                self.dataset.get_temperature(self.idx)[0],  # Ambient temperature
                self.dataset.get_solar(self.idx)[0],  # Solar radiation
            ]
        )

        # Deterministic state update
        x_next = self.Ad @ self.state + self.Bd @ action + self.Ed @ exog

        # Add occupancy heat gains:
        # - Every weekday from 08:00 to 16:45: +300W
        day_of_week = self.dataset.index[self.idx].dayofweek  # Monday=0, Sunday=6
        quarter_hour, _ = self.dataset.get_time_features(self.idx)
        if day_of_week < 5 and 32 <= quarter_hour <= 67:
            x_next[0] += (1000.0 * self.cfg.step_size) / (self.cfg.thermal_params.dynamics.Ci)

        # Add Gaussian noise if enabled
        if self.cfg.enable_noise:
            # Sample from multivariate normal distribution with exact covariance
            noise = self.np_random.multivariate_normal(mean=np.zeros(3), cov=self.Qd)
            x_next += noise

        self.state = x_next
        self.idx += 1
        self.step_counter += 1

        time_forecast = self.dataset.get_time_forecast(self.idx, self.N_forecast)

        obs = self._get_observation()
        reward, reward_info = self._reward_function(state=self.state, action=action)

        # Check if episode should be truncated
        reached_max_steps = self.step_counter >= self.max_steps
        reached_end_of_data = self.idx >= len(self.dataset) - self.N_forecast
        truncated = reached_end_of_data or reached_max_steps

        # Check if dataset is exhausted (continual learning complete)
        terminated = self.dataset.is_exhausted(self.N_forecast)
        current_datetime = self.dataset.index[self.idx]
        info = {
            "time_forecast": time_forecast,
            "task": reward_info,
            "datetime": current_datetime.isoformat(),
        }

        return obs, reward, terminated, truncated, info

    def reset(
        self, state_0: np.ndarray | None = None, seed=None, options=None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to an initial state.

        Args:
            state_0: Initial state [Ti, Th, Te] in Kelvin. If None, samples a random Ti
            (between 19 and 23 deg Celsius) and computes Th and Te steady-state values.
            seed: Random seed for reproducibility.
            options: Additional reset options. Supported keys:
                - "split": Dataset split to use ("train", "test", or "all"). Defaults to "all".

        Returns:
            Tuple containing:
                - observation: Initial observation.
                - info: Dictionary with 'time_forecast' array.
        """
        super().reset(seed=seed)

        # Sample start index from dataset
        if options is not None and "mode" in options and options["mode"] == "train":
            split = "train"
        else:
            split = "test"
            # Reset test counter for reproducible evaluation (random mode only)
            if self.dataset.cfg.mode == "random" and seed is not None:
                self.dataset.reset_test_counter()

        self.idx, self.max_steps = self.dataset.sample_start_index(
            rng=self.np_random,
            horizon=self.N_forecast,
            split=split,
        )

        self.step_counter = 0

        if state_0 is None:
            Ti_ss = self.np_random.uniform(
                low=convert_temperature(19.0, "celsius", "kelvin"),
                high=convert_temperature(23.0, "celsius", "kelvin"),
            )

            _, Th_ss, Te_ss = compute_steady_state(
                Ti_ss=Ti_ss,
                temperature_ss=self.dataset.get_temperature(self.idx)[0],
                solar_ss=self.dataset.get_solar(self.idx)[0],
                params=self.params.dynamics,
            )

            self.state = np.array([Ti_ss, Th_ss, Te_ss])

        obs = self._get_observation()

        time_forecast = self.dataset.get_time_forecast(self.idx, self.N_forecast)
        current_datetime = self.dataset.index[self.idx]
        info = {
            "time_forecast": time_forecast,
            "datetime": current_datetime.isoformat(),
        }

        return obs, info
