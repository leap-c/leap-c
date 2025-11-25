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
from leap_c.examples.hvac.planner import HvacPlannerCtx
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
    randomize_params: bool = True
    param_noise_scale: float = 0.3
    random_seed: int = 0


class StochasticThreeStateRcEnv(MatplotlibRenderEnv):
    """Simulator for a three-state RC thermal model with exact discretization of Gaussian noise.

    This environment uses the matrix exponential approach to exactly discretize both the
    deterministic dynamics and the stochastic noise terms.

    Observation Space:
    ------------------

    The observation is a `ndarray` with shape `(5 + 3*15*horizon_hours,)` and dtype `np.float32`,
    where 3*5*horizon_hours includes forecasts for ambient temperature, solar irradiation, and
    electricity prices. The observation space looks as follows:

    | Num  | Observation                                  | Min | Max          |
    |------|----------------------------------------------|-----|--------------|
    | 0    | quarter hour of the day                      | 0   | 95           |
    | 1    | day of year                                  | 0   | 365          |
    | 2    | indoor air temperature Ti [K]                | 0   | 303.15       |
    | 3    | radiator temperature Th [K]                  | 0   | 773.15       |
    | 4    | envelope temperature Te [K]                  | 0   | 303.15       |
    | 5    | ambient temperature forecast t+0 [K]         | 0   | 313.15       |
    | 6    | ambient temperature forecast t+1 [K]         | 0   | 313.15       |
    | ...  | ambient temperature forecast t+N-1 [K]       | 0   | 313.15       |
    | 5+N  | solar radiation forecast t+0 [W/m²]          | 0   | 200.0        |
    | 6+N  | solar radiation forecast t+1 [W/m²]          | 0   | 200.0        |
    | ...  | solar radiation forecast t+N-1 [W/m²]        | 0   | 200.0        |
    | 5+2N | electricity price forecast t+0 [EUR/kWh]     | 0   | max(dataset) |
    | 6+2N | electricity price forecast t+1 [EUR/kWh]     | 0   | max(dataset) |
    | ...  | electricity price forecast t+N-1 [EUR/kWh]   | 0   | max(dataset) |

    Additional notes:
    - Quarter hour: 0=00:00, 1=00:15, 2=00:30, 3=00:45, ..., 95=23:45
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
        ctx: Context for the HVAC planner (used for rendering).
        dataset: HVAC dataset containing price, weather, and time features.
        N_forecast: Number of forecast steps.
        max_steps: Maximum number of simulation steps.
        params: Thermal parameters dictionary.
        state: Current system state [Ti, Th, Te].
        idx: Current index in the data.
        step_counter: Current step counter.
        Ad, Bd, Ed, Qd: Discrete-time system matrices.
        trajectory_plots: Dictionary of matplotlib line plots for rendering.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    cfg: HvacEnvConfig
    dataset: HvacDataset
    forecaster: Forecaster
    ctx: HvacPlannerCtx | None
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
        self.ctx = None
        self.N_forecast = 4 * self.forecaster.cfg.horizon_hours
        self.max_steps = int(self.dataset.cfg.max_hours * 3600 / self.cfg.step_size)

        print("env N_forecast: ", self.N_forecast)

        # Setup observation and action spaces
        obs_low = np.array(
            [
                0.0,  # quarter hour within a day
                0.0,  # day within a year
                0.0,  # Indoor temperature
                0.0,  # Radiator temperature
                0.0,  # Envelope temperature
            ]
            + [0.0] * self.N_forecast  # Ambient temperatures
            + [0.0] * self.N_forecast  # Solar radiation
            + [0.0] * self.N_forecast,  # Prices  TODO: Allow negative prices
            dtype=np.float32,
        )

        obs_high = np.array(
            [
                24 * 4 - 1,  # quarter hour within a day
                365,  # day within a year
                convert_temperature(30.0, "celsius", "kelvin"),  # Indoor temperature
                convert_temperature(500.0, "celsius", "kelvin"),  # Radiator temperature
                convert_temperature(30.0, "celsius", "kelvin"),  # Envelope temperature
            ]
            + [convert_temperature(40.0, "celsius", "kelvin")]
            * self.N_forecast  # Ambient temperatures
            + [200.0] * self.N_forecast  # Solar radiation
            + [self.dataset.price_max] * self.N_forecast,  # Prices
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

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

        self.trajectory_plots = None

    def _get_observation(self) -> np.ndarray:
        """Get the current observation.

        The observation includes time, state, ambient temperatures, solar radiation, and prices
        up to the prediction horizon.

        Returns:
            np.ndarray: Observation vector containing temporal, state, and forecast information

        Notes:
            The observation structure is described in the class docstring.
        """
        quarter_hour, day_of_year = self.dataset.get_time_features(self.idx)

        price_forecast = self.dataset.get_price(self.idx, self.N_forecast)

        forecasts = self.forecaster.get_forecast(
            idx=self.idx,
            data=self.dataset.data,
            N_forecast=self.N_forecast,
            np_random=self.np_random,
        )
        ambient_temperature_forecast = forecasts["temperature"]
        solar_forecast = forecasts["solar"]

        return np.concatenate(
            [
                np.array([quarter_hour, day_of_year], dtype=np.float32),
                self.state.astype(np.float32),
                ambient_temperature_forecast.astype(np.float32),
                solar_forecast.astype(np.float32),
                price_forecast.astype(np.float32),
            ]
        )

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
        comfort_reward = int(lb <= state[0] <= ub)

        # Reward for energy saving
        price = self.dataset.get_price(self.idx)[0]
        energy_consumption_normalized = np.abs(action[0]) / self.max_power

        price_normalized = price / self.dataset.price_max
        energy_reward = -price_normalized * energy_consumption_normalized

        # scale energy_reward
        reward = 0.5 * (comfort_reward + energy_reward)

        reward_info = {
            "prize": price,
            "energy": np.abs(action[0]),
            "comfort_reward": comfort_reward,
            "energy_reward": energy_reward,
            "success": comfort_reward,
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

        terminated = False  # We do not terminate
        info = {"time_forecast": time_forecast, "task": reward_info}

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

        self.idx = self.dataset.sample_start_index(
            rng=self.np_random,
            horizon=self.N_forecast,
            max_steps=self.max_steps,
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
        info = {"time_forecast": time_forecast}

        return obs, info

    def _render_setup(self):
        if self.render_mode == "human":
            plt.ion()

        # Create figure with 2 columns: left for states, right for params/actions
        self._fig, axes_array = plt.subplots(
            7, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [2, 1]}
        )
        self.axes = axes_array[:, 0]  # Left column for existing plots
        self.param_axes = axes_array[:, 1]  # Right column for parameters/actions
        
        # Adjust spacing
        self._fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.95, bottom=0.05)
        self._fig.suptitle("HVAC Controller Analysis", fontsize=14)

        # Initialize empty lines for each subplot
        self.trajectory_plots = {}

        ax = self.axes[0]
        (self.trajectory_plots["Ti"],) = ax.step([], [], where="post", label="Ti")
        (self.trajectory_plots["Ti_lb"],) = ax.step(
            [],
            [],
            where="post",
            label="Ti_lb",
            linestyle="--",
            color="black",
        )
        (self.trajectory_plots["Ti_ub"],) = ax.step(
            [],
            [],
            where="post",
            label="Ti_ub",
            linestyle="--",
            color="black",
        )
        ax.set_ylim(0, 30)
        ax.set_ylabel("Ti [°C]")
        ax.grid(visible=True, alpha=0.3)

        ax = self.axes[1]
        (self.trajectory_plots["Th"],) = ax.step([], [], where="post", label="Th")
        ax.set_ylim(-400, 400)
        ax.set_ylabel("Th [°C]")
        ax.grid(visible=True, alpha=0.3)

        ax = self.axes[2]
        (self.trajectory_plots["Te"],) = ax.step([], [], where="post", label="Te")
        ax.set_ylim(0, 30)
        ax.set_ylabel("Te [°C]")
        ax.grid(visible=True, alpha=0.3)

        # Heating power subplot
        ax = self.axes[3]
        (self.trajectory_plots["qh"],) = ax.step([], [], where="post", label="qh")
        ax.set_ylim(-5.05e3, 5.05e3)
        ax.set_ylabel("qh [kW]")
        ax.grid(visible=True, alpha=0.3)

        ax = self.axes[4]
        (self.trajectory_plots["price"],) = ax.step([], [], where="post", label="price")
        ax.set_ylim(0, 0.2)
        ax.set_ylabel("price [NOK/kWh]")

        ax = self.axes[5]
        (self.trajectory_plots["temperature"],) = ax.step([], [], where="post", label="temperature")
        ax.set_ylim(-30, 30)
        ax.set_ylabel("temperature [°C]")

        ax = self.axes[6]
        (self.trajectory_plots["solar"],) = ax.step([], [], where="post", label="solar")
        ax.set_ylim(0.0, 600.0)
        ax.set_ylabel("solar [W/m²]")

        for ax in self.axes:
            ax.set_xlim(0, self.N_forecast)
            ax.grid(visible=True, alpha=0.3)

        # Setup parameter/action plots in right column
        # Plot 0: ddqh (acceleration)
        self.param_axes[0].set_ylabel("ddqh [W/s²]")
        self.param_axes[0].grid(visible=True, alpha=0.3)
        (self.trajectory_plots["ddqh"],) = self.param_axes[0].step(
            [], [], where="post", label="ddqh", color="purple"
        )

        # Plot 1: q_Ti (weight)
        self.param_axes[1].set_ylabel("q_Ti")
        self.param_axes[1].grid(visible=True, alpha=0.3)
        (self.trajectory_plots["q_Ti"],) = self.param_axes[1].step(
            [], [], where="post", label="q_Ti", color="green"
        )

        # Plot 2: ref_Ti (reference temperature)
        self.param_axes[2].set_ylabel("ref_Ti [°C]")
        self.param_axes[2].grid(visible=True, alpha=0.3)
        (self.trajectory_plots["ref_Ti"],) = self.param_axes[2].step(
            [], [], where="post", label="ref_Ti", color="red"
        )

        # Plot 3: q_dqh (weight)
        self.param_axes[3].set_ylabel("q_dqh")
        self.param_axes[3].grid(visible=True, alpha=0.3)
        (self.trajectory_plots["q_dqh"],) = self.param_axes[3].step(
            [], [], where="post", label="q_dqh", color="orange"
        )

        # Plot 4: q_ddqh (weight)
        self.param_axes[4].set_ylabel("q_ddqh")
        self.param_axes[4].grid(visible=True, alpha=0.3)
        (self.trajectory_plots["q_ddqh"],) = self.param_axes[4].step(
            [], [], where="post", label="q_ddqh", color="brown"
        )

        # Plots 5-6: Empty for now
        self.param_axes[5].axis("off")
        self.param_axes[6].axis("off")

        # Set x-limits for parameter plots
        for ax in self.param_axes[:5]:
            ax.set_xlim(0, self.N_forecast)

    def _render_frame(self) -> np.ndarray | None:

        ctx: HvacPlannerCtx = self.ctx


        N_horizon = self.N_forecast

        quarter_hours = self.dataset.get_quarter_hours(self.idx, N_horizon)

        Ti_lb, Ti_ub = set_temperature_limits(quarter_hours=quarter_hours)

        obs = self._get_observation()
        temperature = obs[5 : 5 + self.N_forecast][:N_horizon]
        solar_forecast = obs[5 + self.N_forecast : 5 + 2 * self.N_forecast][:N_horizon]
        price_forecast = obs[5 + 2 * self.N_forecast : 5 + 3 * self.N_forecast][:N_horizon]

        self.trajectory_plots["Ti_lb"].set_data(
            range(N_horizon), convert_temperature(Ti_lb, "k", "c")
        )
        self.trajectory_plots["Ti_ub"].set_data(
            range(N_horizon), convert_temperature(Ti_ub, "k", "c")
        )

        if ctx is not None:
            x = ctx.iterate.x.reshape(-1, 5)
            self.trajectory_plots["Ti"].set_data(
                range(N_horizon - 3),
                convert_temperature(x[:, 0].flatten(), "k", "c"),
            )
            self.trajectory_plots["Th"].set_data(
                range(N_horizon - 3),
                convert_temperature(x[:, 1].flatten(), "k", "c"),
            )
            self.trajectory_plots["Te"].set_data(
                range(N_horizon - 3),
                convert_temperature(x[:, 2].flatten(), "k", "c"),
            )

            self.trajectory_plots["qh"].set_data(range(N_horizon - 1 - 3), x[:-1, 3].flatten())

        self.trajectory_plots["price"].set_data(range(N_horizon), price_forecast)
        self.trajectory_plots["temperature"].set_data(
            range(N_horizon), convert_temperature(temperature, "k", "c")
        )
        self.trajectory_plots["solar"].set_data(range(N_horizon), solar_forecast)

        # Update parameter/action plots if render_info is available
        if hasattr(ctx, "render_info") and ctx.render_info is not None:
            import pdb; pdb.set_trace()
            render_info = ctx.render_info
            
            # Plot action trajectory (ddqh)
            if "u_trajectory" in render_info:
                # First batch, all steps, first action
                u_traj = render_info["u_trajectory"][0, :, 0]
                self.trajectory_plots["ddqh"].set_data(range(len(u_traj)), u_traj)
            
            # Plot parameters if available (they're scalar, so repeat for visualization)
            param_names = ["q_Ti", "ref_Ti", "q_dqh", "q_ddqh"]
            for param_name in param_names:
                if param_name in render_info and param_name in self.trajectory_plots:
                    value = render_info[param_name][0]  # First batch element
                    # Convert ref_Ti from Kelvin to Celsius if it exists
                    if param_name == "ref_Ti":
                        value = convert_temperature(value, "k", "c")
                    # Plot as horizontal line to show constant parameter
                    self.trajectory_plots[param_name].set_data(
                        [0, N_horizon], [value, value]
                    )

    def set_ctx(self, ctx: HvacPlannerCtx) -> None:
        self.ctx: HvacPlannerCtx = ctx
