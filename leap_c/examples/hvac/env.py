from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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

    def __post_init__(self):
        if self.step_size != 900.0:
            raise NotImplementedError("Only step_size of 900s is currently supported.")


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
        self.max_steps = -1  #  int(self.dataset.cfg.max_hours * 3600 / self.cfg.step_size)

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
            + [self.dataset.min["temperature_forecast"]] * self.N_forecast  # Ambient temperatures
            + [self.dataset.min["solar_forecast"]] * self.N_forecast  # Solar radiation
            + [self.dataset.min["price"]] * self.N_forecast,  # Prices  TODO: Allow negative prices
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
            + [self.dataset.max["temperature_forecast"]] * self.N_forecast  # Ambient temperatures
            + [self.dataset.max["solar_forecast"]] * self.N_forecast  # Solar radiation
            + [self.dataset.max["price"]] * self.N_forecast,  # Prices
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

        # Initialize history buffers for rendering (FIFO with max 100 steps)
        self.history_length = 100
        self.history = {
            "Ti": deque(maxlen=self.history_length),
            "Th": deque(maxlen=self.history_length),
            "Te": deque(maxlen=self.history_length),
            "qh": deque(maxlen=self.history_length),
            "ref_Ti": deque(maxlen=self.history_length),
            "lb_Ti": deque(maxlen=self.history_length),
            "ub_Ti": deque(maxlen=self.history_length),
            "temperature": deque(maxlen=self.history_length),
            "solar": deque(maxlen=self.history_length),
            "price": deque(maxlen=self.history_length),
            "ddqh": deque(maxlen=self.history_length),
            "q_Ti": deque(maxlen=self.history_length),
        }

        self.param_axes_limits_set = False

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

        forecasts = self.forecaster.get_forecast(
            idx=self.idx,
            dataset=self.dataset,
            N_forecast=self.N_forecast,
            np_random=self.np_random,
        )

        return np.concatenate(
            [
                np.array([quarter_hour, day_of_year], dtype=np.float32),
                self.state.astype(np.float32),
                forecasts["temperature"].astype(np.float32),
                forecasts["solar"].astype(np.float32),
                forecasts["price"].astype(np.float32),
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
        success = int(lb <= state[0] <= ub)
        constraint_violation = 0.0
        if state[0] < lb:
            constraint_violation += lb - state[0]
        elif state[0] > ub:
            constraint_violation += state[0] - ub

        comfort_reward = -constraint_violation ** 2 - abs(constraint_violation)

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

    def _render_setup(self):
        if self.render_mode == "human":
            plt.ion()

        # Create figure with 2 columns: left for states, right for params/actions
        self._fig = plt.figure(figsize=(16, 10))

        gs = gridspec.GridSpec(7, 2, figure=self._fig)

        # Create left column axes (7 rows: (Ti, Th, Te, qh, price, temperature, solar))
        left_axes = [self._fig.add_subplot(gs[i, 0]) for i in range(7)]

        # Create right column axes (3 evenly spaced rows spanning full height)
        right_axes = [
            self._fig.add_subplot(gs[0:3, 1]),
            self._fig.add_subplot(gs[3:5, 1]),
            self._fig.add_subplot(gs[5:7, 1]),
        ]

        # Combine into 2D array structure for compatibility
        self.axes = np.empty((7, 2), dtype=object)
        for i in range(7):
            self.axes[i, 0] = left_axes[i]
        for i in range(3):
            self.axes[i, 1] = right_axes[i]

        # Fill remaining right column slots with None
        for i in range(3, 7):
            self.axes[i, 1] = None

        # Adjust spacing
        self._fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.95, bottom=0.05)
        self._fig.suptitle("HVAC Controller Analysis", fontsize=14)

        # Initialize empty lines for each subplot
        self.trajectory_plots = {}

        ax: plt.Axes = self.axes[0, 0]
        (self.trajectory_plots["Ti"],) = ax.step([], [], where="post", label="Ti")
        (self.trajectory_plots["ref_Ti"],) = ax.step(
            [],
            [],
            where="post",
            linestyle="-",
            color="red",
            label="Ti_ref",
        )
        (self.trajectory_plots["lb_Ti"],) = ax.step(
            [],
            [],
            where="post",
            label="Ti_lb",
            linestyle="--",
            color="black",
        )
        (self.trajectory_plots["ub_Ti"],) = ax.step(
            [],
            [],
            where="post",
            label="Ti_ub",
            linestyle="--",
            color="black",
        )
        ax.set(ylim=(0, 30), ylabel="Ti [°C]")
        ax.grid(visible=True, alpha=0.3)

        ax: plt.Axes = self.axes[1, 0]
        (self.trajectory_plots["Th"],) = ax.step([], [], where="post", label="Th")
        ax.set(ylim=(-400, 400), ylabel="Th [°C]")
        ax.grid(visible=True, alpha=0.3)

        ax: plt.Axes = self.axes[2, 0]
        (self.trajectory_plots["Te"],) = ax.step([], [], where="post", label="Te")
        ax.set(ylim=(0, 30), ylabel="Te [°C]")
        ax.grid(visible=True, alpha=0.3)

        # Heating power subplot
        ax: plt.Axes = self.axes[3, 0]
        (self.trajectory_plots["qh"],) = ax.step([], [], where="post", label="qh")
        ax.set(ylim=(0.0, 5.10e3), ylabel="qh [kW]")
        ax.grid(visible=True, alpha=0.3)

        ax: plt.Axes = self.axes[4, 0]
        (self.trajectory_plots["price_observation"],) = ax.step(
            [],
            [],
            where="post",
            label="price observation",
        )
        (self.trajectory_plots["price"],) = ax.step(
            [],
            [],
            where="post",
            label="price parameter",
        )
        ax.set(
            ylim=(self.dataset.min["price"], self.dataset.max["price"]),
            ylabel="price [NOK/kWh]",
        )

        ax: plt.Axes = self.axes[5, 0]
        (self.trajectory_plots["temperature_observation"],) = ax.step(
            [],
            [],
            where="post",
            label="temperature observation",
        )
        (self.trajectory_plots["temperature"],) = ax.step(
            [],
            [],
            where="post",
            label="temperature parameter",
        )
        ax.set(
            ylim=(
                convert_temperature(
                    val=self.dataset.min["temperature"],
                    old_scale="k",
                    new_scale="c",
                ),
                convert_temperature(
                    val=self.dataset.max["temperature"],
                    old_scale="k",
                    new_scale="c",
                ),
            ),
            ylabel="temperature [°C]",
        )

        ax: plt.Axes = self.axes[6, 0]
        (self.trajectory_plots["solar_observation"],) = ax.step(
            [],
            [],
            where="post",
            label="solar observation",
        )
        (self.trajectory_plots["solar"],) = ax.step(
            [],
            [],
            where="post",
            label="solar parameter",
        )
        ax.set(
            ylim=(self.dataset.min["solar"], self.dataset.max["solar"]),
            ylabel="solar [W/m²]",
        )

        # 2D plot of ref_Ti vs q_Ti
        ax: plt.Axes = self.axes[0, 1]
        (self.trajectory_plots["ref_Ti_over_q_Ti"],) = ax.plot(
            [], [], "o-", markersize=3, label="ref_Ti vs q_Ti"
        )
        ax.set(
            xlabel="ref_Ti [°C]",
            ylabel="q_Ti [-]",
            title="Reference Temperature vs Weight",
        )
        ax.grid(visible=True, alpha=0.3)
        ax.legend(loc="upper right")

        # Histogram of ref_Ti
        ax: plt.Axes = self.axes[1, 1]
        ax.set(
            xlabel="ref_Ti [°C]",
            ylabel="Frequency",
            title="ref_Ti Distribution",
        )
        ax.grid(visible=True, alpha=0.3, axis="y")

        # Histogram of q_Ti
        ax: plt.Axes = self.axes[2, 1]
        ax.set(
            xlabel="q_Ti [-]",
            ylabel="Frequency",
            title="q_Ti Distribution",
        )
        ax.grid(visible=True, alpha=0.3, axis="y")

        # Set x-limits and legend for trajectory plots
        for ax in self.axes[:, 0]:
            ax.set(xlim=(-self.history_length, self.N_forecast))
            ax.grid(visible=True, alpha=0.3)
            ax.legend(loc="upper right")
            # Add vertical line at x=0 to mark current time
            ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5, alpha=0.7)

    def _render_frame(self) -> np.ndarray | None:
        ctx: HvacPlannerCtx = self.ctx

        obs = self._get_observation()
        temperature_forecast = obs[5 : 5 + self.N_forecast]
        solar_forecast = obs[5 + self.N_forecast : 5 + 2 * self.N_forecast]
        price_forecast = obs[5 + 2 * self.N_forecast : 5 + 3 * self.N_forecast]

        # Update history buffers with current values (at time step -1)
        # Get current state for history
        if ctx is not None and hasattr(ctx, "render_info") and ctx.render_info is not None:
            render_info: dict[str, np.ndarray] = ctx.render_info

            # Add current values to history (these will appear at position -1)
            self.history["Ti"].append(convert_temperature(self.state[0], "k", "c"))
            self.history["Th"].append(convert_temperature(self.state[1], "k", "c"))
            self.history["Te"].append(convert_temperature(self.state[2], "k", "c"))

            if "ref_Ti" in render_info:
                self.history["ref_Ti"].append(render_info["ref_Ti"].flatten()[0])
            if "lb_Ti" in render_info:
                self.history["lb_Ti"].append(render_info["lb_Ti"].flatten()[0])
            if "ub_Ti" in render_info:
                self.history["ub_Ti"].append(render_info["ub_Ti"].flatten()[0])
            if "temperature" in render_info:
                self.history["temperature"].append(render_info["temperature"].flatten()[0])
            if "solar" in render_info:
                self.history["solar"].append(render_info["solar"].flatten()[0])
            if "price" in render_info:
                self.history["price"].append(render_info["price"].flatten()[0])
            if "ddqh" in render_info:
                self.history["ddqh"].append(render_info["ddqh"].flatten()[0])
            if "q_Ti" in render_info:
                self.history["q_Ti"].append(render_info["q_Ti"].flatten()[0])
            if "qh" in render_info:
                self.history["qh"].append(render_info["qh"].flatten()[0])

        # Plot forecast observations (future only, from 0 to N_forecast)
        self.trajectory_plots["price_observation"].set_data(range(self.N_forecast), price_forecast)
        self.trajectory_plots["temperature_observation"].set_data(
            range(self.N_forecast),
            convert_temperature(temperature_forecast, "k", "c"),
        )
        self.trajectory_plots["solar_observation"].set_data(range(self.N_forecast), solar_forecast)

        # Update parameter/action plots if render_info is available
        if hasattr(ctx, "render_info") and ctx.render_info is not None:
            render_info: dict[str, np.ndarray] = ctx.render_info

            for key in [
                "lb_Ti",
                "ub_Ti",
                "temperature",
                "price",
                "solar",
                "qh",
                "Ti",
                "Th",
                "Te",
            ]:
                if key in render_info:
                    # Future predictions
                    future_x = np.arange(0, len(render_info[key].flatten()))
                    future_y = render_info[key].flatten()

                    # Historical data
                    hist_x = np.arange(-len(self.history[key]), 0)
                    hist_y = np.array(list(self.history[key]))

                    self.trajectory_plots[key].set_data(
                        np.concatenate([hist_x, future_x]),
                        np.concatenate([hist_y, future_y]),
                    )

            # Update 2D plot: ref_Ti vs q_Ti
            if "ref_Ti" in render_info and "q_Ti" in render_info:
                if not self.param_axes_limits_set:
                    # Draw black rectangle showing the valid parameter region
                    width = render_info["ref_Ti_max"] - render_info["ref_Ti_min"]
                    height = render_info["q_Ti_max"] - render_info["q_Ti_min"]
                    self.axes[0, 1].add_patch(
                        mpatches.Rectangle(
                            xy=(render_info["ref_Ti_min"], render_info["q_Ti_min"]),
                            width=width,
                            height=height,
                            linewidth=2,
                            edgecolor="black",
                            facecolor="none",
                            linestyle="-",
                        )
                    )

                    self.axes[0, 1].set_xlim(
                        render_info["ref_Ti_min"] - 0.05 * width,
                        render_info["ref_Ti_max"] + 0.05 * width,
                    )
                    self.axes[0, 1].set_ylim(
                        render_info["q_Ti_min"] - 0.05 * height,
                        render_info["q_Ti_max"] + 0.05 * height,
                    )

                    # Set histogram x-limits (only once)
                    self.axes[1, 1].set_xlim(render_info["ref_Ti_min"], render_info["ref_Ti_max"])
                    self.axes[2, 1].set_xlim(render_info["q_Ti_min"], render_info["q_Ti_max"])

                    self.param_axes_limits_set = True

                # Historical data
                hist_ref_Ti = np.array(list(self.history["ref_Ti"]))
                hist_q_Ti = np.array(list(self.history["q_Ti"]))

                # Future predictions
                future_ref_Ti = render_info["ref_Ti"].flatten()
                future_q_Ti = render_info["q_Ti"].flatten()

                # Combine history and future
                combined_ref_Ti = np.concatenate([hist_ref_Ti, future_ref_Ti])
                combined_q_Ti = np.concatenate([hist_q_Ti, future_q_Ti])

                self.trajectory_plots["ref_Ti_over_q_Ti"].set_data(combined_ref_Ti, combined_q_Ti)

                # Update histograms
                # Histogram for ref_Ti
                self.axes[1, 1].clear()
                if len(combined_ref_Ti) > 0:
                    # Flatten to ensure 1D array and convert to numpy
                    data_ref_Ti = np.asarray(combined_ref_Ti).flatten()
                    # Check if range is valid
                    if (
                        render_info["ref_Ti_max"] > render_info["ref_Ti_min"]
                        and len(data_ref_Ti) > 0
                    ):
                        self.axes[1, 1].hist(
                            data_ref_Ti,
                            bins=20,
                            range=(
                                float(render_info["ref_Ti_min"]),
                                float(render_info["ref_Ti_max"]),
                            ),
                            color="blue",
                            alpha=0.7,
                            edgecolor="black",
                        )
                # Re-apply formatting after clear (clear() removes all properties)
                self.axes[1, 1].set(
                    xlabel="ref_Ti [°C]",
                    ylabel="Frequency",
                )
                self.axes[1, 1].grid(visible=True, alpha=0.3, axis="y")

                # Histogram for q_Ti
                self.axes[2, 1].clear()
                if len(combined_q_Ti) > 0:
                    # Flatten to ensure 1D array and convert to numpy
                    data_q_Ti = np.asarray(combined_q_Ti).flatten()
                    # Check if range is valid
                    if render_info["q_Ti_max"] > render_info["q_Ti_min"] and len(data_q_Ti) > 0:
                        self.axes[2, 1].hist(
                            data_q_Ti,
                            bins=20,
                            range=(
                                float(render_info["q_Ti_min"]),
                                float(render_info["q_Ti_max"]),
                            ),
                            color="blue",
                            alpha=0.7,
                            edgecolor="black",
                        )
                # Re-apply formatting after clear (clear() removes all properties)
                self.axes[2, 1].set(
                    xlabel="q_Ti [-]",
                    ylabel="Frequency",
                )
                self.axes[2, 1].grid(visible=True, alpha=0.3, axis="y")

    def set_ctx(self, ctx: HvacPlannerCtx) -> None:
        self.ctx: HvacPlannerCtx = ctx
