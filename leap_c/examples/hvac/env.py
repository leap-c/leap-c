from __future__ import annotations
import torch

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from config import (
    BestestHydronicParameters,
    BestestParameters,
)
from gymnasium import spaces
from scipy.constants import convert_temperature

from leap_c.examples.hvac.util import (
    BestestHydronicParameters,
    BestestParameters,
    load_price_data,
    load_weather_data,
    transcribe_continuous_state_space,
    transcribe_discrete_state_space,
)

from util import merge_price_weather_data, transcribe_continuous_state_space

# Constants
DAYLIGHT_START_HOUR = 6
DAYLIGHT_END_HOUR = 18
MEAN_AMBIENT_TEMPERATURE = convert_temperature(0, "celsius", "kelvin")
MAGNITUDE_AMBIENT_TEMPERATURE = 5
MAGNITUDE_SOLAR_RADIATION = 200


class StochasticThreeStateRcEnv(gym.Env):
    """
    Simulator for a three-state RC thermal model with exact discretization of Gaussian noise.

    This environment uses the matrix exponential approach to exactly discretize both the
    deterministic dynamics and the stochastic noise terms.
    """

    def __init__(
        self,
        params: None | BestestParameters = None,
        step_size: float = 900.0,  # Default 15 minutes
        start_time: pd.Timestamp | None = None,
        horizon_hours: int = 36,
        price_zone: str = "NO_1",
        price_data_path: Path | None = None,
        weather_data_path: Path | None = None,
        enable_noise: bool = True,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the stochastic environment.

        Args:
            params: Dictionary of thermal parameters
            step_size: Time step for the simulation in seconds
            ambient_temperature_function: Function for ambient temperature
            solar_radiation_function: Function for solar radiation
            enable_noise: Whether to include stochastic noise
            random_seed: Random seed for reproducibility
        """
        N_forecast = 4 * horizon_hours  # Number of forecasted ambient temperatures

        self.N_forecast = N_forecast

        print("env N_forecast: ", self.N_forecast)

        self.obs_low = np.array(
            [
                0.0,  # quarter hour within a day
                0.0,  # day within a year
                0.0,  # Indoor temperature
                0.0,  # Radiator temperature
                0.0,  # Envelope temperature
            ]
            + [0.0] * N_forecast  # Ambient temperatures
            + [0.0] * N_forecast  # Solar radiation
            + [0.0] * N_forecast,  # Prices  TODO: Allow negative prices
            dtype=np.float32,
        )

        self.obs_high = np.array(
            [
                24 * 4 - 1,  # quarter hour within a day
                365,  # day within a year
                convert_temperature(30.0, "celsius", "kelvin"),  # Indoor temperature
                convert_temperature(500.0, "celsius", "kelvin"),  # Radiator temperature
                convert_temperature(30.0, "celsius", "kelvin"),  # Envelope temperature
            ]
            + [convert_temperature(40.0, "celsius", "kelvin")]
            * N_forecast  # Ambient temperatures
            + [MAGNITUDE_SOLAR_RADIATION] * N_forecast  # Solar radiation
            + [np.inf] * N_forecast,  # Prices
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high)

        self.action_low = np.array([0.0], dtype=np.float32)
        self.action_high = np.array([5000.0], dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)

        # Store parameters
        self.params = (
            params if params is not None else BestestHydronicParameters().to_dict()
        )

        self.step_size = step_size
        self.enable_noise = enable_noise

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initial state variables [K]
        self.Ti = convert_temperature(20.0, "celsius", "kelvin")
        self.Th = convert_temperature(20.0, "celsius", "kelvin")
        self.Te = convert_temperature(20.0, "celsius", "kelvin")
        self.state_0 = np.array([self.Ti, self.Th, self.Te])

        # Precompute discrete-time matrices including noise covariance
        self.Ad, self.Bd, self.Ed, self.Qd = self._compute_discrete_matrices()

        if price_data_path is None:
            price_data_path = Path(__file__).parent / "spot_prices.csv"
        if weather_data_path is None:
            weather_data_path = Path(__file__).parent / "weather.csv"

        price_data = load_price_data(csv_path=price_data_path).resample("15T").ffill()

        weather_data = (
            load_weather_data(csv_path=weather_data_path)
            .resample("15T")
            .interpolate(method="linear")
        )

        data = merge_price_weather_data(
            price_data=price_data, weather_data=weather_data, merge_type="inner"
        )

        self.data = data

        # Rename NO1 to price
        self.data.rename(
            columns={price_zone: "price", "Tout_K": "Ta", "SolGlob": "solar"},
            inplace=True,
        )

        # Drop all columns except the ones we need
        self.data = self.data[["price", "Ta", "solar"]].copy()
        self.data["price"] = self.data["price"].astype(np.float32)
        self.data["Ta"] = self.data["Ta"].astype(np.float32)
        self.data["solar"] = self.data["solar"].astype(np.float32)

        self.start_time = start_time

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation including time, state, ambient temperatures,
        solar radiation, and prices.

        Returns:
            np.ndarray: Observation vector containing temporal, state, and forecast information

        | Num | Observation                                     |
        | --- | ------------------------------------------------|
        | 0   | quarter hour of day (0-95, 15-min intervals)    |
        | 1   | day of year (1-365/366)                         |
        | 2   | indoor air temperature Ti [K]                   |
        | 3   | radiator temperature Th [K]                     |
        | 4   | envelope temperature Te [K]                     |
        | 5   | ambient temperature forecast t+0 [K]            |
        | 6   | ambient temperature forecast t+1 [K]            |
        | ... | ambient temperature forecast t+N-1 [K]          |
        | 5+N | solar radiation forecast t+0 [W/m²]             |
        | 6+N | solar radiation forecast t+1 [W/m²]             |
        | ... | solar radiation forecast t+N-1 [W/m²]           |
        | 5+2N| electricity price forecast t+0 [EUR/kWh]        |
        | 6+2N| electricity price forecast t+1 [EUR/kWh]        |
        | ... | electricity price forecast t+N-1 [EUR/kWh]      |
        | 5+3N| datetime for forecast t+0 |
        | 6+3N| datetime for forecast t+1 |

        Total observation size: 5 + 3*N_forecast

        Notes:
            - Quarter hour: 0=00:00, 1=00:15, 2=00:30, 3=00:45, ..., 95=23:45
            - All forecasts are at 15-minute intervals starting from current time
            - N_forecast is the prediction horizon length (typically 96 for 24h)
        """
        datetime = self.data.index[self.idx]
        quarter_hour = (datetime.hour * 4 + datetime.minute // 15) % (24 * 4)
        day_of_year = datetime.timetuple().tm_yday

        price_forecast = (
            self.data["price"]
            .iloc[self.idx : self.idx + self.N_forecast + 1]
            .to_numpy()
        )

        # TODO: Implement forecasts for weather that is not a perfect copy of the data
        ambient_temperature_forecast = (
            self.data["Ta"].iloc[self.idx : self.idx + self.N_forecast + 1].to_numpy()
        )
        solar_radiation_forecast = (
            self.data["solar"]
            .iloc[self.idx : self.idx + self.N_forecast + 1]
            .to_numpy()
        )

        datetime_forecast = self.data.index[self.idx : self.idx + self.N_forecast + 1]

        return np.concatenate(
            [
                np.array([quarter_hour, day_of_year], dtype=np.float32),
                self.state.astype(np.float32),
                ambient_temperature_forecast.astype(np.float32),
                solar_radiation_forecast.astype(np.float32),
                price_forecast.astype(np.float32),
                datetime_forecast,
            ]
        )

    def _compute_discrete_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute discrete-time matrices using exact discretization via matrix exponential.
        This includes both deterministic dynamics and noise covariance.
        """
        # Create noise intensity matrix Σ from parameters
        # The stochastic terms are σᵢω̇ᵢ, σₕω̇ₕ, σₑω̇ₑ
        sigma_i = np.exp(self.params["sigmai"])
        sigma_h = np.exp(self.params["sigmah"])
        sigma_e = np.exp(self.params["sigmae"])

        # Compute continuous-time Ac
        Ac, _, _ = transcribe_continuous_state_space(
            Ac=np.zeros((3, 3)),
            Bc=np.zeros((3, 1)),
            Ec=np.zeros((3, 2)),
            params=self.params,
        )

        Qd = self._compute_noise_covariance(
            Ac=Ac,
            Sigma=np.diag([sigma_i, sigma_h, sigma_e]),
            dt=self.step_size,
        )

        # Compute discrete-time state-space matrices
        Ad, Bd, Ed = transcribe_discrete_state_space(
            Ad=np.zeros((3, 3)),
            Bd=np.zeros((3, 1)),
            Ed=np.zeros((3, 2)),
            dt=self.step_size,
            params=self.params,
        )

        return Ad, Bd, Ed, Qd

    def _compute_noise_covariance(
        self, Ac: np.ndarray, Sigma: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        TODO: Check if this is correct. See, e.g., Farrell, J. Sec 4.7.2
        Compute the exact discrete-time noise covariance matrix using matrix exponential.
        Q_d = ∫₀^Δt e^(Aτ) Σ Σᵀ e^(Aᵀτ) dτ.

        Args:
            Ac: Continuous-time system matrix
            Sigma: Noise intensity matrix (diagonal)
            dt: Sampling time

        Returns:
            Qd: Discrete-time noise covariance matrix
        """
        n = Ac.shape[0]  # State dimension (3)

        # Create the augmented matrix for computing the noise covariance integral
        # [ A    Σ Σᵀ ]
        # [ 0      -Aᵀ ]
        SigmaSigmaT = Sigma @ Sigma.T

        # Augmented matrix (6x6)
        M = np.block([[Ac, SigmaSigmaT], [np.zeros((n, n)), -Ac.T]])

        # Matrix exponential of augmented system
        exp_M = scipy.linalg.expm(M * dt)

        # Extract the noise covariance from the upper-right block
        # Qd = e^(A*dt) * (upper-right block of exp_M)
        Ad = exp_M[:n, :n]
        Phi = exp_M[:n, n:]

        # The discrete-time covariance is Qd = Ad @ Phi
        return Ad @ Phi

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Perform a simulation step with exact discrete-time dynamics including noise.

        Args:
            action: Control input (heat input to radiator)
            time: Current time in seconds since the start of the simulation

        Returns:
            next_state: Next state after applying control input and noise
        """
        # Get exogenous inputs
        exog = np.array(
            [
                self.data["Ta"].iloc[self.idx],  # Ambient temperature
                self.data["solar"].iloc[self.idx],  # Solar radiation
            ]
        )

        # Deterministic state update
        x_next = self.Ad @ self.state + self.Bd @ action + self.Ed @ exog

        # Add Gaussian noise if enabled
        if self.enable_noise:
            # Sample from multivariate normal distribution with exact covariance
            noise = np.random.default_rng().multivariate_normal(
                mean=np.zeros(3), cov=self.Qd
            )
            x_next += noise

        self.state = x_next
        self.idx += 1

        self.Ti, self.Th, self.Te = self.state[0], self.state[1], self.state[2]

        return self._get_observation()

    def reset(self, state_0: np.ndarray | None = None) -> None:
        """Reset the model state to initial values."""
        if state_0 is None:
            state_0 = self.state_0
        self.state = state_0.copy()

        if self.start_time is not None:
            self.idx = self.data.index.get_loc(self.start_time, method="nearest")
        else:
            self.idx = 0

        obs = self._get_observation()
        info = {}

        return obs, info

    def get_noise_statistics(self) -> dict:
        """
        Get statistics about the noise model.

        Returns:
            Dictionary with noise statistics
        """
        sigma_i = np.exp(self.params["sigmai"])
        sigma_h = np.exp(self.params["sigmah"])
        sigma_e = np.exp(self.params["sigmae"])

        return {
            "continuous_noise_intensities": {
                "sigma_i": sigma_i,
                "sigma_h": sigma_h,
                "sigma_e": sigma_e,
            },
            "discrete_noise_covariance": self.Qd,
            "discrete_noise_std": np.sqrt(np.diag(self.Qd)),
            "step_size": self.step_size,
        }


def decompose_observation(obs: np.ndarray) -> tuple:
    """
    Decompose the observation vector into its components.

    Args:
        obs: Observation vector from the environment.

    Returns:
        Tuple containing:
        - quarter_hour: Current quarter hour of the day (0-95)
        - day_of_year: Current day of the year (1-365)
        - Ti: Indoor air temperature in Kelvin
        - Th: Radiator temperature in Kelvin
        - Te: Envelope temperature in Kelvin
        - Ta_forecast: Ambient temperature forecast for the next N steps
        - solar_forecast: Solar radiation forecast for the next N steps
        - price_forecast: Electricity price forecast for the next N steps
    """
    N_forecast = (len(obs) - 5) // 4

    quarter_hour = obs[0]
    day_of_year = obs[1]
    Ti = obs[2]
    Th = obs[3]
    Te = obs[4]

    Ta_forecast = obs[5 : 5 + 1 * N_forecast]
    solar_forecast = obs[5 + 1 * N_forecast : 5 + 2 * N_forecast]
    price_forecast = obs[5 + 2 * N_forecast : 5 + 3 * N_forecast]
    # Assuming datetime is the last part of the observation
    datetime_forecast = obs[5 + 3 * N_forecast :]

    for forecast in [Ta_forecast, solar_forecast, price_forecast, datetime_forecast]:
        assert forecast.ndim == 1, (
            f"Expected 1D array for forecast, got {forecast.ndim}D array"
        )
        assert len(forecast) == N_forecast, (
            f"Expected {N_forecast} forecasts, got {len(forecast)}"
        )

    return (
        quarter_hour,
        day_of_year,
        Ti,
        Th,
        Te,
        Ta_forecast,
        solar_forecast,
        price_forecast,
        datetime_forecast,
    )


if __name__ == "__main__":
    days = 100
    n_steps = days * 24 * 4  # 4 time steps per hour

    env = StochasticThreeStateRcEnv(
        step_size=900.0,  # 15 minutes in seconds
        enable_noise=False,
    )

    x = []
    u = []
    Ta = []  # Use the first forecasted ambient temperature
    solar = []  # Use the first forecasted datetime
    time = []  # Use the first forecasted datetime

    # Run simulation
    obs, _ = env.reset()
    for _ in range(n_steps):
        (
            quarter_hour,
            day_of_year,
            Ti,
            Th,
            Te,
            Ta_forecast,
            solar_forecast,
            price_forecast,
            datetime_forecast,
        ) = decompose_observation(obs)

        # Get current inputs
        action = np.array([0.0], dtype=np.float32)  # No control input

        x.append(np.array([Ti, Th, Te], dtype=np.float32))
        u.append(action)
        Ta.append(Ta_forecast[0])
        solar.append(solar_forecast[0])
        time.append(datetime_forecast[0])

        # Update state using the integrator
        obs = env.step(action)

    x = np.array(x)
    u = np.array(u)
    Ta = np.array(Ta)
    solar = np.array(solar)
    time = np.array(time)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.ylabel("Temperature (K)")
    plt.plot(time, x[:, 0], label="Indoor (Ti)")
    plt.plot(time, x[:, 1], label="Radiator (Th)")
    plt.plot(time, x[:, 2], label="Envelope (Te)")
    plt.plot(time, Ta, label="Ambient (Ta)")
    plt.grid(visible=True)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.ylabel("Solar Radiation (W/m²)")
    plt.plot(time, solar, label="Solar Radiation")
    plt.grid(visible=True)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.ylabel("Control Input (W)")
    plt.plot(time, u[:, 0], label="Heat Input (u)")
    plt.xlabel("Time (s)")
    plt.grid(visible=True)
    plt.legend()
    plt.tight_layout()
    plt.show()
