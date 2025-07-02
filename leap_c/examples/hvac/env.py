from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.constants import convert_temperature

from leap_c.examples.hvac.util import (
    BestestHydronicParameters,
    BestestParameters,
    transcribe_continuous_state_space,
    transcribe_discrete_state_space,
)


from leap_c.examples.hvac.util import (
    BestestHydronicParameters,
    EnergyPriceProfile,
    create_constant_comfort_bounds,
    create_constant_disturbance,
    create_disturbance_from_weather,
    create_price_profile_from_spot_sprices,
    create_realistic_comfort_bounds,
    create_time_of_use_energy_costs,
    load_price_data,
    load_weather_data,
    plot_ocp_results,
    transcribe_discrete_state_space,
)

from pathlib import Path

import pandas as pd

from gymnasium import spaces

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
        ambient_temperature_function: Callable | None = None,
        solar_radiation_function: Callable | None = None,
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

        # Set default functions if not provided
        if ambient_temperature_function is None:
            self.ambient_temperature_function = self._get_default_ambient_temperature
        else:
            self.ambient_temperature_function = ambient_temperature_function

        if solar_radiation_function is None:
            self.solar_radiation_function = self._get_default_solar_radiation
        else:
            self.solar_radiation_function = solar_radiation_function

        if price_data_path is None:
            price_data_path = Path(__file__).parent / "spot_prices.csv"
        if weather_data_path is None:
            weather_data_path = Path(__file__).parent / "weather.csv"

        # price_data = resample_prices_to_quarters(
        #     load_price_data(csv_path=price_data_path)
        # )

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
            self.data["price"].iloc[self.idx : self.idx + self.N_forecast].to_numpy()
        )

        # TODO: Implement forecasts for weather that is not a perfect copy of the data
        ambient_temperature_forecast = (
            self.data["Ta"].iloc[self.idx : self.idx + self.N_forecast].to_numpy()
        )
        solar_radiation_forecast = (
            self.data["solar"].iloc[self.idx : self.idx + self.N_forecast].to_numpy()
        )

        return np.concatenate(
            [
                np.array([quarter_hour, day_of_year], dtype=np.float32),
                self.state.astype(np.float32),
                ambient_temperature_forecast.astype(np.float32),
                solar_radiation_forecast.astype(np.float32),
                price_forecast.astype(np.float32),
            ]
        )

    def _compute_discrete_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute discrete-time matrices using exact discretization via matrix exponential.
        This includes both deterministic dynamics and noise covariance.
        """
        # Get continuous-time matrices
        Ac, Bc, Ec = transcribe_continuous_state_space(
            Ac=np.zeros((3, 3)),
            Bc=np.zeros((3, 1)),
            Ec=np.zeros((3, 2)),
            params=self.params,
        )

        # Compute discrete-time state-space matrices
        Ad, Bd, Ed = transcribe_discrete_state_space(
            Ad=np.zeros((3, 3)),
            Bd=np.zeros((3, 1)),
            Ed=np.zeros((3, 2)),
            dt=self.step_size,
            params=self.params,
        )

        # Create noise intensity matrix Σ from parameters
        # The stochastic terms are σᵢω̇ᵢ, σₕω̇ₕ, σₑω̇ₑ
        sigma_i = np.exp(self.params["sigmai"])
        sigma_h = np.exp(self.params["sigmah"])
        sigma_e = np.exp(self.params["sigmae"])

        Qd = self._compute_noise_covariance(
            Ac=Ac,
            Sigma=np.diag([sigma_i, sigma_h, sigma_e]),
            dt=self.step_size,
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

    def _get_default_ambient_temperature(self, t: float) -> float:
        """Get default ambient temperature function value."""
        MEAN_AMBIENT_TEMPERATURE = convert_temperature(0, "celsius", "kelvin")
        MAGNITUDE_AMBIENT_TEMPERATURE = 5
        return MEAN_AMBIENT_TEMPERATURE + MAGNITUDE_AMBIENT_TEMPERATURE * np.sin(
            2 * np.pi * t / (24 * 3600)
        )

    def _get_default_solar_radiation(self, t: float) -> float:
        """Get default solar radiation function value."""
        DAYLIGHT_START_HOUR = 6
        DAYLIGHT_END_HOUR = 18
        MAGNITUDE_SOLAR_RADIATION = 200

        hour = (t % (24 * 3600)) / 3600
        if DAYLIGHT_START_HOUR <= hour <= DAYLIGHT_END_HOUR:
            return MAGNITUDE_SOLAR_RADIATION * np.sin(np.pi * (hour - 6) / 12)
        return 0.0


def resample_prices_to_quarters(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample hourly price data to 15-minute intervals.
    Each hour's price is kept constant for the following four 15-minute periods.

    Parameters:
    -----------
    price_data : pd.DataFrame
        DataFrame with hourly price data indexed by timestamp

    Returns:
    --------
    pd.DataFrame
        DataFrame with 15-minute price data
    """
    # Ensure the index is datetime
    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data.index = pd.to_datetime(price_data.index)

    print(f"Original data shape: {price_data.shape}")
    print(f"Original frequency: {pd.infer_freq(price_data.index)}")
    print(f"Time range: {price_data.index.min()} to {price_data.index.max()}")

    # Resample to 15-minute intervals using forward fill
    # This keeps each hour's price constant for the following four quarters
    price_quarterly = price_data.resample("15T").ffill()

    print(f"\nResampled data shape: {price_quarterly.shape}")
    print("New frequency: 15 minutes")
    print(f"Time range: {price_quarterly.index.min()} to {price_quarterly.index.max()}")
    print(f"Expansion factor: {price_quarterly.shape[0] / price_data.shape[0]:.1f}x")

    return price_quarterly


def merge_price_weather_data(
    price_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    merge_type: str = "inner",
) -> pd.DataFrame:
    """
    Merge price and weather dataframes on their timestamp indices.

    Parameters:
    -----------
    price_data : pd.DataFrame
        DataFrame with price data indexed by timestamp
    weather_data : pd.DataFrame
        DataFrame with weather data indexed by timestamp
    merge_type : str
        Type of merge: 'inner', 'outer', 'left', 'right'

    Returns:
    --------
    pd.DataFrame
        Merged dataframe
    """
    print(
        f"Price data time range: {price_data.index.min()} to {price_data.index.max()}"
    )
    print(
        f"Weather data time range: {weather_data.index.min()} to {weather_data.index.max()}"
    )
    print(f"Price data shape: {price_data.shape}")
    print(f"Weather data shape: {weather_data.shape}")

    # Ensure both indices are datetime and timezone-aware
    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data.index = pd.to_datetime(price_data.index)
    if not isinstance(weather_data.index, pd.DatetimeIndex):
        weather_data.index = pd.to_datetime(weather_data.index)

    # Perform the merge based on the timestamp index
    if merge_type == "inner":
        # Only keep timestamps that exist in both dataframes
        merged_df = price_data.join(weather_data, how="inner")
        print(f"\nInner join: {merged_df.shape[0]} overlapping timestamps")

    elif merge_type == "outer":
        # Keep all timestamps from both dataframes
        merged_df = price_data.join(weather_data, how="outer")
        print(f"\nOuter join: {merged_df.shape[0]} total timestamps")

    elif merge_type == "left":
        # Keep all price data timestamps, add weather where available
        merged_df = price_data.join(weather_data, how="left")
        print(f"\nLeft join: {merged_df.shape[0]} timestamps (all price data)")

    elif merge_type == "right":
        # Keep all weather data timestamps, add price where available
        merged_df = price_data.join(weather_data, how="right")
        print(f"\nRight join: {merged_df.shape[0]} timestamps (all weather data)")

    else:

        class MergeTypeError(ValueError):
            def __init__(self) -> None:
                super().__init__(
                    "merge_type must be one of: 'inner', 'outer', 'left', 'right'"
                )

        raise MergeTypeError

    # Print information about missing values
    print("\nMissing values per column:")
    missing_counts = merged_df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} missing ({count / len(merged_df) * 100:.1f}%)")

    # Sort by index to ensure chronological order
    return merged_df.sort_index()


def get_alternating_heating_power(t) -> float:
    if (t // 3600) % 10 == 0:
        return np.array([2000.0])  # 2 kW
    return np.array([0.0])


def get_constant_heating_power(t: float) -> float:
    # Example: Constant heating power
    return np.array([2000.0])


# Example usage
def plot_hvac_results(
    get_ambient_temperature: Callable,
    get_solar_radiation: Callable,
    get_heating_power: Callable,
    time: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
) -> plt.Figure:
    """
    Plot the results of an HVAC simulation, including temperatures, heating power,
    and solar radiation over time.

    Parameters:
    -----------
    get_ambient_temperature : callable
        A function that takes a time value (in seconds) and returns the ambient
        (outdoor) temperature at that time.
    get_solar_radiation : callable
        A function that takes a time value (in seconds) and returns the solar
        radiation (in W/m²) at that time.
    get_heating_power : callable
        A function that takes a time value (in seconds) and returns the heating
        power (in W) at that time.
    time : numpy.ndarray
        A 1D array of time values (in seconds) corresponding to the simulation time steps.
    x : numpy.ndarray
        A 2D array where each row corresponds to the state variables at a given time step:
        - x[:, 0]: Indoor temperature (°C)
        - x[:, 1]: Radiator temperature (°C)
        - x[:, 2]: Envelope temperature (°C)
    u : numpy.ndarray
        A 1D array of heating power values (in W) at each time step.

    Returns:
    --------
    None
        This function does not return any value. It generates and displays plots
        of the simulation results.

    Notes:
    ------
    - The time values are converted from seconds to hours for plotting.
    - The function creates two subplots:
        1. Temperatures (indoor, radiator, envelope, and outdoor) over time.
        2. Heating power and solar radiation over time.
    - The plots include legends, grid lines, and appropriate labels for clarity.
    """
    results = {
        "Ti": convert_temperature(x[:, 0], "kelvin", "celsius"),  # Indoor temperature
        "Th": convert_temperature(x[:, 1], "kelvin", "celsius"),  # Radiator temperature
        "Te": convert_temperature(x[:, 2], "kelvin", "celsius"),  # Envelope temperature
        "Ta": convert_temperature(
            np.array([get_ambient_temperature(t) for t in time]), "kelvin", "celsius"
        ),  # Outdoor temperature
        "time": time,  # Time array
        "qh": [get_heating_power(t) for t in time],  # Heating power
        "Phi_s": [get_solar_radiation(t) for t in time],  # Solar radiation
    }

    dt = time[1] - time[0]  # Time step in seconds:w

    # Plot results
    plt.figure(figsize=(12, 8))
    # Convert time from seconds to hours for plotting
    time_hours = time / 3600

    # Plot temperatures (first subplot - unchanged)
    plt.subplot(3, 1, 1)
    plt.plot(time_hours, results["Ti"], "b-", label="Indoor Temperature")
    plt.plot(time_hours, results["Te"], "g-", label="Envelope Temperature")
    plt.plot(time_hours, results["Ta"], "k--", label="Outdoor Temperature")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time_hours, results["Th"], "r-", label="Radiator Temperature")
    plt.legend()
    plt.ylabel("Temperature (°C)")
    plt.grid()

    # Plot heating power and solar radiation (second subplot - with twin axes)
    ax1 = plt.subplot(3, 1, 3)
    # Primary y-axis (left) for heating power
    (line1,) = ax1.plot(
        time_hours,
        u,
        "r-",
        label="Heating Power",
    )
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Heating Power (W)", color="r")
    ax1.tick_params(axis="y", labelcolor="r")
    ax1.grid()

    # Secondary y-axis (right) for solar radiation
    ax2 = ax1.twinx()
    solar_data = [get_solar_radiation(t) for t in results["time"]]
    (line2,) = ax2.plot(
        time_hours,
        solar_data,
        "y-",
        label="Solar Radiation",
    )
    ax2.set_ylabel("Solar Radiation (W/m²)", color="y")
    ax2.tick_params(axis="y", labelcolor="y")

    # Create a combined legend for the second subplot
    ax1.legend([line1, line2], ["Heating Power", "Solar Radiation"], loc="upper left")

    plt.xlabel("Time (hours)")
    plt.tight_layout()

    # Add super title with dt
    plt.suptitle(f"HVAC Simulation Results (dt = {dt:.2f} seconds)", fontsize=16)

    # Save the figure
    plt.savefig(f"hvac_simulation_results_{int(dt)}.png")

    return plt.gcf()


if __name__ == "__main__":
    # Create the model

    # Simulation parameters

    # ambient_temperate_function = get_ambient_temperature
    # solar_radiation_function = get_solar_radiation
    ambient_temperate_function = lambda t: convert_temperature(10, "c", "k")
    solar_radiation_function = lambda t: 0.0  # No solar radiation for simplicity

    heating_power_function = get_alternating_heating_power

    for time_step in [900.0]:
        days = 7
        time_span = (0, days * 24 * 3600)
        time = np.arange(time_span[0], time_span[1], time_step)

        env = StochasticThreeStateRcEnv(
            step_size=time_step,
            ambient_temperature_function=ambient_temperate_function,
            solar_radiation_function=solar_radiation_function,
            enable_noise=True,
        )

        env.reset()

        x = [env.state]
        u = []

        # Run simulation
        for t in time:
            # Get current inputs
            action = heating_power_function(t)

            # Update state using the integrator
            obs = env.step(action)

            x.append(env.state)
            u.append(action)

        # Pop the last state
        x.pop()

        x = np.array(x)
        u = np.array(u)

        fig = plot_hvac_results(
            ambient_temperate_function,
            solar_radiation_function,
            heating_power_function,
            time,
            x,
            u,
        )

    plt.show()
