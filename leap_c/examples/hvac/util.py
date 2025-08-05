from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
from scipy.constants import convert_temperature


@dataclass
class ComfortBounds:
    """Temperature comfort bounds over the prediction horizon."""

    T_lower: np.ndarray  # Lower temperature bounds [K] for each time step
    T_upper: np.ndarray  # Upper temperature bounds [K] for each time step

    def __post_init__(self) -> None:
        assert len(self.T_lower) == len(self.T_upper), (
            "Lower and upper bounds must have same length"
        )
        assert np.all(self.T_lower <= self.T_upper), (
            "Lower bounds must be <= upper bounds"
        )


@dataclass
class DisturbanceProfile:
    """Disturbance profile over the prediction horizon."""

    T_outdoor: np.ndarray  # Outdoor temperature [K] for each time step
    solar_radiation: np.ndarray  # Solar radiation [W/m²] for each time step

    def __post_init__(self) -> None:
        assert len(self.T_outdoor) == len(self.solar_radiation), (
            "Outdoor temp and solar radiation must have same length"
        )

    @property
    def horizon_length(self) -> int:
        return len(self.T_outdoor)

    def get_disturbance_vector(self, k: int) -> np.ndarray:
        """Get disturbance vector at time step k."""
        return np.array([self.T_outdoor[k], self.solar_radiation[k]])


@dataclass
class EnergyPriceProfile:
    """Energy price profile over the prediction horizon."""

    price: np.ndarray  # Energy price for each time step

    def __post_init__(self) -> None:
        assert np.all(self.price >= 0), "Energy price must be non-negative"


def load_price_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load electricity price data from CSV file.

    Args:
        csv_path: Path to the price CSV file
    Returns:
        DataFrame with processed price data
    """
    # Load CSV with comma separator and first column as index
    price_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Ensure all price values are non-negative (handle any data quality issues)
    price_columns = price_data.columns
    for col in price_columns:
        price_data[col] = np.maximum(0, price_data[col])

    print(
        f"Loaded price data: {len(price_data)} records from {price_data.index[0]} to {price_data.index[-1]}"
    )
    print(f"Price regions: {', '.join(price_data.columns)}")
    print(
        f"Price range across all regions: {price_data.min().min():.5f} to {price_data.max().max():.5f}"
    )

    return price_data


def create_price_profile_from_spot_sprices(
    price_df: pd.DataFrame,
    start_time: str,
    horizon_hours: float = 24.0,
    dt_minutes: float = 15.0,
    region: str = "NO_1",
) -> EnergyPriceProfile:
    """
    Create a price profile from electricity price data using zero-order hold.

    Args:
        price_df: Price DataFrame from load_price_data()
        start_time: Start time as string (e.g., '2020-01-01 00:00:00')
        horizon_hours: Prediction horizon in hours
        dt_minutes: Time step in minutes
        region: Price region to extract (e.g., 'NO_1', 'NO_2', 'DK_1', etc.)

    Returns:
        numpy array with price values at 15-minute resolution
    """
    start_dt = pd.to_datetime(start_time)

    # Create time vector for the horizon
    n_steps = int(horizon_hours * 60 / dt_minutes)
    time_vector = [start_dt + timedelta(minutes=i * dt_minutes) for i in range(n_steps)]

    # Check if region exists in the data
    if region not in price_df.columns:
        available_regions = ", ".join(price_df.columns)
        raise ValueError(
            f"Region '{region}' not found. Available regions: {available_regions}"
        )

    # Extract price values using zero-order hold
    prices = []

    for t in time_vector:
        try:
            # Find the most recent price data point (zero-order hold)
            # Since prices are hourly, we want the price from the current or previous hour
            available_times = price_df.index[price_df.index <= t]

            if len(available_times) > 0:
                # Use the most recent available price (zero-order hold)
                latest_time = available_times[-1]
                price_value = price_df.loc[latest_time, region]
                prices.append(price_value)
            # If no previous data available, use the first available price
            elif len(price_df) > 0:
                first_time = price_df.index[0]
                price_value = price_df.loc[first_time, region]
                prices.append(price_value)
                print(
                    f"Warning: Using first available price for time {t} (before data start)"
                )
            else:
                raise ValueError("No price data available")

        except Exception:
            print(f"Warning: Could not get price data at {t}, using nearest value")
            # Use nearest available data as fallback
            if len(price_df) > 0:
                nearest_idx = price_df.index[np.argmin(np.abs(price_df.index - t))]
                price_value = price_df.loc[nearest_idx, region]
                prices.append(price_value)
            else:
                prices.append(0.0)  # Default fallback

    price_array = np.array(prices)

    print(f"Created price profile for region '{region}': {len(price_array)} samples")
    print(f"Time range: {time_vector[0]} to {time_vector[-1]}")
    print(f"Price range: {price_array.min():.5f} to {price_array.max():.5f}")

    return EnergyPriceProfile(price=price_array)


def load_weather_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load weather data from CSV file.

    Args:
        csv_path: Path to the weather CSV file

    Returns:
        DataFrame with processed weather data
    """
    # Load CSV with semicolon separator
    weather_data = pd.read_csv(csv_path, sep=";")

    # Parse timestamp
    weather_data["TimeStamp"] = pd.to_datetime(weather_data["TimeStamp"])

    # Convert temperature from Celsius to Kelvin
    weather_data["Tout_K"] = convert_temperature(weather_data["Tout"], "c", "k")

    # Ensure solar radiation is non-negative (handle numerical precision issues)
    weather_data["SolGlob"] = np.maximum(0, weather_data["SolGlob"])

    # Set timestamp as index for easier time-based operations
    weather_data.set_index("TimeStamp", inplace=True)

    print(
        f"Loaded weather data: {len(weather_data)} records from {weather_data.index[0]} to {weather_data.index[-1]}"
    )
    print(
        f"Temperature range: {weather_data['Tout'].min():.1f}°C to {weather_data['Tout'].max():.1f}°C"
    )
    print(
        f"Solar radiation range: {weather_data['SolGlob'].min():.1f} to {weather_data['SolGlob'].max():.1f} W/m²"
    )

    return weather_data


def create_disturbance_from_weather(
    weather_df: pd.DataFrame,
    start_time: str,
    horizon_hours: float = 24.0,
    dt_minutes: float = 15.0,
) -> DisturbanceProfile:
    """
    Create a disturbance profile from weather data.

    Args:
        weather_df: Weather DataFrame from load_weather_data()
        start_time: Start time as string (e.g., '2021-01-01 00:00:00')
        horizon_hours: Prediction horizon in hours
        dt_minutes: Time step in minutes

    Returns:
        DisturbanceProfile object
    """
    start_dt = pd.to_datetime(start_time)

    # Create time vector for the horizon
    n_steps = int(horizon_hours * 60 / dt_minutes)
    time_vector = [start_dt + timedelta(minutes=i * dt_minutes) for i in range(n_steps)]

    # Interpolate weather data to the desired time grid
    T_outdoor = []
    solar_radiation = []

    for t in time_vector:
        # Find closest weather data point or interpolate
        if t in weather_df.index:
            T_outdoor.append(weather_df.loc[t, "Tout_K"])
            solar_radiation.append(weather_df.loc[t, "SolGlob"])
        else:
            # Linear interpolation between nearest points
            try:
                # Get surrounding data points
                before_mask = weather_df.index <= t
                after_mask = weather_df.index >= t

                if before_mask.any() and after_mask.any():
                    t_before = weather_df.index[before_mask][-1]
                    t_after = weather_df.index[after_mask][0]

                    if t_before == t_after:
                        # Exact match
                        T_outdoor.append(weather_df.loc[t_before, "Tout_K"])
                        solar_radiation.append(weather_df.loc[t_before, "SolGlob"])
                    else:
                        # Interpolate
                        dt_total = (t_after - t_before).total_seconds()
                        dt_current = (t - t_before).total_seconds()
                        weight = dt_current / dt_total

                        T_interp = (
                            weather_df.loc[t_before, "Tout_K"] * (1 - weight)
                            + weather_df.loc[t_after, "Tout_K"] * weight
                        )
                        sol_interp = (
                            weather_df.loc[t_before, "SolGlob"] * (1 - weight)
                            + weather_df.loc[t_after, "SolGlob"] * weight
                        )

                        T_outdoor.append(T_interp)
                        solar_radiation.append(
                            max(0, sol_interp)
                        )  # Ensure non-negative
                else:
                    # Extrapolate from nearest point
                    if before_mask.any():
                        nearest_idx = weather_df.index[before_mask][-1]
                    else:
                        nearest_idx = weather_df.index[after_mask][0]

                    T_outdoor.append(weather_df.loc[nearest_idx, "Tout_K"])
                    solar_radiation.append(weather_df.loc[nearest_idx, "SolGlob"])

            except Exception:
                print(
                    f"Warning: Could not interpolate weather data at {t}, using nearest value"
                )
                # Use nearest available data
                nearest_idx = weather_df.index[np.argmin(np.abs(weather_df.index - t))]
                T_outdoor.append(weather_df.loc[nearest_idx, "Tout_K"])
                solar_radiation.append(weather_df.loc[nearest_idx, "SolGlob"])

    return DisturbanceProfile(
        T_outdoor=np.array(T_outdoor), solar_radiation=np.array(solar_radiation)
    )


def create_realistic_comfort_bounds(
    N: int,
    start_time: str,
    dt_minutes: float = 15.0,
    day_temp_range: tuple[float, float] = (20.0, 22.0),
    night_temp_range: tuple[float, float] = (18.0, 21.0),
    day_start_hour: int = 7,
    day_end_hour: int = 22,
) -> ComfortBounds:
    """
    Create realistic time-varying comfort bounds based on occupancy patterns.

    Args:
        n_steps: Number of time steps (N+1 for states)
        start_time: Start time as string
        dt_minutes: Time step in minutes
        day_temp_range: (lower, upper) temperature bounds during day in Celsius
        night_temp_range: (lower, upper) temperature bounds during night in Celsius
        day_start_hour: Hour when day period starts (7 = 7 AM)
        day_end_hour: Hour when day period ends (22 = 10 PM)

    Returns:
        ComfortBounds object
    """
    start_dt = pd.to_datetime(start_time)

    T_lower = []
    T_upper = []

    for i in range(N + 1):
        current_time = start_dt + timedelta(minutes=i * dt_minutes)
        hour = current_time.hour

        # Determine if it's day or night
        if day_start_hour <= hour <= day_end_hour:
            # Day time - stricter comfort bounds
            T_lower.append(convert_temperature(day_temp_range[0], "celsius", "kelvin"))
            T_upper.append(convert_temperature(day_temp_range[1], "celsius", "kelvin"))
        else:
            # Night time - more relaxed bounds
            T_lower.append(
                convert_temperature(night_temp_range[0], "celsius", "kelvin")
            )
            T_upper.append(
                convert_temperature(night_temp_range[1], "celsius", "kelvin")
            )

    return ComfortBounds(T_lower=np.array(T_lower), T_upper=np.array(T_upper))


def create_constant_disturbance(
    N: int, T_outdoor: float, solar_radiation: float
) -> DisturbanceProfile:
    """Create a constant disturbance profile."""
    return DisturbanceProfile(
        T_outdoor=np.full(N, T_outdoor), solar_radiation=np.full(N, solar_radiation)
    )


def create_constant_comfort_bounds(
    N: int, T_lower: float, T_upper: float
) -> ComfortBounds:
    """Create constant comfort bounds."""
    return ComfortBounds(
        T_lower=np.full(N + 1, T_lower), T_upper=np.full(N + 1, T_upper)
    )


def create_time_of_use_energy_costs(
    N: int,
    start_time: str,
    dt_minutes: float = 15.0,
    base_cost: float = 0.05,
    peak_cost: float = 0.20,
    peak_hours: tuple[int, int] = (17, 21),  # 5 PM to 9 PM
) -> EnergyPriceProfile:
    """
    Create time-of-use energy cost profile with peak pricing.

    Args:
        N: Number of time steps
        start_time: Start time as string
        dt_minutes: Time step in minutes
        base_cost: Base energy cost (off-peak)
        peak_cost: Peak energy cost
        peak_hours: (start_hour, end_hour) for peak pricing

    Returns:
        EnergyCostProfile object
    """
    start_dt = pd.to_datetime(start_time)

    costs = []
    for i in range(N):
        current_time = start_dt + timedelta(minutes=i * dt_minutes)
        hour = current_time.hour

        if peak_hours[0] <= hour <= peak_hours[1]:
            costs.append(peak_cost)
        else:
            costs.append(base_cost)

    return EnergyPriceProfile(price=np.array(costs))


def create_constant_energy_price(N: int, cost: float) -> EnergyPriceProfile:
    """Create constant energy costs."""
    return EnergyPriceProfile(price=np.full(N, cost))


def plot_ocp_results(
    solution: dict[str, Any],
    disturbance_profile: DisturbanceProfile,
    energy_prices: EnergyPriceProfile | None = None,
    comfort_bounds: ComfortBounds | None = None,
    dt: float = 15 * 60,
    figsize: tuple[float, float] = (12, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot the OCP solution results in a figure with three vertically stacked subplots.

    Args:
        solution: Solution dictionary from ThermalControlOCP.solve()
        disturbance_profile: Disturbance profile used in optimization
        comfort_bounds: Comfort bounds used in optimization
        dt: Time step in seconds
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    if not solution["success"]:
        print("Cannot plot: optimization was not successful")
        return None

    # Create time vectors
    N = len(solution["controls"])
    time_hours_states = np.arange(N + 1) * dt / 3600  # For states (N+1 points)
    time_hours_controls = np.arange(N) * dt / 3600  # For controls (N points)
    time_hours_disturbance = (
        np.arange(min(len(disturbance_profile.T_outdoor), N)) * dt / 3600
    )

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        "Thermal Building Control - OCP Solution", fontsize=16, fontweight="bold"
    )

    # Subplot 1: Thermal States
    ax0 = axes[0]

    # Convert temperatures to Celsius for plotting
    Ti_celsius = convert_temperature(
        solution["indoor_temperatures"], "kelvin", "celsius"
    )
    Th_celsius = convert_temperature(
        solution["radiator_temperatures"], "kelvin", "celsius"
    )
    Te_celsius = convert_temperature(
        solution["envelope_temperatures"], "kelvin", "celsius"
    )

    if comfort_bounds is not None:
        # Plot comfort bounds
        T_lower_celsius = convert_temperature(
            comfort_bounds.T_lower[: N + 1], "kelvin", "celsius"
        )
        T_upper_celsius = convert_temperature(
            comfort_bounds.T_upper[: N + 1], "kelvin", "celsius"
        )

        ax0.fill_between(
            time_hours_states,
            T_lower_celsius,
            T_upper_celsius,
            alpha=0.2,
            color="lightgreen",
            label="Comfort zone",
        )

        # Plot comfort bounds as dashed lines
        ax0.step(
            time_hours_states, T_lower_celsius, "g--", alpha=0.7, label="Lower bound"
        )
        ax0.step(
            time_hours_states, T_upper_celsius, "g--", alpha=0.7, label="Upper bound"
        )

    # Plot state trajectories
    ax0.step(
        time_hours_states, Ti_celsius, "b-", linewidth=2, label="Indoor temp. (Ti)"
    )
    ax0.step(
        time_hours_states,
        Te_celsius,
        "orange",
        linewidth=2,
        label="Envelope temp. (Te)",
    )

    ax0.set_ylabel("Temperature [°C]", fontsize=12)
    ax0.legend(loc="best")
    ax0.grid(visible=True, alpha=0.3)
    ax0.set_title("Indoor/Envelope Temperature", fontsize=14, fontweight="bold")

    # Subplot 1: Heater Temperature
    ax1 = axes[1]
    ax1.step(
        time_hours_states, Th_celsius, "b-", linewidth=2, label="Radiator temp. (Th)"
    )
    ax1.set_ylabel("Temperature [°C]", fontsize=12)
    ax1.grid(visible=True, alpha=0.3)
    ax1.set_title("Heater Temperature", fontsize=14, fontweight="bold")

    # Subplot 2: Disturbance Signals (twin axes)
    ax2 = axes[2]

    # Outdoor temperature (left y-axis)
    To_celsius = convert_temperature(
        disturbance_profile.T_outdoor[: len(time_hours_disturbance)],
        "kelvin",
        "celsius",
    )
    ax2.step(
        time_hours_disturbance,
        To_celsius,
        "b-",
        where="post",
        linewidth=2,
        label="Outdoor temp.",
    )
    ax2.set_ylabel("Outdoor Temperature [°C]", color="b", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="b")

    # Solar radiation (right y-axis)
    ax2_twin = ax2.twinx()
    solar_rad = disturbance_profile.solar_radiation[: len(time_hours_disturbance)]
    ax2_twin.step(
        time_hours_disturbance,
        solar_rad,
        color="orange",
        where="post",
        linewidth=2,
        label="Solar radiation",
    )
    ax2_twin.set_ylabel("Solar Radiation [W/m²]", color="orange", fontsize=12)
    ax2_twin.tick_params(axis="y", labelcolor="orange")

    ax2.grid(visible=True, alpha=0.3)
    ax2.set_title("Exogeneous Signals", fontsize=14, fontweight="bold")

    # Subplot 3: Control Input
    ax3 = axes[3]

    # Plot control as step function
    ax3.step(
        time_hours_controls,
        solution["controls"],
        "b-",
        where="post",
        linewidth=2,
        label="Heat input",
    )

    ax3.set_xlabel("Time [hours]", fontsize=12)
    ax3.set_ylabel("Heat Input [W]", color="b", fontsize=12)
    ax3.grid(visible=True, alpha=0.3)
    ax3.set_title("Control Input", fontsize=14, fontweight="bold")

    # Set y-axis lower limit to 0 for better visualization
    ax3.set_ylim(bottom=0)

    if energy_prices is not None:
        ax3_twin = ax3.twinx()
        # Add energy cost as a secondary y-axis
        ax3_twin.step(
            time_hours_controls,
            energy_prices.price,
            color="orange",
            where="post",
            linewidth=2,
            label="Energy cost (scaled)",
        )
        ax3_twin.set_ylabel("Energy Price [EUR/kWh]", color="orange", fontsize=12)
        ax3_twin.tick_params(axis="y", labelcolor="orange")
        ax3_twin.grid(visible=False)  # Disable grid for twin axis
        ax3_twin.set_ylim(bottom=0)  # Set lower limit to 0 for energy cost

    # Adjust layout
    plt.tight_layout()

    # Add summary statistics as text
    total_energy_kWh = solution["controls"].sum() * dt / 3600 / 1000  # Convert to kWh
    max_comfort_violation = max(
        solution["slack_lower"].max(), solution["slack_upper"].max()
    )

    stats_text = (
        f"Total Energy: {total_energy_kWh:.1f} kWh | "
        f"Total Cost: {solution['cost']:.2f} | "
        f"Max Comfort Violation: {max_comfort_violation:.2f} K"
    )

    fig.text(
        0.78,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.8},
    )

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_comfort_violations(
    solution: dict[str, Any],
    dt: float = 15 * 60,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Plot comfort violations (slack variables) over time.

    Args:
        solution: Solution dictionary from ThermalControlOCP.solve()
        dt: Time step in seconds
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    if not solution["success"]:
        print("Cannot plot: optimization was not successful")
        return None

    N = len(solution["controls"])
    time_hours = np.arange(N + 1) * dt / 3600

    fig, ax = plt.subplots(figsize=figsize)

    # Plot slack variables
    ax.plot(
        time_hours,
        solution["slack_lower"],
        "b-",
        linewidth=2,
        label="Lower bound violation (δ_l)",
    )
    ax.plot(
        time_hours,
        solution["slack_upper"],
        "r-",
        linewidth=2,
        label="Upper bound violation (δ_u)",
    )

    ax.set_xlabel("Time [hours]", fontsize=12)
    ax.set_ylabel("Temperature Violation [K]", fontsize=12)
    ax.set_title("Comfort Constraint Violations", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis lower limit to 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def transcribe_continuous_state_space(
    Ac: ca.SX | np.ndarray,
    Bc: ca.SX | np.ndarray,
    Ec: ca.SX | np.ndarray,
    params: dict[str, float],
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """
    Create continuous-time state-space matrices Ac, Bc, Ec as per equation (6).

    Args:
        Ac: State-space matrix (system dynamics)
        Bc: State-space matrix (control input)
        Ec: State-space matrix (disturbances)
        params: Dictionary with thermal parameters

    Returns:
        Ac, Bc, Ec: State-space matrices

    """
    # Extract parameters
    Ch = params["Ch"]  # Radiator thermal capacitance
    Ci = params["Ci"]  # Indoor air thermal capacitance
    Ce = params["Ce"]  # Envelope thermal capacitance
    Rhi = params["Rhi"]  # Radiator to indoor air resistance
    Rie = params["Rie"]  # Indoor air to envelope resistance
    Rea = params["Rea"]  # Envelope to outdoor resistance
    gAw = params["gAw"]  # Effective window area

    # Create Ac matrix (system dynamics)
    # Indoor air temperature equation coefficients [Ti, Th, Te]
    Ac[0, 0] = -(1 / (Ci * Rhi) + 1 / (Ci * Rie))  # Ti coefficient
    Ac[0, 1] = 1 / (Ci * Rhi)  # Th coefficient
    Ac[0, 2] = 1 / (Ci * Rie)  # Te coefficient

    # Radiator temperature equation coefficients [Ti, Th, Te]
    Ac[1, 0] = 1 / (Ch * Rhi)  # Ti coefficient
    Ac[1, 1] = -1 / (Ch * Rhi)  # Th coefficient
    Ac[1, 2] = 0  # Te coefficient

    # Envelope temperature equation coefficients [Ti, Th, Te]
    Ac[2, 0] = 1 / (Ce * Rie)  # Ti coefficient
    Ac[2, 1] = 0  # Th coefficient
    Ac[2, 2] = -(1 / (Ce * Rie) + 1 / (Ce * Rea))  # Te coefficient

    # Create Bc matrix (control input)
    Bc[0, 0] = 0  # No direct effect on indoor temperature
    Bc[1, 0] = 1 / Ch  # Effect on radiator temperature
    Bc[2, 0] = 0  # No direct effect on envelope temperature

    # Create Ec matrix (disturbances: outdoor temperature and solar radiation)
    Ec[0, 0] = 0  # No direct effect of outdoor temperature on indoor temp
    Ec[0, 1] = gAw / Ci  # Effect of solar radiation on indoor temperature

    Ec[1, 0] = 0  # No effect of outdoor temp or solar on radiator
    Ec[1, 1] = 0

    Ec[2, 0] = 1 / (Ce * Rea)  # Effect of outdoor temperature on envelope
    Ec[2, 1] = 0  # No direct effect of solar radiation on envelope

    return Ac, Bc, Ec


def transcribe_discrete_state_space(
    Ad: ca.SX | np.ndarray,
    Bd: ca.SX | np.ndarray,
    Ed: ca.SX | np.ndarray,
    dt: float,
    params: dict[str, float],
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """
    Create discrete-time state-space matrices Ad, Bd, Ed as per equation (7).

    Args:
        Ad: State-space matrix (system dynamics)
        Bd: State-space matrix (control input)
        Ed: State-space matrix (disturbances)
        dt: Sampling time
        params: Dictionary with thermal parameters

    Returns:
        Ad, Bd, Ed: Discrete-time state-space matrices

    """
    # Extract type of Ad
    if isinstance(Ad, np.ndarray):
        # Create continuous-time state-space matrices
        Ac, Bc, Ec = transcribe_continuous_state_space(
            Ac=np.zeros((3, 3)),
            Bc=np.zeros((3, 1)),
            Ec=np.zeros((3, 2)),
            params=params,
        )

        # Discretize the continuous-time state-space representation
        Ad = scipy.linalg.expm(Ac * dt)  # Discrete-time state matrix
        Bd = np.linalg.solve(Ac, (Ad - np.eye(3))) @ Bc
        Ed = np.linalg.solve(Ac, (Ad - np.eye(3))) @ Ec

    elif isinstance(Ad, ca.SX):
        # Create continuous-time state-space matrices
        Ac, Bc, Ec = transcribe_continuous_state_space(
            Ac=ca.SX.zeros(3, 3),
            Bc=ca.SX.zeros(3, 1),
            Ec=ca.SX.zeros(3, 2),
            params=params,
        )

        # Discretize the continuous-time state-space representation
        Ad = ca.expm(Ac * dt)  # Discrete-time state matrix
        Bd = ca.mtimes(
            ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Bc
        )  # Discrete-time input matrix
        Ed = ca.mtimes(
            ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Ec
        )  # Discrete-time disturbance matrix

    return Ad, Bd, Ed


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


def set_temperature_limits(
    quarter_hours: np.ndarray,
    night_start_hour: int = 22,
    night_end_hour: int = 8,
    lb_night: float = convert_temperature(12.0, "celsius", "kelvin"),
    lb_day: float = convert_temperature(19.0, "celsius", "kelvin"),
    ub_night: float = convert_temperature(25.0, "celsius", "kelvin"),
    ub_day: float = convert_temperature(22.0, "celsius", "kelvin"),
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Set temperature limits based on the time of day."""
    hours = np.floor(quarter_hours / 4)

    # Vectorized night detection
    night_idx = (hours >= night_start_hour) | (hours < night_end_hour)

    # Initialize and set values
    lb = np.where(night_idx, lb_night, lb_day)
    ub = np.where(night_idx, ub_night, ub_day)
    return lb, ub
