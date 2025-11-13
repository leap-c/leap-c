from pathlib import Path

import casadi as ca
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
from scipy.constants import convert_temperature


def load_price_data(csv_path: str | Path) -> pd.DataFrame:
    """Load electricity price data from CSV file.

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
        f"Loaded price data: {len(price_data)} records from "
        f"{price_data.index[0]} to {price_data.index[-1]}"
    )
    print(f"Price regions: {', '.join(price_data.columns)}")
    print(
        "Price range across all regions: "
        f"{price_data.min().min():.5f} to {price_data.max().max():.5f}"
    )

    return price_data


def load_weather_data(csv_path: str | Path) -> pd.DataFrame:
    """Load weather data from CSV file.

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
        f"Loaded weather data: {len(weather_data)} records from "
        f"{weather_data.index[0]} to {weather_data.index[-1]}"
    )
    print(
        f"Temperature range: {weather_data['Tout'].min():.1f}°C "
        f"to {weather_data['Tout'].max():.1f}°C"
    )
    print(
        f"Solar radiation range: {weather_data['SolGlob'].min():.1f} "
        f"to {weather_data['SolGlob'].max():.1f} W/m²"
    )

    return weather_data


def transcribe_continuous_state_space(
    Ac: ca.SX | np.ndarray,
    Bc: ca.SX | np.ndarray,
    Ec: ca.SX | np.ndarray,
    params: dict[str, float],
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Create continuous-time state-space matrices Ac, Bc, Ec as per equation (6).

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
    """Create discrete-time state-space matrices Ad, Bd, Ed as per equation (7).

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
        Bd = ca.mtimes(ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Bc)  # Discrete-time input matrix
        Ed = ca.mtimes(
            ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Ec
        )  # Discrete-time disturbance matrix

    return Ad, Bd, Ed


def merge_price_weather_data(
    price_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    merge_type: str = "inner",
) -> pd.DataFrame:
    """Merge price and weather dataframes on their timestamp indices.

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
    print(f"Price data time range: {price_data.index.min()} to {price_data.index.max()}")
    print(f"Weather data time range: {weather_data.index.min()} to {weather_data.index.max()}")
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
                super().__init__("merge_type must be one of: 'inner', 'outer', 'left', 'right'")

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
