"""Dataset management for HVAC environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.constants import convert_temperature


@dataclass(kw_only=True)
class DataConfig:
    """Configuration for price and weather data.

    Attributes:
        price_zone: Electricity price zone.
        price_data_path: Path to the price data CSV file.
        weather_data_path: Path to the weather data CSV file.
        start_time: Simulation start time. If None, samples randomly from data.
        valid_months: List of valid months (1-12) for random sampling.
            Default is heating season months (Jan-Apr, Sep-Dec).
            Set to None to allow all months.
        max_hours: Maximum simulation time in hours for episodes.
        test_ratio: Ratio of weeks to use for testing (0.0 to 1.0).
        split_seed: Random seed for reproducible train/test split.
    """

    price_zone: Literal["NO_1", "NO_2", "NO_3", "DK_1", "DK_2", "DE_LU"] = "NO_1"
    price_data_path: Path | None = None
    weather_data_path: Path | None = None
    start_time: pd.Timestamp | None = None  # if None, samples randomly from data

    # train / test split configuration
    max_hours: int = 3 * 24  # Maximum episode length in hours
    valid_months: list[int] | None = field(
        default_factory=lambda: [1, 2, 3, 4, 9, 10, 11, 12]
    )  # Heating season months: Jan-Apr, Sep-Dec
    test_ratio: float = 0.2  # 20% of weeks for testing
    split_seed: int = 42  # Seed for reproducible train/test split


class HvacDataset:
    """Manages HVAC simulation data (prices, weather, time features).

    Provides convenient access to price, temperature, solar, and temporal data
    for the HVAC environment simulation.

    Attributes:
        data: Combined DataFrame with price, weather, and time features.
        price_max: Maximum price value (for normalization).
        cfg: Data configuration.
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        cfg: DataConfig | None = None,
    ):
        """Initialize HvacDataset.

        Args:
            data: Combined DataFrame with price, weather, and time features.
                If None, loads from cfg paths.
            cfg: Data configuration. If None, uses default DataConfig.
                If data is None, cfg is used to load data from files.

        Note:
            If data is provided, it is used directly.
            Otherwise, data is loaded from cfg paths (or default paths if cfg is None).
        """
        self.cfg = cfg or DataConfig()

        if data is not None:
            # Direct injection of prepared data
            self.data = data
        else:
            # Load from configuration
            self.data = load_and_prepare_data(
                price_zone=self.cfg.price_zone,
                price_data_path=self.cfg.price_data_path,
                weather_data_path=self.cfg.weather_data_path,
            )

        self.min = {key: self.data[key].min() for key in self.data.columns}
        self.max = {key: self.data[key].max() for key in self.data.columns}
        
        # Generate reproducible train/test split based on week numbers
        self._train_weeks, self._test_weeks = self._generate_week_split()

    def __len__(self) -> int:
        """Total number of timesteps in dataset."""
        return len(self.data)

    @property
    def index(self) -> pd.DatetimeIndex:
        """Get the datetime index of the dataset."""
        return self.data.index

    def get_price(self, idx: int, horizon: int = 1) -> np.ndarray:
        """Get price data from index.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Price array of shape (horizon,).
        """
        return self.data["price"].iloc[idx : idx + horizon].to_numpy()

    def get_temperature(self, idx: int, horizon: int = 1) -> np.ndarray:
        """Get ambient temperature from index.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Temperature array of shape (horizon,).
        """
        return self.data["temperature"].iloc[idx : idx + horizon].to_numpy()

    def get_solar(self, idx: int, horizon: int = 1) -> np.ndarray:
        """Get solar radiation from index.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Solar radiation array of shape (horizon,).
        """
        return self.data["solar"].iloc[idx : idx + horizon].to_numpy()

    def get_time_features(self, idx: int) -> tuple[int, int]:
        """Get time features (quarter hour, day of year).

        Args:
            idx: Data index.

        Returns:
            Tuple of (quarter_hour, day_of_year).
        """
        quarter_hour = self.data["quarter_hour"].iloc[idx]
        day_of_year = self.data["day"].iloc[idx]
        return quarter_hour, day_of_year

    def get_time_forecast(self, idx: int, horizon: int) -> np.ndarray:
        """Get time forecast array.

        Args:
            idx: Starting index.
            horizon: Number of forecast steps.

        Returns:
            Array of datetime64[m] timestamps.
        """
        return self.data["time"].iloc[idx : idx + horizon + 1].to_numpy(dtype="datetime64[m]")

    def get_quarter_hours(self, idx: int, horizon: int) -> np.ndarray:
        """Get quarter hour values for a range.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Array of quarter hour values.
        """
        return self.data["quarter_hour"].iloc[idx : idx + horizon].to_numpy()

    def is_valid_index(self, idx: int, horizon: int = 1) -> bool:
        """Check if index allows fetching horizon steps.

        Args:
            idx: Starting index to check.
            horizon: Number of steps needed.

        Returns:
            True if valid, False otherwise.
        """
        return 0 <= idx < len(self.data) - horizon

    def _generate_week_split(self) -> tuple[set[int], set[int]]:
        """Generate reproducible train/test split by week numbers.

        Returns:
            Tuple of (train_weeks, test_weeks) as sets of ISO week numbers.
        """
        # Get all unique week numbers in the dataset
        all_weeks = sorted(set(self.index.isocalendar().week))
        
        # Reproducibly shuffle and split
        rng = np.random.default_rng(self.cfg.split_seed)
        shuffled_weeks = rng.permutation(all_weeks)
        
        n_test = max(1, int(len(all_weeks) * self.cfg.test_ratio))
        test_weeks = set(shuffled_weeks[:n_test])
        train_weeks = set(shuffled_weeks[n_test:])
        
        return train_weeks, test_weeks

    def sample_start_index(
        self,
        rng: np.random.Generator,
        horizon: int,
        max_steps: int,
        split: Literal["train", "test", "all"] = "all",
        max_attempts: int = 1000,
    ) -> int:
        """Sample a valid start index for episode initialization.

        Args:
            rng: NumPy random generator for reproducibility.
            horizon: Forecast horizon length.
            max_steps: Maximum number of simulation steps.
            split: Data split to sample from ('train', 'test', 'all').
            max_attempts: Maximum number of sampling attempts.

        Returns:
            Valid starting index.

        Raises:
            RuntimeError: If no valid start date found within max_attempts.
        """
        if self.cfg.start_time is not None:
            return self.index.get_loc(self.cfg.start_time)

        min_start_idx = 0
        max_start_idx = len(self.data) - horizon - max_steps + 1

        if max_start_idx <= min_start_idx:
            raise ValueError(
                f"Dataset too small: need at least {horizon + max_steps} steps, "
                f"but dataset has only {len(self.data)} steps."
            )

        # Determine which weeks to sample from based on split
        if split == "all":
            allowed_weeks = None
        elif split == "train":
            allowed_weeks = self._train_weeks
        elif split == "test":
            allowed_weeks = self._test_weeks
        else:
            raise ValueError(f"Invalid split value: {split}. Must be 'train', 'test', or 'all'.")

        # If no filtering at all, return random index directly
        if allowed_weeks is None and self.cfg.valid_months is None:
            return rng.integers(low=min_start_idx, high=max_start_idx + 1)

        # Sample with week and/or month filtering
        for _ in range(max_attempts):
            idx = rng.integers(low=min_start_idx, high=max_start_idx + 1)
            date = self.index[idx]
            
            # Check month constraint if specified
            if self.cfg.valid_months is not None and date.month not in self.cfg.valid_months:
                continue
            
            # Check week constraint if specified
            if allowed_weeks is not None:
                week_of_year = date.isocalendar()[1]  # ISO week number (1-53)
                if week_of_year not in allowed_weeks:
                    continue
            
            return idx

        raise RuntimeError(
            f"Could not find a valid start date in {max_attempts} attempts. "
            f"Split: {split}, Allowed weeks: {allowed_weeks}, Valid months: {self.cfg.valid_months}. "
            "Please check the data and configuration."
        )


# ============================================================================
# Data Loading and Preparation Utilities
# ============================================================================


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


def merge_price_weather_data(
    price_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    merge_type: str = "inner",
) -> pd.DataFrame:
    """Merge price and weather dataframes on their timestamp indices.

    Args:
        price_data: DataFrame with price data indexed by timestamp.
        weather_data: DataFrame with weather data indexed by timestamp.
        merge_type: Type of merge ('inner', 'outer', 'left', 'right').

    Returns:
        Merged dataframe.
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


def load_and_prepare_data(
    price_zone: str,
    price_data_path: Path | None = None,
    weather_data_path: Path | None = None,
) -> pd.DataFrame:
    """Load and prepare price and weather data.

    Args:
        price_zone: Electricity price zone (e.g., "NO_1", "NO_2", "DK_1", etc.).
        price_data_path: Path to the price data CSV file. If None, uses default path.
        weather_data_path: Path to the weather data CSV file. If None, uses default path.

    Returns:
        Tuple containing:
            - Prepared DataFrame with price, temperature, and solar data.
            - Maximum price value in the dataset.
    """
    # Determine data paths
    if price_data_path is None:
        price_data_path = Path(__file__).parent / "assets" / "spot_prices.csv"
    if weather_data_path is None:
        weather_data_path = Path(__file__).parent / "assets" / "weather.csv"

    # Load raw data
    price_data = load_price_data(csv_path=price_data_path).resample("15min").ffill()

    weather_data = (
        load_weather_data(csv_path=weather_data_path).resample("15min").interpolate(method="linear")
    )

    # Merge datasets
    data = merge_price_weather_data(
        price_data=price_data, weather_data=weather_data, merge_type="inner"
    )

    # Rename and select columns
    data.rename(
        columns={price_zone: "price", "Tout_K": "temperature", "SolGlob": "solar"},
        inplace=True,
    )
    data = data[["price", "temperature", "solar"]].copy()

    # Convert to float32 and add time features
    data["price"] = data["price"].astype(np.float32)
    data["temperature"] = data["temperature"].astype(np.float32)
    data["solar"] = data["solar"].astype(np.float32)
    data["time"] = data.index.to_numpy(dtype="datetime64[m]")
    data["quarter_hour"] = (data.index.hour * 4 + data.index.minute // 15) % (24 * 4)
    data["day"] = data["time"].dt.dayofyear % 366

    return data
