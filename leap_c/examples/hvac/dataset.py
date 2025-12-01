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
            f"Split: {split}, Allowed weeks: {allowed_weeks}, "
            f"Valid months: {self.cfg.valid_months}. "
            "Please check the data and configuration."
        )


# ============================================================================
# Data Loading and Preparation Utilities
# ============================================================================


def load_price_data(csv_path: str | Path, price_zone: str = "NO_1") -> pd.DataFrame:
    """Load electricity price data from CSV file.

    Args:
        csv_path: Path to the price CSV file
        price_zone: Electricity price zone.

    Returns:
        DataFrame with processed price data
    """
    price_data = []

    for file in csv_path.iterdir():
        if (
            file.is_file()
            and file.name.startswith(
                "energy-charts_Electricity_production_and_spot_prices_in_Norway"
            )
            and file.suffix == ".csv"
        ):
            # price_data_path.append(file)
            price_data.append(
                load_and_preprocess_energy_chart(
                    file_path=file,
                    price_zone=price_zone,
                )
            )
    price_data = pd.concat(price_data).sort_index().resample("15min").ffill()

    return price_data


def load_and_preprocess_energy_chart(file_path: str, price_zone: str = "NO_1") -> pd.DataFrame:
    """Load and preprocess energy chart data for a given price zone.

    The data file is available for, e.g., 2020, at:
    https://energy-charts.info/charts/price_spot_market/chart.htm?l=en&c=NO&year=2020&interval=year&minuteInterval=60min
    """
    # Load the dataset, skipping the second row which contains units
    df = pd.read_csv(file_path, skiprows=[1])

    # Parse the date column and set it as index
    df["Date (GMT+1)"] = pd.to_datetime(df["Date (GMT+1)"], utc=True)

    # Rename to Timestamp
    df.rename(columns={"Date (GMT+1)": "Timestamp"}, inplace=True)
    df.rename(
        columns={
            f"Day Ahead Auction ({price_zone.replace('_', '')})": price_zone,
        },
        inplace=True,
    )

    df.set_index("Timestamp", inplace=True)

    # Select relevant column and convert to EUR/kWh
    df = df[[price_zone]] * 1e-3

    return df


def load_and_preprocess_open_meteo(csv_path: str) -> pd.DataFrame:
    """Load and preprocess weather data from Open-Meteo.

    The data file is available at:
    https://open-meteo.com/en/docs/historical-forecast-api
    """
    # Load the dataset
    dataframe = pd.read_csv(csv_path)

    # Parse the date column and set it as index
    dataframe["Timestamp"] = pd.to_datetime(dataframe["date"])
    dataframe.set_index("Timestamp", inplace=True)

    # Rename columns for clarity
    dataframe.rename(
        columns={
            "temperature_2m": "Tout_C",
            "shortwave_radiation": "SolGlob_W_m2",
        },
        inplace=True,
    )

    # Select relevant columns
    dataframe = dataframe[["Tout_C", "SolGlob_W_m2"]]

    dataframe["Tout_K"] = convert_temperature(dataframe["Tout_C"], "C", "K")

    print(
        f"Loaded weather data: {len(dataframe)} records from "
        f"{dataframe.index[0]} to {dataframe.index[-1]}"
    )
    print(
        f"Temperature range: {dataframe['Tout_C'].min():.1f}°C to {dataframe['Tout_C'].max():.1f}°C"
    )
    print(
        f"Solar radiation range: {dataframe['SolGlob_W_m2'].min():.1f} "
        f"to {dataframe['SolGlob_W_m2'].max():.1f} W/m²"
    )

    return dataframe


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
        price_data_path = Path(__file__).parent / "assets" / "energy_charts"
    if weather_data_path is None:
        weather_data_path = (
            Path(__file__).parent
            / "assets"
            / "open_meteo"
            / "oslo_weather_minutely_15_2017_2025.csv"
        )

    # Load raw data
    data = {
        "price": load_price_data(csv_path=price_data_path, price_zone=price_zone),
        "weather": load_and_preprocess_open_meteo(csv_path=weather_data_path),
    }

    data = pd.merge(data["price"], data["weather"], left_index=True, right_index=True)

    # Rename and select columns
    data.rename(
        columns={
            price_zone: "price",
            "Tout_K": "temperature",
            "SolGlob_W_m2": "solar",
        },
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

    print("Prepared combined dataset:")
    print(f"  Price range: {data['price'].min():.3f} to {data['price'].max():.3f} EUR/kWh")
    print(
        f"  Temperature range: {data['temperature'].min():.1f}K to {data['temperature'].max():.1f}K"
    )
    print(f"  Solar radiation range: {data['solar'].min():.1f} to {data['solar'].max():.1f} W/m²")
    print(f"  Total records: {len(data)} from {data.index[0]} to {data.index[-1]}")

    return data
