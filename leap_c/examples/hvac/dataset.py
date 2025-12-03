"""Dataset management for HVAC environment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from retry_requests import retry
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

    price_zone: Literal["NO1", "NO2", "NO3", "DK1", "DK2", "DE-AT-LU"] = "NO1"
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

        # Check for NaN values in the dataset
        assert not self.data.isnull().any().any(), (
            f"Dataset contains NaN values in columns: "
            f"{self.data.columns[self.data.isnull().any()].tolist()}"
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

    def get_temperature_forecast(self, idx: int, horizon: int = 1) -> np.ndarray:
        """Get ambient temperature forecastfrom index.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Temperature forecast array of shape (horizon,).
        """
        return self.data["temperature_forecast"].iloc[idx : idx + horizon].to_numpy()

    def get_solar(self, idx: int, horizon: int = 1) -> np.ndarray:
        """Get solar radiation from index.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Solar radiation array of shape (horizon,).
        """
        return self.data["solar"].iloc[idx : idx + horizon].to_numpy()

    def get_solar_forecast(self, idx: int, horizon: int = 1) -> np.ndarray:
        """Get solar radiation forecast from index.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Solar radiation array of shape (horizon,).
        """
        return self.data["solar_forecast"].iloc[idx : idx + horizon].to_numpy()

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


def load_price_data(
    csv_path: str | Path,
    price_zone: str = "NO1",
    start_date: str = "2017-01-01",
    end_date: str = "2017-11-30",
) -> pd.DataFrame:
    """Load electricity price data from CSV file.

    Args:
        csv_path: Path to the price CSV file
        price_zone: Electricity price zone.
        start_date: Start date for filtering price data.
        end_date: End date for filtering price data.

    Returns:
        DataFrame with processed price data

    """
    # Check if file exists
    csv_path = Path(csv_path)
    try:
        price_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        price_data.index = pd.to_datetime(price_data.index, utc=True)
        # TODO: Need to take care that the price zone exists in the data.
        # If not, download separately and append it to the existing dataset.
    except FileNotFoundError:
        print(f"WARNING: Price data file not found: {csv_path}")
        price_data = get_energy_charts_data(
            price_zone=price_zone,
            start_date=start_date,
            end_date=end_date,
        )

        # Set datetime as index and sort
        price_data.set_index("Timestamp", inplace=True)

        # Resample price data to 15-minute intervals with zero-order hold
        price_data = price_data.resample("15min").ffill()

        # Save to CSV for future use
        price_data.to_csv(csv_path, index=True)

    return price_data


def get_energy_charts_data(
    price_zone: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download and preprocess energy chart data for a given price zone.

    The data file is available for, e.g., 2020, at:
    https://energy-charts.info/charts/price_spot_market/chart.htm?l=en&c=NO&year=2020&interval=year&minuteInterval=60min
    """
    # API endpoint and parameters

    # Make the request
    response = requests.get(
        url="https://api.energy-charts.info/price",
        params={
            "bzn": price_zone.replace("_", ""),
            "start": f"{start_date}T00:00",
            "end": f"{end_date}T00:00",
        },
    )

    # Check if request was successful
    if response.status_code == 200:
        print("Data downloaded successfully!")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    df = pd.DataFrame(response.json())[["unix_seconds", "price", "unit"]]

    df["Timestamp"] = pd.to_datetime(df["unix_seconds"], unit="s", utc=True)

    df = df.drop(columns=["unix_seconds", "unit"])

    # Assume unit is EUR/MWh, convert to EUR/kWh
    df["price"] = df["price"] / 1000.0

    return df


def get_open_meteo_data(
    latitude: float = 59.91387,
    longitude: float = 10.7522,
    start_date: str = "2017-01-01",
    end_date: str = "2017-11-26",
) -> pd.DataFrame:
    """Download and preprocess Oslo weather data from Open-Meteo.

    The data file is available at:
    https://open-meteo.com/en/docs/historical-forecast-api
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    responses = openmeteo.weather_api(
        url="https://historical-forecast-api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "minutely_15": [
                "temperature_2m",
                "apparent_temperature",
                "shortwave_radiation",
                "direct_normal_irradiance",
            ],
        },
    )

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process minutely_15 data. The order of variables needs to be the same as requested.
    minutely_15 = response.Minutely15()

    dataframe = pd.DataFrame(
        data={
            "date": pd.date_range(
                start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
                end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=minutely_15.Interval()),
                inclusive="left",
            ),
            "temperature_2m": minutely_15.Variables(0).ValuesAsNumpy(),
            "apparent_temperature": minutely_15.Variables(1).ValuesAsNumpy(),
            "shortwave_radiation": minutely_15.Variables(2).ValuesAsNumpy(),
            "direct_normal_irradiance": minutely_15.Variables(3).ValuesAsNumpy(),
        }
    )

    # Parse the date column and set it as index
    dataframe["Timestamp"] = pd.to_datetime(dataframe["date"])
    dataframe.set_index("Timestamp", inplace=True)

    return dataframe


def load_weather_data(
    csv_path: Path,
    latitude: float = 59.91387,
    longitude: float = 10.7522,
    start_date: str = "2017-01-01",
    end_date: str = "2025-11-30",
) -> pd.DataFrame:
    """Load and preprocess weather data from Open-Meteo.

    The data file is available at:
    https://open-meteo.com/en/docs/historical-forecast-api
    """
    # Load the dataset

    try:
        weather_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        weather_data.index = pd.to_datetime(weather_data.index, utc=True)
        # TODO: Need to take care that the price zone exists in the data.
        # If not, download separately and append it to the existing dataset.
    except FileNotFoundError:
        print(f"WARNING: Weather data file not found: {csv_path}")
        weather_data = get_open_meteo_data(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
        )
        weather_data.to_csv(csv_path, index=True)

    return weather_data


def load_and_prepare_data(
    price_zone: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    price_data_path: Path | None = None,
    weather_data_path: Path | None = None,
) -> pd.DataFrame:
    """Load and prepare price and weather data.

    Args:
        price_zone: Electricity price zone (e.g., "NO_1", "NO_2", "DK_1", etc.).
        start_date: Start date for filtering data (e.g., "2017-01-01").
        end_date: End date for filtering data (e.g., "2017-11-30").
        price_data_path: Path to the price data CSV file. If None, uses default path.
        weather_data_path: Path to the weather data CSV file. If None, uses default path.
        time_zone: Timezone for the data (e.g., "Europe/Berlin"). If None, uses "Europe/Berlin".

    Returns:
        Tuple containing:
            - Prepared DataFrame with price, temperature, and solar data.
            - Maximum price value in the dataset.
    """
    # Determine data paths
    if price_data_path is None:
        price_data_path = Path(__file__).parent / "assets" / "prices.csv"
    if weather_data_path is None:
        weather_data_path = Path(__file__).parent / "assets" / "weather.csv"
    if price_zone is None:
        price_zone = "NO1"
    if start_date is None:
        start_date = "2017-01-01"
    if end_date is None:
        end_date = "2025-02-15"  # After this, the price data is missing in large parts

    price = load_price_data(
        csv_path=price_data_path,
        price_zone=price_zone,
        start_date=start_date,
        end_date=end_date,
    )

    weather = load_weather_data(
        csv_path=weather_data_path,
        start_date=start_date,
        end_date=end_date,
    )
    # Load raw data
    data: pd.DataFrame = pd.merge(
        left=price,
        right=weather,
        left_index=True,
        right_index=True,
    )

    # Convert to float32 and add time features
    data["time"] = data.index.to_numpy(dtype="datetime64[m]")
    data["quarter_hour"] = (data.index.hour * 4 + data.index.minute // 15) % (24 * 4)
    data["day"] = data["time"].dt.dayofyear % 366  # 366 to account for leap years

    data["price"] = data["price"].astype(np.float32)
    data["temperature"] = data["temperature_2m"].astype(np.float32)
    data["temperature_forecast"] = data["apparent_temperature"].astype(np.float32)
    data["solar"] = data["shortwave_radiation"].astype(np.float32)
    data["solar_forecast"] = data["direct_normal_irradiance"].astype(np.float32)

    for key in ["temperature", "temperature_forecast"]:
        data[key] = convert_temperature(data[key], "C", "K")

    # Drop date column
    data.drop(
        columns=[
            "date",
            "temperature_2m",
            "apparent_temperature",
            "shortwave_radiation",
            "direct_normal_irradiance",
        ],
        inplace=True,
    )

    # Check for NaN values and apply zero-order hold (forward fill)
    nan_counts = data.isnull().sum()
    if nan_counts.any():
        print("NaN values detected before forward fill:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count} NaN values")

        # Apply zero-order hold (forward fill) to replace NaNs
        data = data.ffill()

        # Check if any NaNs remain (e.g., at the beginning of the dataset)
        remaining_nans = data.isnull().sum()
        if remaining_nans.any():
            print("NaN values remaining after forward fill (filling with backward fill):")
            for col, count in remaining_nans[remaining_nans > 0].items():
                print(f"  {col}: {count} NaN values")
            # Use backward fill for any remaining NaNs at the start
            data = data.bfill()

    print("Prepared combined dataset:")
    print(f"  Price range: {data['price'].min():.3f} to {data['price'].max():.3f} EUR/kWh")
    print(
        f"  Temperature range: {data['temperature'].min():.1f}K to {data['temperature'].max():.1f}K"
    )
    print(f"  Solar radiation range: {data['solar'].min():.1f} to {data['solar'].max():.1f} W/m²")
    print(f"  Total records: {len(data)} from {data.index[0]} to {data.index[-1]}")

    return data


if __name__ == "__main__":
    # Test loading and preparing data
    dataset = HvacDataset()
    print(f"Dataset length: {len(dataset)}")

    # Plot all price data
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(dataset.data.index, dataset.data["price"], label="Price (EUR/kWh)")
    plt.xlabel("Time")
    plt.ylabel("Price (EUR/kWh)")
    plt.title("Electricity Price Over Time")
    plt.legend()
    plt.show()
