"""Dataset management for HVAC environment."""

from dataclasses import dataclass
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
        start_time: Simulation start time. If `None`, samples randomly from data.
        mode: Dataset mode - "random" for random sampling with fixed episode length, "continual"
            for sequential episodes scanning the full dataset.
        max_hours: Maximum simulation time in hours for episodes.
        total_test_episodes: Exact number of fixed test episodes, stratified evenly across all
            (year, month) combinations in the dataset.
            These episodes are always the same for reproducible evaluation.
        split_seed: Random seed for reproducible train/test split.
    """

    price_zone: Literal["NO1", "NO2", "NO3", "DK1", "DK2", "DE-AT-LU", "DE-LU"] = "DE-LU"
    price_data_path: Path | None = None
    weather_data_path: Path | None = None
    start_time: pd.Timestamp | None = None

    mode: Literal["random", "continual"] = "random"

    max_hours: int = int(4.5 * 24)  # Maximum episode length in hours
    total_test_episodes: int = 0  # Exact number of test episodes (stratified across months/years)
    split_seed: int = 42  # Seed for reproducible train/test split


class HvacDataset:
    """Manages HVAC simulation data (prices, weather, time features).

    Provides convenient access to price, temperature, solar, and temporal data for the HVAC
    environment simulation.

    Note:
        Currently, only supports fixed 15-minute interval data.

    Attributes:
        data: Combined DataFrame with price, weather, and time features.
        cfg: Data configuration.
    """

    def __init__(self, data: pd.DataFrame | None = None, cfg: DataConfig | None = None) -> None:
        """Initialize HvacDataset.

        Args:
            data: Combined DataFrame with price, weather, and time features.
                If `None`, loads from `cfg` paths.
            cfg: Data configuration. If `None`, uses the default `DataConfig` instance.
                If data is `None`, `cfg` is used to load data from files.

        Note:
            If `data` is provided, it is used directly.
            Otherwise, `data` is loaded from `cfg` paths (or default paths if `cfg` is `None`).
        """
        self.cfg = cfg or DataConfig()

        if data is not None:
            self.data = data
        else:
            self.data = load_and_prepare_data(
                price_zone=self.cfg.price_zone,
                price_data_path=self.cfg.price_data_path,
                weather_data_path=self.cfg.weather_data_path,
            )

        assert not self.data.isnull().any().any(), (
            f"Dataset contains NaN values in columns: "
            f"{self.data.columns[self.data.isnull().any()].tolist()}"
        )

        self.min = {key: self.data[key].min() for key in self.data.columns}
        self.max = {key: self.data[key].max() for key in self.data.columns}

        # Pre-built numpy arrays for fast indexed access
        self._price = self.data["price"].to_numpy()
        self._temperature = self.data["temperature"].to_numpy()
        self._temperature_forecast = self.data["temperature_forecast"].to_numpy()
        self._solar = self.data["solar"].to_numpy()
        self._solar_forecast = self.data["solar_forecast"].to_numpy()
        self._quarter_hour = self.data["quarter_hour"].to_numpy()
        self._day = self.data["day"].to_numpy()
        self._time = self.data["time"].to_numpy(dtype="datetime64[m]")

        # Continual mode: track current position in dataset
        self._continual_idx: int = 0

        # Random mode: fixed stratified test episodes
        self._test_indices: list[int] = []
        self._test_episode_counter: int = 0

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
        return self._price[idx : idx + horizon]

    def get_temperature(self, idx: int, horizon: int = 1) -> np.ndarray:
        """Get ambient temperature from index.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Temperature array of shape (horizon,).
        """
        return self._temperature[idx : idx + horizon]

    def get_temperature_forecast(self, idx: int, horizon: int = 1) -> np.ndarray:
        """Get ambient temperature forecastfrom index.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Temperature forecast array of shape (horizon,).
        """
        return self._temperature_forecast[idx : idx + horizon]

    def get_solar(self, idx: int, horizon: int = 1) -> np.ndarray:
        """Get solar radiation from index.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Solar radiation array of shape (horizon,).
        """
        return self._solar[idx : idx + horizon]

    def get_solar_forecast(self, idx: int, horizon: int = 1) -> np.ndarray:
        """Get solar radiation forecast from index.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Solar radiation array of shape (horizon,).
        """
        return self._solar_forecast[idx : idx + horizon]

    def get_time_features(self, idx: int) -> tuple[int, int]:
        """Get time features (quarter hour, day of year).

        Args:
            idx: Data index.

        Returns:
            Tuple of (quarter_hour, day_of_year).
        """
        return self._quarter_hour[idx], self._day[idx]

    def get_time_forecast(self, idx: int, horizon: int) -> np.ndarray:
        """Get time forecast array.

        Args:
            idx: Starting index.
            horizon: Number of forecast steps.

        Returns:
            Array of datetime64[m] timestamps.
        """
        return self._time[idx : idx + horizon + 1]

    def get_quarter_hours(self, idx: int, horizon: int) -> np.ndarray:
        """Get quarter hour values for a range.

        Args:
            idx: Starting index.
            horizon: Number of steps to retrieve.

        Returns:
            Array of quarter hour values.
        """
        return self._quarter_hour[idx : idx + horizon]

    def is_valid_index(self, idx: int, horizon: int = 1) -> bool:
        """Check if index allows fetching horizon steps.

        Args:
            idx: Starting index to check.
            horizon: Number of steps needed.

        Returns:
            `True` if valid, `False` otherwise.
        """
        return 0 <= idx < len(self.data) - horizon

    def _generate_stratified_test_episodes(self, horizon: int, max_steps: int) -> list[int]:
        """Generate exactly `total_test_episodes` stratified across months and years.

        Distributes episodes evenly across all (year, month) combinations in the dataset.

        Args:
            horizon: Forecast horizon length.
            max_steps: Maximum number of simulation steps per episode.

        Returns:
            List of exactly `total_test_episodes` start indices.
        """
        rng = np.random.default_rng(self.cfg.split_seed)

        max_idx = len(self.data) - horizon - max_steps

        if max_idx <= 0:
            raise ValueError(
                f"Dataset too small: need at least {horizon + max_steps} steps, "
                f"but dataset has only {len(self.data)} steps."
            )

        year_month_indices: dict[tuple[int, int], list[int]] = {}
        for idx in range(max_idx + 1):
            date = self.index[idx]
            key = (date.year, date.month)
            if key not in year_month_indices:
                year_month_indices[key] = []
            year_month_indices[key].append(idx)

        all_keys = sorted(year_month_indices.keys())
        n_buckets = len(all_keys)

        if n_buckets == 0:
            raise ValueError("No (year, month) combinations found in dataset.")

        base_per_bucket = self.cfg.total_test_episodes // n_buckets
        remainder = self.cfg.total_test_episodes % n_buckets

        if remainder > 0:
            step = n_buckets / remainder
            extra_indices = {int(i * step) for i in range(remainder)}
        else:
            extra_indices = set()

        test_indices = []
        for i, key in enumerate(all_keys):
            available_indices = year_month_indices[key]
            n_samples = base_per_bucket + (1 if i in extra_indices else 0)
            n_samples = min(n_samples, len(available_indices))
            if n_samples > 0:
                sampled = rng.choice(available_indices, size=n_samples, replace=False)
                test_indices.extend(sampled.tolist())

        rng2 = np.random.default_rng(self.cfg.split_seed)
        rng2.shuffle(test_indices)
        return test_indices

    def sample_start_index(
        self,
        rng: np.random.Generator,
        horizon: int,
        split: Literal["train", "test", "all"] = "train",
        max_attempts: int = 1000,
    ) -> tuple[int, int]:
        """Sample a valid start index for episode initialization.

        Behavior depends on `cfg.mode`:
        - "random": Randomly samples a start index with fixed episode length.
        - "continual": Returns sequential episodes scanning the full dataset.

        Args:
            rng: NumPy random generator for reproducibility.
            horizon: Forecast horizon length.
            split: Data split ("train", "test", or "all"). Ignored in continual mode.
            max_attempts: Maximum number of sampling attempts (random mode only).

        Returns:
            Tuple of (start_index, max_steps) for the episode.
        """
        if self.cfg.start_time is not None:
            idx = self.index.get_loc(self.cfg.start_time)
            max_steps = int(self.cfg.max_hours * 4)
            return idx, max_steps

        if self.cfg.mode == "continual":
            return self._sample_continual(horizon)
        else:
            return self._sample_random(rng, horizon, split, max_attempts)

    def _sample_continual(self, horizon: int) -> tuple[int, int]:
        """Sample next episode for continual learning mode.

        Scans the dataset sequentially. Returns (start_idx, 0) when exhausted.

        Args:
            horizon: Forecast horizon length.

        Returns:
            Tuple of (start_index, max_steps). Returns 0 max_steps when exhausted.
        """
        max_steps = int(self.cfg.max_hours * 4)
        start_idx = self._continual_idx

        if start_idx + horizon + max_steps > len(self.data):
            return start_idx, 0

        self._continual_idx = start_idx + max_steps
        return start_idx, max_steps

    def _sample_random(
        self,
        rng: np.random.Generator,
        horizon: int,
        split: Literal["train", "test", "all"],
        max_attempts: int,
    ) -> tuple[int, int]:
        """Sample random episode with fixed length and optional train/test split.

        Args:
            rng: NumPy random generator.
            horizon: Forecast horizon length.
            split: Data split ("train", "test", or "all").
            max_attempts: Maximum sampling attempts (used only for train split).

        Returns:
            Tuple of (start_index, max_steps).
        """
        max_steps = int(self.cfg.max_hours * 4)
        min_start_idx = 0
        max_start_idx = len(self.data) - horizon - max_steps + 1

        if max_start_idx <= min_start_idx:
            raise ValueError(
                f"Dataset too small: need at least {horizon + max_steps} steps, "
                f"but dataset has only {len(self.data)} steps."
            )

        if split == "all":
            idx = int(rng.integers(low=min_start_idx, high=max_start_idx + 1))
            return idx, max_steps

        if not self._test_indices:
            self._test_indices = self._generate_stratified_test_episodes(horizon, max_steps)
            self._test_episode_counter = 0

        episode_length = horizon + max_steps

        if split == "test":
            if not self._test_indices:
                return self._sample_random(rng, horizon, "all", max_attempts)
            idx = self._test_indices[self._test_episode_counter % len(self._test_indices)]
            self._test_episode_counter += 1
            return idx, max_steps

        # Train: exclude zones around test episodes
        test_exclusion_set: set[int] = set()
        for test_idx in self._test_indices:
            start_exclude = max(0, test_idx - episode_length + 1)
            end_exclude = min(max_start_idx, test_idx + episode_length - 1)
            test_exclusion_set.update(range(start_exclude, end_exclude + 1))

        for _ in range(max_attempts):
            idx = int(rng.integers(low=min_start_idx, high=max_start_idx + 1))
            if idx not in test_exclusion_set:
                return idx, max_steps

        raise RuntimeError(
            f"Could not find a valid start date in {max_attempts} attempts. "
            f"Split: {split}. Please check the data and configuration."
        )

    def reset_continual_index(self) -> None:
        """Reset the continual mode index to start from the beginning."""
        self._continual_idx = 0

    def is_exhausted(self, horizon: int) -> bool:
        """Check if the dataset has been fully traversed in continual mode.

        Args:
            horizon: Forecast horizon length.

        Returns:
            True if no more valid data remains, False otherwise.
        """
        if self.cfg.mode != "continual":
            return False
        max_steps = int(self.cfg.max_hours * 4)
        return self._continual_idx + horizon + max_steps > len(self.data)

    def reset_test_counter(self) -> None:
        """Reset the test episode counter to start from the first episode."""
        self._test_episode_counter = 0

    def get_num_test_episodes(self, horizon: int) -> int:
        """Get the total number of fixed test episodes.

        Args:
            horizon: Forecast horizon length.

        Returns:
            Number of test episodes.
        """
        max_steps = int(self.cfg.max_hours * 4)
        if not self._test_indices:
            self._test_indices = self._generate_stratified_test_episodes(horizon, max_steps)
        return len(self._test_indices)


# ============================================================================
# Data Loading and Preparation Utilities
# ============================================================================


def load_price_data(
    csv_path: str | Path,
    price_zone: str = "NO1",
    start_date: str = "2017-01-01",
    end_date: str = "2017-11-30",
    min_price: float = 0.0001,
) -> pd.DataFrame:
    """Load electricity price data from CSV file.

    Args:
        csv_path: Path to the price CSV file
        price_zone: Electricity price zone.
        start_date: Start date for filtering price data.
        end_date: End date for filtering price data.
        min_price: The cut-off value to prevent negative prices.

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

    # ensure price is non-negative and has a minimum value
    price_data["price"] = price_data["price"].clip(lower=min_price)

    return price_data


def get_energy_charts_data(price_zone: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download and preprocess energy chart data for a given price zone.

    The data file is available for, e.g., 2020, at:
    https://energy-charts.info/charts/price_spot_market/chart.htm?l=en&c=NO&year=2020&interval=year&minuteInterval=60min
    """
    # API endpoint and parameters

    # Make the request
    from requests import get

    response = get(
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
    from openmeteo_requests import Client
    from requests_cache import CachedSession
    from retry_requests import retry

    cache_session = CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = Client(session=retry_session)

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
    latitude: float = 47.995,  #  latitude: float = 59.91387,
    longitude: float = 7.850,  #  longitude: float = 10.7522,
    start_date: str = "2017-01-01",
    end_date: str = "2025-11-30",
) -> pd.DataFrame:
    """Load and preprocess weather data from Open-Meteo.

    The data file is available at:
    https://open-meteo.com/en/docs/historical-forecast-api
    """
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
        price_data_path: Path to the price data CSV file. If `None`, uses default path.
        weather_data_path: Path to the weather data CSV file. If `None`, uses default path.

    Returns:
        Prepared DataFrame with price, temperature, and solar data.
    """
    if price_data_path is None:
        price_data_path = Path(__file__).parent / "assets" / "prices.csv"
    if weather_data_path is None:
        weather_data_path = Path(__file__).parent / "assets" / "weather.csv"
    if price_zone is None:
        price_zone = "NO1"
    if start_date is None:
        start_date = "2021-03-23"
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
    data: pd.DataFrame = pd.merge(
        left=price,
        right=weather,
        left_index=True,
        right_index=True,
    )

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

        data = data.ffill()

        remaining_nans = data.isnull().sum()
        if remaining_nans.any():
            print("NaN values remaining after forward fill (filling with backward fill):")
            for col, count in remaining_nans[remaining_nans > 0].items():
                print(f"  {col}: {count} NaN values")
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
    import matplotlib.pyplot as plt

    dataset = HvacDataset(cfg=DataConfig(total_test_episodes=100))
    print(f"Dataset length: {len(dataset)}")

    horizon = 96  # 24 hours at 15-min intervals
    max_steps = int(1.5 * 24 * 4)  # 1.5 days

    test_indices = dataset._generate_stratified_test_episodes(horizon, max_steps)
    print(f"Total test episodes: {len(test_indices)}")

    test_dates = [dataset.index[idx] for idx in test_indices]
    test_year_months = [(d.year, d.month) for d in test_dates]

    from collections import Counter

    counts = Counter(test_year_months)

    years = sorted(set(ym[0] for ym in counts.keys()))
    months = sorted(set(ym[1] for ym in counts.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    year_counts = Counter(d.year for d in test_dates)
    ax1.bar(year_counts.keys(), year_counts.values(), color="steelblue", edgecolor="black")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Test Episodes")
    ax1.set_title("Test Episodes Distribution by Year")
    ax1.set_xticks(list(year_counts.keys()))

    ax2 = axes[1]
    heatmap_data = np.zeros((len(years), len(months)))
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            heatmap_data[i, j] = counts.get((year, month), 0)

    im = ax2.imshow(heatmap_data, aspect="auto", cmap="Blues")
    ax2.set_xticks(range(len(months)))
    ax2.set_xticklabels([f"{m}" for m in months])
    ax2.set_yticks(range(len(years)))
    ax2.set_yticklabels(years)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Year")
    ax2.set_title("Test Episodes per (Year, Month)")

    for i in range(len(years)):
        for j in range(len(months)):
            val = int(heatmap_data[i, j])
            if val > 0:
                ax2.text(j, i, str(val), ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax2, label="Episodes")
    plt.tight_layout()
    plt.savefig("test_episodes_distribution.png", dpi=150)
    plt.show()

    print("\nTest episodes per (year, month):")
    for key in sorted(counts.keys()):
        print(f"  {key[0]}-{key[1]:02d}: {counts[key]} episodes")

    from scipy.constants import convert_temperature

    n_episodes = len(test_indices)
    episode_length = horizon + max_steps

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("Test Episodes: Price, Temperature, Solar (10 sample episodes)", fontsize=14)

    time_hours = np.arange(episode_length) * 0.25

    sample_indices = [test_indices[i] for i in range(0, n_episodes, n_episodes // 10)][:10]

    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]

    for i, idx in enumerate(sample_indices):
        prices = dataset.get_price(idx, episode_length)
        temps = dataset.get_temperature(idx, episode_length)
        temps_celsius = convert_temperature(temps, "K", "C")
        solar = dataset.get_solar(idx, episode_length)

        start_date = dataset.index[idx]
        label = start_date.strftime("%Y-%m-%d")
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]

        axes2[0].plot(
            time_hours, prices, color=color, linewidth=1.0, linestyle=linestyle, label=label
        )
        axes2[1].plot(
            time_hours, temps_celsius, color=color, linewidth=1.0, linestyle=linestyle, label=label
        )
        axes2[2].plot(
            time_hours, solar, color=color, linewidth=1.0, linestyle=linestyle, label=label
        )

    axes2[0].set_xlabel("Hours")
    axes2[0].set_ylabel("Price [EUR/kWh]")
    axes2[0].set_title("Price")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(fontsize=7, loc="upper right")

    axes2[1].set_xlabel("Hours")
    axes2[1].set_ylabel("Temperature [°C]")
    axes2[1].set_title("Temperature")
    axes2[1].grid(True, alpha=0.3)

    axes2[2].set_xlabel("Hours")
    axes2[2].set_ylabel("Solar [W/m²]")
    axes2[2].set_title("Solar Radiation")
    axes2[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_episodes_data.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved test_episodes_data.png")

    # =========================================================================
    # Plot 3: Timeline visualization for Train/Test Split (Random Mode)
    # =========================================================================
    all_years = sorted(set(dataset.index.year))

    fig3, ax3 = plt.subplots(figsize=(16, 6))
    ax3.set_title("Test Episodes Timeline with Exclusion Zones (Random Mode)", fontsize=14)

    for idx in test_indices:
        start_date = dataset.index[idx]
        year = start_date.year
        day_of_year = start_date.dayofyear

        excl_start_idx = max(0, idx - episode_length + 1)
        excl_end_idx = min(len(dataset) - 1, idx + episode_length - 1)
        excl_start_date = dataset.index[excl_start_idx]
        excl_end_date = dataset.index[excl_end_idx]

        excl_start_day = excl_start_date.dayofyear
        excl_end_day = excl_end_date.dayofyear

        if excl_start_date.year == year and excl_end_date.year == year:
            ax3.barh(
                y=year,
                width=excl_end_day - excl_start_day,
                left=excl_start_day,
                height=0.6,
                color="lightcoral",
                alpha=0.4,
                edgecolor="none",
            )

        ax3.barh(
            y=year,
            width=episode_length / 4 / 24,
            left=day_of_year,
            height=0.6,
            color="steelblue",
            alpha=0.8,
            edgecolor="darkblue",
            linewidth=0.5,
        )

    from datetime import date as dt_date

    def get_month_starts(year) -> list[int]:
        """Get day-of-year for the first day of each month."""
        return [dt_date(year, m, 1).timetuple().tm_yday for m in range(1, 13)]

    month_starts = get_month_starts(2023)
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    ax3.set_xticks(month_starts)
    ax3.set_xticklabels(month_labels)
    ax3.set_xlabel("Day of Year")
    ax3.set_ylabel("Year")
    ax3.set_yticks(all_years)
    ax3.set_xlim(0, 366)
    ax3.set_ylim(min(all_years) - 0.5, max(all_years) + 0.5)
    ax3.grid(True, alpha=0.3, axis="x")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", edgecolor="darkblue", label="Test Episode"),
        Patch(facecolor="lightcoral", alpha=0.4, label="Exclusion Zone (no training)"),
    ]
    ax3.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig("train_test_split_timeline.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved train_test_split_timeline.png")

    # =========================================================================
    # Plot 4: Timeline visualization for Continual Learning Mode
    # =========================================================================
    continual_cfg = DataConfig(mode="continual")
    continual_dataset = HvacDataset(cfg=continual_cfg)
    continual_dataset.reset_continual_index()

    continual_episodes = []
    max_episodes = 50
    for _ in range(max_episodes):
        start_idx, ep_max_steps = continual_dataset._sample_continual(horizon)
        if ep_max_steps == 0:
            break
        continual_episodes.append((start_idx, ep_max_steps))

    print(f"Continual learning episodes: {len(continual_episodes)}")

    fig4, ax4 = plt.subplots(figsize=(16, 6))
    ax4.set_title("Continual Learning Episodes Timeline", fontsize=14)

    n_episodes = len(continual_episodes)
    colors = plt.cm.viridis(np.linspace(0, 1, n_episodes))

    for i, (start_idx, ep_steps) in enumerate(continual_episodes):
        start_date = continual_dataset.index[start_idx]
        end_idx = min(start_idx + ep_steps - 1, len(continual_dataset) - 1)
        end_date = continual_dataset.index[end_idx]

        year = start_date.year
        start_day = start_date.dayofyear
        end_day = end_date.dayofyear

        if end_date.year != year:
            days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            first_part_days = days_in_year - start_day + 1
            second_part_days = end_date.dayofyear

            ax4.barh(
                y=year,
                width=first_part_days,
                left=start_day,
                height=0.6,
                color=colors[i],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )
            ax4.barh(
                y=year + 1,
                width=second_part_days,
                left=1,
                height=0.6,
                color=colors[i],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )
        else:
            ax4.barh(
                y=year,
                width=end_day - start_day + 1,
                left=start_day,
                height=0.6,
                color=colors[i],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

    ax4.set_xticks(get_month_starts(2023))
    ax4.set_xticklabels(month_labels)
    ax4.set_xlabel("Day of Year")
    ax4.set_ylabel("Year")
    ax4.set_yticks(all_years)
    ax4.set_xlim(0, 366)
    ax4.set_ylim(min(all_years) - 0.5, max(all_years) + 0.5)
    ax4.grid(True, alpha=0.3, axis="x")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=1, vmax=n_episodes))
    sm.set_array([])
    plt.colorbar(sm, ax=ax4, label="Episode Order", shrink=0.8)

    plt.tight_layout()
    plt.savefig("continual_learning_timeline.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved continual_learning_timeline.png")
