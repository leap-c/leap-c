"""Forecasting utilities for HVAC environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from leap_c.examples.hvac.dataset import HvacDataset


@dataclass(kw_only=True)
class TemperatureUncertaintyConfig:
    """Configuration for temperature forecast uncertainty parameters (AR1 with Normal).

    Attributes:
        F0: Mean of the initial error distribution.
        K0: Standard deviation of the initial error distribution.
        F: Autocorrelation factor (0-1).
        mu: Mean of the AR error distribution.
        K: Standard deviation of the AR error distribution.
    """

    F0: float
    K0: float
    F: float
    mu: float
    K: float


@dataclass(kw_only=True)
class SolarUncertaintyConfig:
    """Configuration for solar forecast uncertainty parameters (AR1 with Laplace).

    Attributes:
        ag0: Mean of the initial error distribution.
        bg0: Scale of the initial error distribution.
        phi: Autocorrelation factor (0-1).
        ag: Mean of the AR error distribution.
        bg: Scale of the AR error distribution.
    """

    ag0: float
    bg0: float
    phi: float
    ag: float
    bg: float


@dataclass(kw_only=True)
class ForecastConfig:
    """Configuration for forecasting and uncertainty.

    Attributes:
        horizon_hours: Prediction horizon in hours.
        temp_uncertainty: Temperature uncertainty configuration.
            Can be an UncertaintyConfig object, a preset level string
            ('low', 'medium', 'high'), or None to disable.
        solar_uncertainty: Solar uncertainty configuration.
            Can be a SolarUncertaintyConfig object, a preset level string
            ('low', 'medium', 'high'), or None to disable.

    Note:
        Source for forecast params can be found at:
        https://github.com/ibpsa/project1-boptest/blob/master/forecast/
        forecast_uncertainty_params.json
    """

    horizon_hours: int = 24  # prediction horizon in hours
    temp_uncertainty: (
        TemperatureUncertaintyConfig | Literal["low", "medium", "high", "negative_bias"] | None
    ) = "negative_bias"
    solar_uncertainty: (
        SolarUncertaintyConfig | Literal["low", "medium", "high", "negative_bias"] | None
    ) = "negative_bias"

    def __post_init__(self):
        """Resolve string literals to actual config objects."""
        if self.temp_uncertainty == "low":
            self.temp_uncertainty = TemperatureUncertaintyConfig(
                F0=0.0, K0=0.6, F=0.92, mu=0.0, K=0.4
            )
        elif self.temp_uncertainty == "medium":
            self.temp_uncertainty = TemperatureUncertaintyConfig(
                F0=0.15, K0=1.2, F=0.93, mu=0.0, K=0.6
            )
        elif self.temp_uncertainty == "high":
            self.temp_uncertainty = TemperatureUncertaintyConfig(
                F0=-0.58, K0=1.5, F=0.95, mu=-0.015, K=0.7
            )
        elif self.temp_uncertainty == "negative_bias":
            self.temp_uncertainty = TemperatureUncertaintyConfig(
                F0=0.0, K0=0.0, F=0.2, mu=-0.8, K=0.2
            )

        # Resolve solar uncertainty string to config
        if self.solar_uncertainty == "low":
            self.solar_uncertainty = SolarUncertaintyConfig(
                ag0=4.44, bg0=57.42, phi=0.62, ag=1.86, bg=45.64
            )
        elif self.solar_uncertainty == "medium":
            self.solar_uncertainty = SolarUncertaintyConfig(
                ag0=15.02, bg0=122.6, phi=0.63, ag=4.44, bg=91.97
            )
        elif self.solar_uncertainty == "high":
            self.solar_uncertainty = SolarUncertaintyConfig(
                ag0=32.09, bg0=119.94, phi=0.67, ag=10.63, bg=87.44
            )
        elif self.solar_uncertainty == "negative_bias":
            self.solar_uncertainty = SolarUncertaintyConfig(
                ag0=0.0, bg0=1.0, phi=0.7, ag=-5.0, bg=1.8
            )


class Forecaster:
    """Generate forecasts with uncertainty from historic data.

    This class handles the generation of temperature and solar radiation forecasts
    with configurable uncertainty models.

    Attributes:
        cfg: Forecast configuration.
        horizon_hours: Prediction horizon in hours (for convenience).
    """

    def __init__(self, cfg: ForecastConfig | None = None) -> None:
        """Initialize the forecaster.

        Args:
            cfg: Forecast configuration. If None, uses default configuration.
        """
        self.cfg = cfg or ForecastConfig()
        self.horizon_hours = self.cfg.horizon_hours

    def get_forecast(
        self,
        idx: int,
        dataset: HvacDataset,
        N_forecast: int,
        np_random,
        historical_data_bias: bool = False,
    ) -> dict[str, np.ndarray]:
        """Generate both temperature and solar forecasts.

        Args:
            idx: Current data index.
            dataset: Full dataset containing 'price', 'temperature' and 'solar'.
            N_forecast: Number of forecast steps.
            np_random: Random number generator.
            historical_data_bias: If True, use historical data without uncertainty.

        Returns:
            Dictionary with 'price', 'temperature' and 'solar' forecast arrays.
        """
        price_forecast = dataset.get_price(idx=idx, horizon=N_forecast)

        if historical_data_bias:
            return {
                "price": price_forecast,
                "temperature": dataset.get_temperature_forecast(idx, N_forecast),
                "solar": dataset.get_solar_forecast(idx, N_forecast),
            }

        return {
            "price": price_forecast,
            "temperature": self.get_temperature_forecast(idx, dataset.data, N_forecast, np_random),
            "solar": self.get_solar_forecast(idx, dataset.data, N_forecast, np_random),
        }

    def get_temperature_forecast(
        self,
        idx: int,
        data: pd.DataFrame,
        N_forecast: int,
        np_random,
    ) -> np.ndarray:
        """Generate temperature forecast with AR(1) uncertainty.

        Args:
            idx: Current data index.
            data: Full dataset containing 'temperature' (ambient temperature).
            N_forecast: Number of forecast steps.
            np_random: Random number generator.

        Returns:
            Array of forecasted temperatures with uncertainty.
        """
        base_forecast = data["temperature"].iloc[idx : idx + N_forecast].to_numpy()

        if self.cfg.temp_uncertainty:
            error = predict_ar1_error(
                hp=N_forecast,
                initial_mean=self.cfg.temp_uncertainty.F0,
                initial_scale=self.cfg.temp_uncertainty.K0,
                ar_factor=self.cfg.temp_uncertainty.F,
                ar_mean=self.cfg.temp_uncertainty.mu,
                ar_scale=self.cfg.temp_uncertainty.K,
                np_random=np_random,
                distribution="normal",
            )
            return base_forecast + error

        return base_forecast

    def get_solar_forecast(
        self,
        idx: int,
        data: pd.DataFrame,
        N_forecast: int,
        np_random,
    ) -> np.ndarray:
        """Generate solar radiation forecast with optional Laplace uncertainty.

        Args:
            idx: Current data index.
            data: Full dataset containing 'solar' radiation.
            N_forecast: Number of forecast steps.
            np_random: Random number generator.

        Returns:
            Array of forecasted solar radiation.
        """
        base_forecast = data["solar"].iloc[idx : idx + N_forecast].to_numpy()

        # Add uncertainty if configured
        if self.cfg.solar_uncertainty is not None:
            solar_unc = self.cfg.solar_uncertainty
            error = predict_ar1_error(
                hp=N_forecast,
                initial_mean=solar_unc.ag0,
                initial_scale=solar_unc.bg0,
                ar_factor=solar_unc.phi,
                ar_mean=solar_unc.ag,
                ar_scale=solar_unc.bg,
                np_random=np_random,
                distribution="laplace",
            )
            return np.maximum(0, base_forecast + error)

        return base_forecast


def predict_ar1_error(
    hp: int,
    initial_mean: float,
    initial_scale: float,
    ar_factor: float,
    ar_mean: float,
    ar_scale: float,
    np_random,
    distribution: Literal["normal", "laplace"] = "normal",
) -> np.ndarray:
    """Generate forecast error using AR(1) model with configurable distribution.

    This implements an autoregressive model of order 1:
        error[0] ~ Dist(initial_mean, initial_scale)
        error[t] ~ Dist(error[t-1] * ar_factor + ar_mean, ar_scale)

    Args:
        hp: Number of points in the prediction horizon.
        initial_mean: Mean/location of the initial error distribution.
        initial_scale: Scale/std of the initial error distribution.
        ar_factor: Autocorrelation factor (typically 0-1).
        ar_mean: Mean/location parameter for AR model.
        ar_scale: Scale/std parameter for AR model.
        np_random: Random number generator.
        distribution: Distribution type - "normal" or "laplace".

    Returns:
        Array containing error values at hp prediction points.
    """
    error = np.zeros(hp)

    # Sample initial error
    if distribution == "normal":
        distr = np_random.normal
    else:  # laplace
        distr = np_random.laplace

    error[0] = distr(initial_mean, initial_scale)

    # Generate AR(1) process
    for i in range(1, hp):
        mean_t = error[i - 1] * ar_factor + ar_mean
        error[i] = distr(mean_t, ar_scale)

    return error
