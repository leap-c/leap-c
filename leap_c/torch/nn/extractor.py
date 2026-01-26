"""This module contains classes for feature extraction from observations.

We provide an abstraction to allow algorithms to be applied to different
types of observations and using different neural network architectures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import gymnasium as gym
import torch
import torch.nn as nn
from tensordict import TensorDict

from leap_c.torch.nn.scale import min_max_scaling

ExtractorName = Literal["identity", "scaling", "hvac"]


class Extractor(nn.Module, ABC):
    """An abstract class for feature extraction from observations."""

    def __init__(self, observation_space: gym.Space) -> None:
        """Initializes the extractor.

        Args:
            observation_space: The observation space of the environment.
        """
        super().__init__()
        self.observation_space = observation_space

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Returns the embedded vector size."""


class ScalingExtractor(Extractor):
    """An extractor that returns the input normalized to the range [0, 1], using min-max scaling."""

    def __init__(self, observation_space: gym.spaces.Box) -> None:
        """Initializes the extractor.

        Args:
            observation_space: The observation space of the environment. Only works for Box spaces.
        """
        super().__init__(observation_space)

        if len(observation_space.shape) != 1:  # type: ignore
            raise ValueError("ScalingExtractor only supports 1D observations.")

    def forward(self, x):
        """Returns the input normalized to the range [0, 1], using min-max scaling.

        Args:
            x: The input tensor.

        Returns:
            The normalized tensor.
        """
        y = min_max_scaling(x, self.observation_space)  # type: ignore
        return y

    @property
    def output_size(self) -> int:
        return self.observation_space.shape[0]  # type: ignore


class IdentityExtractor(Extractor):
    """An extractor that returns the input as is."""

    def __init__(self, observation_space: gym.Space) -> None:
        """Initializes the extractor.

        Args:
            observation_space: The observation space of the environment.
        """
        super().__init__(observation_space)
        assert (
            len(observation_space.shape) == 1  # type: ignore
        ), "IdentityExtractor only supports 1D observations."

    def forward(self, x):
        """Returns the input as is.

        Args:
            x: The input tensor.

        Returns:
            The input tensor.
        """
        return x

    @property
    def output_size(self) -> int:
        return self.observation_space.shape[0]  # type: ignore


@dataclass
class HvacExtractorConfig:
    """Configuration for the HVAC extractor with 1D convolutions for forecasts.

    Attributes:
        n_forecast: Number of forecast steps (default 96 for 24h at 15min intervals).
        conv_channels: List of channel sizes for conv layers [input -> hidden -> ... -> output].
        kernel_size: Kernel size for 1D convolutions.
        output_dim: Final output dimension after conv layers (uses adaptive pooling + linear).
    """

    n_forecast: int = 96
    conv_channels: list[int] = field(default_factory=lambda: [3, 16, 32])
    kernel_size: int = 5
    output_dim: int = 32


class HvacExtractor(Extractor):
    """An extractor for HVAC environments that uses 1D convolutions for forecast data.

    This extractor expects a flat observation with the following structure:
    - [0:2]: Time features (quarter_hour, day_of_year)
    - [2:5]: State (Ti, Th, Te)
    - [5:5+N]: Ambient temperature forecast
    - [5+N:5+2N]: Solar radiation forecast
    - [5+2N:5+3N]: Electricity price forecast

    The forecasts are stacked as channels and processed with 1D convolutions.
    Time features are embedded using sin/cos transformations. State features are
    normalized. The electricity price forecast is normalized per-instance, with
    its mean and scale reported as additional features.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        cfg: HvacExtractorConfig | None = None,
    ) -> None:
        """Initializes the HVAC extractor.

        Args:
            observation_space: The observation space of the environment.
            cfg: Configuration for the extractor. If None, uses defaults.
        """
        super().__init__(observation_space)

        self.cfg = cfg if cfg is not None else HvacExtractorConfig()

        # Validate observation space
        expected_size = 5 + 3 * self.cfg.n_forecast
        if observation_space.shape[0] != expected_size:
            raise ValueError(
                f"Expected observation size {expected_size}, got {observation_space.shape[0]}. "
                f"Check n_forecast={self.cfg.n_forecast} matches the environment."
            )

        # Build 1D conv layers for forecasts
        conv_layers = []
        channels = self.cfg.conv_channels
        for i in range(len(channels) - 1):
            conv_layers.append(
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=self.cfg.kernel_size,
                    padding="same",
                )
            )
            conv_layers.append(nn.ReLU())
        self.forecast_conv = nn.Sequential(*conv_layers)

        # Adaptive pooling + linear to get fixed output size
        self.forecast_pool = nn.AdaptiveAvgPool1d(1)
        self.forecast_linear = nn.Linear(channels[-1], self.cfg.output_dim)

        # Output: time (4) + state (3) + forecast features + price stats (2)
        self._output_size = 4 + 3 + self.cfg.output_dim + 2

    def forward(
        self, x: dict[str, torch.Tensor | dict[str, torch.Tensor]] | TensorDict
    ) -> torch.Tensor:
        """Extract features from HVAC observations.

        Args:
            x: Input dict or TensorDict with:
               - "time": dict with "quarter_hour", "day_of_year", "day_of_week" tensors
               - "state": tensor of shape (batch, 3) containing [Ti, Th, Te]
               - "forecast": dict with "temperature", "solar", "price" tensors

        Returns:
            Feature tensor of shape (batch, output_size).
        """
        n = self.cfg.n_forecast
        obs_space: gym.spaces.Box = self.observation_space  # type: ignore

        # Split observation into components
        time_features = x[:, :2]  # quarter_hour, day_of_year
        state = x[:, 2:5]  # Ti, Th, Te

        # 1. Time Embedding (Sin/Cos)
        # quarter_hour (0-95), day_of_year (0-365)
        qh = time_features[:, 0:1]
        doy = time_features[:, 1:2]

        qh_sin = torch.sin(2 * torch.pi * qh / 96.0)
        qh_cos = torch.cos(2 * torch.pi * qh / 96.0)
        doy_sin = torch.sin(2 * torch.pi * doy / 366.0)
        doy_cos = torch.cos(2 * torch.pi * doy / 366.0)
        time_embedding = torch.cat([qh_sin, qh_cos, doy_sin, doy_cos], dim=1)

        # 2. State Normalization
        state_low = torch.tensor(obs_space.low[2:5], device=x.device, dtype=x.dtype)
        state_high = torch.tensor(obs_space.high[2:5], device=x.device, dtype=x.dtype)
        state_norm = (state - state_low) / (state_high - state_low + 1e-8)

        # 3. Forecast Processing
        temp_forecast = x[:, 5 : 5 + n]
        solar_forecast = x[:, 5 + n : 5 + 2 * n]
        price_forecast = x[:, 5 + 2 * n : 5 + 3 * n]

        # Price forecast: normalize per instance and extract mean/std
        price_mean = price_forecast.mean(dim=1, keepdim=True)
        price_std = price_forecast.std(dim=1, keepdim=True) + 1e-8
        price_forecast_norm = (price_forecast - price_mean) / price_std

        # Normalize price_mean and price_std using global bounds
        p_low = obs_space.low[5 + 2 * n]
        p_high = obs_space.high[5 + 2 * n]
        price_mean_norm = (price_mean - p_low) / (p_high - p_low + 1e-8)
        price_std_norm = price_std / (p_high - p_low + 1e-8)  # Scale relative to range

        # Other forecasts: global min-max normalization
        temp_low, temp_high = obs_space.low[5], obs_space.high[5]
        solar_low, solar_high = (
            obs_space.low[5 + n],
            obs_space.high[5 + n],
        )

        temp_forecast_norm = (temp_forecast - temp_low) / (temp_high - temp_low + 1e-8)
        solar_forecast_norm = (solar_forecast - solar_low) / (solar_high - solar_low + 1e-8)

        temp_forecast_norm = (temp_forecast - temp_low) / (temp_high - temp_low + 1e-8)
        solar_forecast_norm = (solar_forecast - solar_low) / (solar_high - solar_low + 1e-8)

        # Stack as channels: (batch, 3, n_forecast)
        forecasts = torch.stack(
            [temp_forecast_norm, solar_forecast_norm, price_forecast_norm], dim=1
        )

        # Process forecasts with 1D conv
        conv_out = self.forecast_conv(forecasts)  # (batch, channels[-1], n_forecast)
        pooled = self.forecast_pool(conv_out).squeeze(-1)  # (batch, channels[-1])
        forecast_features = self.forecast_linear(pooled)  # (batch, output_dim)

        # Concatenate all features
        return torch.cat(
            [
                time_embedding,
                state_norm,
                forecast_features,
                price_mean_norm,
                price_std_norm,
            ],
            dim=1,
        )

    @property
    def output_size(self) -> int:
        return self._output_size


EXTRACTOR_REGISTRY = {
    "identity": IdentityExtractor,
    "scaling": ScalingExtractor,
    "hvac": HvacExtractor,
}


def get_extractor_cls(name: ExtractorName):
    try:
        return EXTRACTOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown extractor type: {name}")
