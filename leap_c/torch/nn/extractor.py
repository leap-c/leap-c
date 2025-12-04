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
    Time and state features are normalized and concatenated with the conv output.
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

        # Output: time (2) + state (3) + forecast features
        self._output_size = 2 + 3 + self.cfg.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from HVAC observations.

        Args:
            x: Input tensor of shape (batch, obs_dim).

        Returns:
            Feature tensor of shape (batch, output_size).
        """
        n = self.cfg.n_forecast

        # Split observation into components
        time_features = x[:, :2]  # quarter_hour, day_of_year
        state = x[:, 2:5]  # Ti, Th, Te

        # Extract and stack forecasts: (batch, 3, n_forecast)
        temp_forecast = x[:, 5 : 5 + n]
        solar_forecast = x[:, 5 + n : 5 + 2 * n]
        price_forecast = x[:, 5 + 2 * n : 5 + 3 * n]
        forecasts = torch.stack([temp_forecast, solar_forecast, price_forecast], dim=1)

        # Normalize using min-max scaling based on observation space bounds
        # Time features
        time_low = torch.tensor(self.observation_space.low[:2], device=x.device, dtype=x.dtype)
        time_high = torch.tensor(self.observation_space.high[:2], device=x.device, dtype=x.dtype)
        time_norm = (time_features - time_low) / (time_high - time_low + 1e-8)

        # State
        state_low = torch.tensor(self.observation_space.low[2:5], device=x.device, dtype=x.dtype)
        state_high = torch.tensor(self.observation_space.high[2:5], device=x.device, dtype=x.dtype)
        state_norm = (state - state_low) / (state_high - state_low + 1e-8)

        # Forecasts - normalize each channel
        forecast_low = torch.tensor(
            [
                self.observation_space.low[5],  # temp
                self.observation_space.low[5 + n],  # solar
                self.observation_space.low[5 + 2 * n],  # price
            ],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 3, 1)
        forecast_high = torch.tensor(
            [
                self.observation_space.high[5],
                self.observation_space.high[5 + n],
                self.observation_space.high[5 + 2 * n],
            ],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 3, 1)
        forecasts_norm = (forecasts - forecast_low) / (forecast_high - forecast_low + 1e-8)

        # Process forecasts with 1D conv
        conv_out = self.forecast_conv(forecasts_norm)  # (batch, channels[-1], n_forecast)
        pooled = self.forecast_pool(conv_out).squeeze(-1)  # (batch, channels[-1])
        forecast_features = self.forecast_linear(pooled)  # (batch, output_dim)

        # Concatenate all features
        return torch.cat([time_norm, state_norm, forecast_features], dim=1)

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
