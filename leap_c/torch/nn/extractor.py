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

    This extractor expects a Dict observation with the following structure:
    - "time": Dict with "quarter_hour", "day_of_year", "day_of_week" (each shape (1,))
    - "state": Tensor of shape (3,) containing [Ti, Th, Te]
    - "forecast": Dict with "temperature", "solar", "price" (each shape (N,))

    The forecasts are stacked as channels and processed with 1D convolutions.
    Time features are embedded using sin/cos transformations (6 features total:
    2 for quarter_hour, 2 for day_of_year, 2 for day_of_week).
    State features are normalized. All forecasts are normalized per-instance,
    with their mean and scale reported as additional features (6 total: 2 per forecast).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cfg: HvacExtractorConfig | None = None,
    ) -> None:
        """Initializes the HVAC extractor.

        Args:
            observation_space: The observation space of the environment (Dict space).
            cfg: Configuration for the extractor. If None, uses defaults.
        """
        super().__init__(observation_space)

        self.cfg = cfg if cfg is not None else HvacExtractorConfig()

        # Validate observation space structure
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(
                f"HvacExtractor requires a Dict observation space, got {type(observation_space)}"
            )

        required_keys = {"time", "state", "forecast"}
        if not required_keys.issubset(observation_space.spaces.keys()):
            raise ValueError(
                f"Observation space must contain keys {required_keys}, "
                f"got {observation_space.spaces.keys()}"
            )

        # Validate forecast length matches config
        forecast_space = observation_space["forecast"]["temperature"]  # type: ignore[index]
        if forecast_space.shape[0] != self.cfg.n_forecast:
            raise ValueError(
                f"Expected forecast length {self.cfg.n_forecast}, "
                f"got {forecast_space.shape[0]}. "
                f"Check n_forecast in HvacExtractorConfig matches the environment."
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

        # Output of CNN:
        #  time (6: 2 each for qh, doy, dow)
        #  state (3)
        #  forecast features (output_dim)
        #  forecast stats (6)
        self._output_size = 6 + 3 + self.cfg.output_dim + 6

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
        obs_space: gym.spaces.Dict = self.observation_space  # type: ignore

        # 1. Time Embedding (Sin/Cos)
        # quarter_hour (0-95), day_of_year (0-365), day_of_week (0-6)
        qh = x["time"]["quarter_hour"]  # (batch, 1)
        doy = x["time"]["day_of_year"]  # (batch, 1)
        dow = x["time"]["day_of_week"]  # (batch, 1)

        qh_sin = torch.sin(2 * torch.pi * qh / 96.0)
        qh_cos = torch.cos(2 * torch.pi * qh / 96.0)
        doy_sin = torch.sin(2 * torch.pi * doy / 366.0)
        doy_cos = torch.cos(2 * torch.pi * doy / 366.0)
        dow_sin = torch.sin(2 * torch.pi * dow / 7.0)
        dow_cos = torch.cos(2 * torch.pi * dow / 7.0)
        time_embedding = torch.cat([qh_sin, qh_cos, doy_sin, doy_cos, dow_sin, dow_cos], dim=1)

        # 2. State Normalization
        state = x["state"]  # (batch, 3) - [Ti, Th, Te]

        state_space: gym.spaces.Box = obs_space["state"]  # type: ignore[index,assignment]
        state_low = torch.tensor(state_space.low, device=state.device, dtype=state.dtype)
        state_high = torch.tensor(state_space.high, device=state.device, dtype=state.dtype)
        state_norm = (state - state_low) / (state_high - state_low + 1e-8)

        # 3. Forecast Processing
        temp_forecast = x["forecast"]["temperature"]  # (batch, n_forecast)
        solar_forecast = x["forecast"]["solar"]  # (batch, n_forecast)
        price_forecast = x["forecast"]["price"]  # (batch, n_forecast)

        forecast_space = obs_space["forecast"]  # type: ignore[index]

        # Normalize each forecast per instance and extract mean/std
        # Temperature forecast
        temp_mean = temp_forecast.mean(dim=1, keepdim=True)
        temp_std = temp_forecast.std(dim=1, keepdim=True) + 1e-8
        temp_forecast_norm = (temp_forecast - temp_mean) / temp_std

        temp_low = forecast_space["temperature"].low[0]  # type: ignore[index]
        temp_high = forecast_space["temperature"].high[0]  # type: ignore[index]
        temp_mean_norm = (temp_mean - temp_low) / (temp_high - temp_low + 1e-8)
        temp_std_norm = temp_std / (temp_high - temp_low + 1e-8)

        # Solar forecast
        solar_mean = solar_forecast.mean(dim=1, keepdim=True)
        solar_std = solar_forecast.std(dim=1, keepdim=True) + 1e-8
        solar_forecast_norm = (solar_forecast - solar_mean) / solar_std

        solar_low = forecast_space["solar"].low[0]  # type: ignore[index]
        solar_high = forecast_space["solar"].high[0]  # type: ignore[index]
        solar_mean_norm = (solar_mean - solar_low) / (solar_high - solar_low + 1e-8)
        solar_std_norm = solar_std / (solar_high - solar_low + 1e-8)

        # Price forecast
        price_mean = price_forecast.mean(dim=1, keepdim=True)
        price_std = price_forecast.std(dim=1, keepdim=True) + 1e-8
        price_forecast_norm = (price_forecast - price_mean) / price_std

        price_low = forecast_space["price"].low[0]  # type: ignore[index]
        price_high = forecast_space["price"].high[0]  # type: ignore[index]
        price_mean_norm = (price_mean - price_low) / (price_high - price_low + 1e-8)
        price_std_norm = price_std / (price_high - price_low + 1e-8)

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
                temp_mean_norm,
                temp_std_norm,
                solar_mean_norm,
                solar_std_norm,
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
        entry = EXTRACTOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown extractor type: {name}")
    if isinstance(entry, str):
        module_path, cls_name = entry.rsplit(":", 1)
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, cls_name)
    return entry
