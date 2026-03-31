from dataclasses import dataclass, field

import gymnasium as gym
import torch
import torch.nn as nn
from tensordict import TensorDict

from leap_c.torch.nn.extractor import Extractor


@dataclass
class I4bExtractorConfig:
    """Configuration for the I4B extractor with 1D convolutions for forecasts.

    Attributes:
        n_forecast: Number of forecast steps (default 96 for 24h at 15min intervals).
        conv_channels: List of channel sizes for conv layers [input -> hidden -> ... -> output].
            Input channels = 5: temperature, dhi, ghi, dni, price.
        kernel_size: Kernel size for 1D convolutions.
        output_dim: Final output dimension after conv layers (uses adaptive pooling + linear).
    """

    n_forecast: int = 96
    conv_channels: list[int] = field(default_factory=lambda: [5, 16, 32])
    kernel_size: int = 5
    output_dim: int = 32


class I4bExtractor(Extractor):
    """An extractor for I4B environments that uses 1D convolutions for forecast data.

    This extractor expects a Dict observation with the following structure:
    - "time": Dict with "quarter_hour", "day_of_year", "day_of_week" (each shape (1,))
    - "state": Tensor of shape (3,) containing [Ti, Th, Te]
    - "forecast": Dict with "temperature", "dhi", "ghi", "dni", "price" (each shape (N,))

    The forecasts are stacked as channels and processed with 1D convolutions.
    Time features are embedded using sin/cos transformations (6 features total:
    2 for quarter_hour, 2 for day_of_year, 2 for day_of_week).
    State features are normalized. All forecasts are normalized per-instance,
    with their mean and scale reported as additional features (10 total: 2 per forecast).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cfg: I4bExtractorConfig | None = None,
    ) -> None:
        """Initializes the I4B extractor.

        Args:
            observation_space: The observation space of the environment (Dict space).
            cfg: Configuration for the extractor. If None, uses defaults.
        """
        super().__init__(observation_space)

        self.cfg = cfg if cfg is not None else I4bExtractorConfig()

        # Validate observation space structure
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(
                f"I4bExtractor requires a Dict observation space, got {type(observation_space)}"
            )

        required_keys = {"state", "forecast"}
        if not required_keys.issubset(observation_space.spaces.keys()):
            raise ValueError(
                f"Observation space must contain keys {required_keys}, "
                f"got {observation_space.spaces.keys()}"
            )

        # Validate forecast length matches config
        forecast_space = observation_space["forecast"]["T_amb"]  # type: ignore[index]
        if forecast_space.shape[0] != self.cfg.n_forecast:
            raise ValueError(
                f"Expected forecast length {self.cfg.n_forecast}, "
                f"got {forecast_space.shape[0]}. "
                f"Check n_forecast in I4bExtractorConfig matches the environment."
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

        # Output dimensions:
        #  time (6: 2 each for qh, doy, dow)
        #  state (3)
        #  forecast features (output_dim)
        #  forecast stats (10: 2 each for temperature, dhi, ghi, dni, price)
        self._output_size = 6 + 3 + self.cfg.output_dim + 10

    def forward(
        self, x: dict[str, torch.Tensor | dict[str, torch.Tensor]] | TensorDict
    ) -> torch.Tensor:
        """Extract features from I4B observations.

        Args:
            x: Input dict or TensorDict with:
               - "time": dict with "quarter_hour", "day_of_year", "day_of_week" tensors
               - "state": tensor of shape (batch, 3) containing [Ti, Th, Te]
               - "forecast": dict with "temperature", "dhi", "ghi", "dni", "price" tensors

        Returns:
            Feature tensor of shape (batch, output_size).
        """
        obs_space: gym.spaces.Dict = self.observation_space  # type: ignore

        # 1. Time Embedding (Sin/Cos) — take current time step (index 0)
        # quarter_hour (0-95), day_of_year (0-365), day_of_week (0-6)
        qh = x["forecast"]["quarter_hour"][:, :1].float()  # (batch, 1)
        doy = x["forecast"]["day_of_year"][:, :1].float()  # (batch, 1)
        dow = x["forecast"]["day_of_week"][:, :1].float()  # (batch, 1)

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
        temp_forecast = x["forecast"]["T_amb"]  # (batch, n_forecast)
        dhi_forecast = x["forecast"]["dhi"]  # (batch, n_forecast)
        ghi_forecast = x["forecast"]["ghi"]  # (batch, n_forecast)
        dni_forecast = x["forecast"]["dni"]  # (batch, n_forecast)
        price_forecast = x["forecast"]["price"]  # (batch, n_forecast)

        forecast_space = obs_space["forecast"]  # type: ignore[index]

        def _norm_forecast(fc, space_key):
            """Normalize forecast per instance; return (normalized, mean_norm, std_norm)."""
            mean = fc.mean(dim=1, keepdim=True)
            std = fc.std(dim=1, keepdim=True) + 1e-8
            fc_norm = (fc - mean) / std
            lo = forecast_space[space_key].low[0]  # type: ignore[index]
            hi = forecast_space[space_key].high[0]  # type: ignore[index]
            mean_norm = (mean - lo) / (hi - lo + 1e-8)
            std_norm = std / (hi - lo + 1e-8)
            return fc_norm, mean_norm, std_norm

        temp_forecast_norm, temp_mean_norm, temp_std_norm = _norm_forecast(temp_forecast, "T_amb")
        dhi_forecast_norm, dhi_mean_norm, dhi_std_norm = _norm_forecast(dhi_forecast, "dhi")
        ghi_forecast_norm, ghi_mean_norm, ghi_std_norm = _norm_forecast(ghi_forecast, "ghi")
        dni_forecast_norm, dni_mean_norm, dni_std_norm = _norm_forecast(dni_forecast, "dni")
        price_forecast_norm, price_mean_norm, price_std_norm = _norm_forecast(
            price_forecast, "price"
        )

        # Stack as channels: (batch, 5, n_forecast)
        forecasts = torch.stack(
            [
                temp_forecast_norm,
                dhi_forecast_norm,
                ghi_forecast_norm,
                dni_forecast_norm,
                price_forecast_norm,
            ],
            dim=1,
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
                dhi_mean_norm,
                dhi_std_norm,
                ghi_mean_norm,
                ghi_std_norm,
                dni_mean_norm,
                dni_std_norm,
                price_mean_norm,
                price_std_norm,
            ],
            dim=1,
        )

    @property
    def output_size(self) -> int:
        return self._output_size
