"""This module contains classes for feature extraction from observations.

We provide an abstraction to allow algorithms to be aplied to different
types of observations and using different neural network architectures.
"""

from abc import ABC, abstractmethod
from typing import Literal

import gymnasium as gym
import torch.nn as nn

from leap_c.torch.nn.scale import min_max_scaling


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
    """An extractor that returns the input normalized."""

    def __init__(self, observation_space: gym.Space) -> None:
        """Initializes the extractor.

        Args:
            observation_space: The observation space of the environment.
        """
        super().__init__(observation_space)

        if len(observation_space.shape) != 1:  # type: ignore
            raise ValueError("ScalingExtractor only supports 1D observations.")

    def forward(self, x):
        """Returns the input normalized.

        Args:
            x: The input tensor.

        Returns:
            The input tensor.
        """
        y = min_max_scaling(x, self.observation_space)  # type: ignore
        return y

    @property
    def output_size(self) -> int:
        """Returns the embedded vector size."""
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
        """Returns the embedded vector size."""
        return self.observation_space.shape[0]  # type: ignore


ExtractorName = Literal["identity", "scaling"]


EXTRACTOR_REGISTRY = {
    "identity": IdentityExtractor,
    "scaling": ScalingExtractor,
}


def get_extractor_cls(name: ExtractorName):
    try:
        return EXTRACTOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown extractor type: {name}")
