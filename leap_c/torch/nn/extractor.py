"""This module contains classes for feature extraction from observations.

We provide an abstraction to allow algorithms to be applied to different
types of observations and using different neural network architectures.
"""

from abc import ABC, abstractmethod
from typing import Literal

import gymnasium as gym
import torch.nn as nn

from leap_c.torch.nn.scale import min_max_scaling

ExtractorName = Literal["identity", "scaling"]


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


EXTRACTOR_REGISTRY: dict[ExtractorName, type[Extractor]] = {
    "identity": IdentityExtractor,
    "scaling": ScalingExtractor,
}


def get_extractor_cls(name: ExtractorName) -> type[Extractor]:
    """Get the extract class corresponding to the given name.

    Args:
        name ({"identity", "scaling"}): The name of the extractor.

    Returns:
        type[Extractor]: The class of the requested extractor.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor type: `{name}`")
    return EXTRACTOR_REGISTRY[name]
