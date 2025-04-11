"""This module contains classes for feature extraction from observations.

We provide an abstraction to allow algorithms to be aplied to different
types of observations and using different neural network architectures.
"""
from abc import ABC, abstractmethod

import gymnasium as gym
import torch.nn as nn

from leap_c.nn.utils import min_max_scaling


class Extractor(nn.Module, ABC):
    """An abstract class for feature extraction from observations."""

    def __init__(self, env: gym.Env) -> None:
        """Initializes the extractor.

        Args:
            env: The environment to extract features from.
        """
        super().__init__()
        self.env = env

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Returns the embedded vector size."""


class ScalingExtractor(Extractor):
    """An extractor that returns the input as is."""

    def __init__(self, env: gym.Env) -> None:
        """Initializes the extractor.

        Args:
            env: The environment to extract features from.
        """
        super().__init__(env)

        if not hasattr(env, "observation_space"):
            raise ValueError(
                "The environment must have an observation space."
            )

        if len(env.observation_space.shape) != 1:  # type: ignore
            raise ValueError(
                "ScalingExtractor only supports 1D observations."
            )

    def forward(self, x):
        """Returns the input as is.

        Args:
            x: The input tensor.

        Returns:
            The input tensor.
        """
        return min_max_scaling(x, self.env.observation_space)  # type: ignore

    @property
    def output_size(self) -> int:
        """Returns the embedded vector size."""
        return self.env.observation_space.shape[0]  # type: ignore

