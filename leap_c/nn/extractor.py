"""This module contains classes for feature extraction from observations.

We provide an abstraction to allow algorithms to be aplied to different
types of observations and using different neural network architectures.
"""
import torch.nn as nn
from abc import ABC, abstractmethod

import gymnasium as gym


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


class IdentityExtractor(Extractor):
    """An extractor that returns the input as is."""

    def __init__(self, env: gym.Env) -> None:
        """Initializes the extractor.

        Args:
            env: The environment to extract features from.
        """
        super().__init__(env)
        assert len(env.observation_space.shape) == 1, "IdentityExtractor only supports 1D observations."  # type: ignore

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
        return self.env.observation_space.shape[0]  # type: ignore

