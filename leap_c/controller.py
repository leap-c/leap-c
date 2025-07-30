"""Module defining the abstract interface for differentiable, parameterized
controllers in PyTorch."""

from abc import abstractmethod
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class ParameterizedController(nn.Module):
    """Abstract base class for differentiable parameterized controllers."""

    # should be provided in cases the controller comes with specific collate
    # functions for different data types. This is an advanced feature!
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None

    @abstractmethod
    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        """Computes action from observation, parameters and internal context.

        Args:
            obs: Observation input to the controller (e.g., state vector).
            param: Parameters that define the behavior of the controller.
            ctx: Optional internal context passed between invocations.

        Returns:
            ctx: A context object containing any intermediate values
                needed for backward computation and further invocations.
            action: The computed action as a NumPy array.
        """
        ...

    def jacobian_action_param(self, ctx) -> np.ndarray:
        """Computes da/dp, the Jacobian of the action with respect to the
        parameters.

        This can be used by methods for regularization.

        Args:
            ctx: The context object from the `forward` pass.

        Returns:
            The Jacobian of the initial action with respect to the
            parameters.

        Raises:
            NotImplementedError: If jacobian_action_param is not implemented.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def param_space(self) -> gym.Space:
        """Describes the parameter space of the controller.

        Returns:
            An object describing the valid space of parameters.
        """
        ...

    @abstractmethod
    def default_param(self, obs) -> np.ndarray:
        """Provides a default parameter configuration for the controller.

        Args:
            obs: Observation input to the controller (e.g., state vector).

        Returns:
            A default parameter array or structure matching the expected input
            input of `param`.
        """
        ...
