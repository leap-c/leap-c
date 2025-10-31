"""Module defining the abstract interface for parameterized planners in PyTorch."""

from abc import abstractmethod
from typing import Any, Callable, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .controller import ParameterizedController


class ParameterizedPlanner(nn.Module):
    """Abstract base class for parameterized Planners.

    Attributes:
        collate_fn_map: Optional mapping from data types to custom collate
            functions for batching. Should be provided in cases the planner needs
            specific collate functions, usually for custom data types. For more
            information, please refer to, e.g.,
            https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.default_collate
    """

    collate_fn_map: dict[Union[type, tuple[type, ...]], Callable] | None = None

    @abstractmethod
    def forward(
        self, obs, action=None, param=None, ctx=None
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes state and control trajectories from an initial observation,
        optional initial action, parameters, and internal context.

        Args:
            obs: Initial observation input to the planner (e.g., state vector).
            action: Optional initial action (torch.Tensor).
            param: Parameters that define the behavior of the planner.
            ctx: Optional internal context passed between invocations.

        Returns:
            ctx: A context object containing any intermediate values
                needed for backward computation and further invocations.
            x: The computed sequence of states (torch.Tensor).
                Expected shape (T+1, *state_dims).
            u: The computed sequence of controls (torch.Tensor).
                Expected shape (T, *control_dims).
            value: The cost value of the computed trajectory (torch.Tensor).
        """
        ...

    @property
    @abstractmethod
    def param_space(self) -> gym.Space:
        """Describes the parameter space of the planner.

        Returns:
            An object describing the valid space of parameters.
        """
        ...

    @abstractmethod
    def default_param(self, obs) -> np.ndarray:
        """Provides a default parameter configuration for the planner.

        Args:
            obs: Initial observation input to the planner (e.g., state vector).

        Returns:
            A default parameter array or structure matching the expected
            input of `param`.
        """
        ...


class ControllerFromPlanner(ParameterizedController):
    """Wraps a ParameterizedPlanner as a ParameterizedController.

    This allows using a planner in contexts where a controller is expected,
    by extracting the first action from the planned trajectory.

    Args:
        planner: An instance of ParameterizedPlanner to be wrapped.
    """

    def __init__(self, planner: ParameterizedPlanner):
        super().__init__()
        self.planner = planner

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        """Computes the first action from the planner's trajectory.

        Args:
            obs: Observation input to the controller (e.g., state vector).
            param: Parameters that define the behavior of the controller.
            ctx: Optional internal context passed between invocations.

        Returns:
            ctx: A context object containing any intermediate values
                needed for backward computation and further invocations.
            action: The computed first action from the planned trajectory.
        """
        ctx_planner, _, u_traj, _ = self.planner.forward(obs, param=param, ctx=ctx)
        action = u_traj[0]
        return ctx_planner, action

    @property
    def param_space(self) -> gym.Space:
        """Describes the parameter space of the underlying planner.

        Returns:
            An object describing the valid space of parameters.
        """
        return self.planner.param_space

    def default_param(self, obs) -> np.ndarray:
        """Provides a default parameter configuration for the underlying planner.

        Args:
            obs: Observation input to the controller (e.g., state vector).

        Returns:
            A default parameter array or structure matching the expected
            input of `param`.
        """
        return self.planner.default_param(obs)


def ensure_controller(
    planner_or_controller: ParameterizedPlanner | ParameterizedController,
) -> ParameterizedController:
    """Ensures that the provided object is a ParameterizedController.

    If a ParameterizedPlanner is provided, it is wrapped in a
    ControllerFromPlanner.

    Args:
        planner_or_controller: An instance of ParameterizedPlanner or
            ParameterizedController.

    Returns:
        An instance of ParameterizedController.
    """
    if isinstance(planner_or_controller, ParameterizedController):
        return planner_or_controller
    elif isinstance(planner_or_controller, ParameterizedPlanner):
        return ControllerFromPlanner(planner_or_controller)
    raise TypeError("Input must be an instance of ParameterizedPlanner or ParameterizedController.")
