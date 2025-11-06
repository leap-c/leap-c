"""Module defining the abstract interface for differentiable, parameterized controllers in
PyTorch."""

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Union

import gymnasium as gym
from numpy import ndarray
from torch import Tensor, nn


class ParameterizedController(nn.Module, metaclass=ABCMeta):
    """Abstract base class for differentiable parameterized controllers.

    Attributes:
        collate_fn_map: Optional mapping from data types to custom collate functions for batching.
            Should be provided in cases the controller needs specific collate functions, usually for
            custom data types. For more information, please refer to, e.g.,
            https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.default_collate.
    """

    collate_fn_map: dict[Union[type, tuple[type, ...]], Callable] | None = None

    @abstractmethod
    def forward(self, obs: Tensor, param: Tensor, ctx: Any = None) -> tuple[Any, Tensor]:
        """Computes action from observation, parameters and internal context.

        Args:
            obs (Tensor): Observation input to the controller (e.g., state vector).
            param (Tensor): Parameters that define the behavior of the controller.
            ctx (Any, optional): Optional internal context passed between invocations.

        Returns:
            ctx (Any, optional): A context object containing any intermediate values needed for
                backward computation and further invocations.
                Stats to be logged are expected to be passed in the field `ctx.log`, which should be
                a dictionary mapping string keys to float values.
            action (Tensor): The computed action.
        """
        ...

    def jacobian_action_param(self, ctx: Any) -> ndarray:
        """Computes `da/dp`, the Jacobian of the action with respect to the parameters.

        This can be used by methods for regularization.

        Args:
            ctx (Any): The context object from the `forward` pass.

        Returns:
            array: The Jacobian of the initial action with respect to the parameters.

        Raises:
            NotImplementedError: If jacobian_action_param is not implemented.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `jacobian_action_param`"
        )

    @property
    @abstractmethod
    def param_space(self) -> gym.Space:
        """Describes the parameter space of the controller.

        Returns:
            Gymnasium.Space: An object describing the valid space of parameters.
        """
        ...

    @abstractmethod
    def default_param(self, obs: ndarray | None) -> ndarray:
        """Provides a default parameter configuration for the controller.

        Args:
            obs (array, optional): Observation input to the controller (e.g., state
                vector), used to condition the default parameters. Can be `None` if not needed.

        Returns:
            array: A default parameter array matching the expected shape of `param` in `forward`.
        """
        ...
