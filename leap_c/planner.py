"""Module defining the abstract interface for differentiable, parameterized planners in PyTorch."""

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Generic

import gymnasium as gym
from numpy import ndarray
from torch import Tensor, nn

from leap_c.controller import CtxType, ParameterizedController


class ParameterizedPlanner(nn.Module, Generic[CtxType], metaclass=ABCMeta):
    """Abstract base class for parameterized planners.

    Attributes:
        collate_fn_map: Optional mapping from data types to custom collate functions for batching.
            Should be provided in cases the planner needs specific collate functions, usually for
            custom data types. For more information, please refer to, e.g.,
            https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.default_collate.
    """

    collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None

    @abstractmethod
    def forward(
        self,
        obs: Tensor,
        action: Tensor | None = None,
        params: Any = None,
        ctx: CtxType | None = None,
    ) -> tuple[CtxType, Tensor | None, Tensor, Tensor | None]:
        """Computes state and control trajectories from an observation.

        State and control trajectories are computation can be given optional initial
        action, parameters, and internal context.

        Args:
            obs (Tensor): Initial observation input to the planner (e.g., state vector).
            action (Tensor, optional): Optional initial action.
            params: Parameters that define the behavior of the planner, matching
                :attr:`param_space`.
            ctx (CtxType, optional): Optional internal context passed between invocations.

        Returns:
            A tuple ``(ctx, u0, x, u, value)`` where:

            - ctx (CtxType): A context object containing any intermediate values needed
              for backward computation and further invocations.
            - u0 (Tensor, optional): The computed initial control action. Is ``None`` if
              the action is already provided.
            - x (Tensor): The computed sequence of states. Expected shape ``(N+1, *state_dims)``.
            - u (Tensor): The computed sequence of controls. Expected shape ``(N, *control_dims)``.
            - value (Tensor): The cost value of the computed trajectory.
        """
        ...

    @property
    def param_space(self) -> gym.Space:
        """Describes the parameter space of the planner.

        Returns:
            Gymnasium.Space: An object describing the valid space of parameters.
        """
        return self._param_space

    @abstractmethod
    def default_param(self, obs: ndarray | None) -> Any:
        """Provides a default parameter configuration for the planner.

        Args:
            obs (array, optional): Observation input to the planner (e.g., state vector), used to
                condition the default parameters. Can be `None` if not needed.

        Returns:
            array: A default parameter array matching the expected shape of `param` in `forward`.
        """
        ...


class ControllerFromPlanner(ParameterizedController[CtxType], Generic[CtxType]):
    """Wraps a `ParameterizedPlanner` as a `ParameterizedController`.

    This allows using a planner in contexts where a controller is expected, by extracting the first
    action from the planned trajectory.

    Args:
        planner (ParameterizedPlanner): An instance of `ParameterizedPlanner` to be wrapped.

    Attributes:
        planner (ParameterizedPlanner): The underlying `ParameterizedPlanner` instance.
    """

    planner: ParameterizedPlanner[CtxType]

    def __init__(self, planner: ParameterizedPlanner[CtxType]) -> None:
        super().__init__()
        self.planner = planner

    @property
    def collate_fn_map(self) -> dict[type | tuple[type, ...], Callable] | None:
        """Fetches the collate function map from the underlying planner."""
        return self.planner.collate_fn_map

    def forward(
        self,
        obs: Tensor,
        params: Any = None,
        ctx: CtxType | None = None,
    ) -> tuple[CtxType, Tensor]:
        """Computes the first action from the planner's trajectory.

        Args:
            obs (Tensor): Observation input to the controller (e.g., state vector).
            params: Parameters that define the behavior of the controller, matching the wrapped
                planner's :attr:`param_space`.
            ctx (CtxType, optional): Optional internal context passed between invocations.

        Returns:
            A tuple ``(ctx, action)`` where:

            - ctx (CtxType): A context object containing any intermediate values needed
              for backward computation and further invocations.
            - action (Tensor): The computed first action from the planned trajectory.
        """
        return self.planner.forward(obs, params=params, ctx=ctx)[:2]

    @property
    def param_space(self) -> gym.Space:
        """Describes the parameter space of the underlying planner.

        Returns:
            Gymnasium.Space: An object describing the valid space of parameters.
        """
        return self.planner.param_space

    def default_param(self, obs: ndarray | None) -> Any:
        """Provides a default parameter configuration for the underlying planner.

        Args:
            obs (array, optional): Observation input to the controller (e.g., state vector), used to
                condition the default parameters. Can be `None` if not needed.

        Returns:
            array: A default parameter array matching the expected shape of `param` in `forward`.
        """
        return self.planner.default_param(obs)
