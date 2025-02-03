from abc import ABC, abstractmethod
from typing import Any, Optional, Callable

import gymnasium as gym
import numpy as np
import torch.nn as nn

from leap_c.mpc import MPC, MPCInput
from leap_c.nn.extractor import Extractor, IdentityExtractor


EnvFactory = Callable[[], gym.Env]
ExtractorFactory = Callable[[gym.Env], Extractor]


class Task(ABC):
    """A task describes a concrete problem to be solved by a learning problem.

    This class serves as a base class for tasks that involve a combination of
    a gymnasium environment and a model predictive control (MPC) planner. It
    provides an interface for preparing neural network inputs and MPC inputs
    based on environment observations and states.

    Attributes:
        mpc (MPC): The Model Predictive Control planner to be used for this task.
        env_factory (EnvFactory): A factory function to create a gymnasium en-
            vironment for the task.
        extractor_factory (ExtractorFactory): A factory function to create an
            extractor for the task.
    """

    def __init__(self, mpc: MPC, env_factory: EnvFactory, extractor_factory: Optional[ExtractorFactory] = None):
        """Initializes the Task with an MPC planner and a gymnasium environment.

        Args:
            mpc (MPC): The Model Predictive Control planner to be used for this task.
            env_factory (EnvFactory): A factory function to create a gymnasium en-
                vironment for the task.
            extractor_factory (Optional[ExtractorFactory]): An optional factory function to
                create an extractor for the task.
        """
        super().__init__()
        self.mpc = mpc
        self.env_factory = env_factory
        self.extractor_factory = IdentityExtractor if extractor_factory is None else extractor_factory

    @abstractmethod
    def prepare_nn_input(self, obs: Any) -> np.ndarray:
        """Prepares the neural network input from an environment observation.

        This method processes an observation from the gymnasium environment
        into a format suitable for a torch module.

        Args:
            obs (Any): The observation from the environment.

        Returns:
            torch.Tensor: The processed input for the neural network.
        """
        ...

    @abstractmethod
    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[np.ndarray] = None,
    ) -> MPCInput:
        """Prepares the MPC input from the state and observation for the MPC class.

        Args:
            obs (Any): The observation from the environment.
            param_nn (Optional[torch.Tensor]): Optional parameters predicted
                by a neural network to assist in planning.

        Returns:
            MPCInput: The processed input for the MPC planner.
        """
        ...
