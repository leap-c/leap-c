from abc import ABC, abstractmethod
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from leap_c.mpc import MPC, MPCBatchedState, MPCInput, MPCSingleState


class Task(ABC):
    """A task describes a concrete problem to be solved by a learning problem.

    This class serves as a base class for tasks that involve a combination of
    a gymnasium environment and a model predictive control (MPC) planner. It
    provides an interface for preparing neural network inputs and MPC inputs
    based on environment observations and states.

    Attributes:
        mpc (MPC): The Model Predictive Control planner to be used for this task.
        env (gym.Env): The gymnasium environment for the task.
    """

    def __init__(self, mpc: MPC, env: gym.Env):
        """Initializes the Task with an MPC planner and a gymnasium environment.

        Args:
            mpc (MPC): The Model Predictive Control planner to be used for this task.
            env (gym.Env): The gymnasium environment for the task.
        """
        super().__init__()
        self.mpc = mpc
        self.env = env

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
