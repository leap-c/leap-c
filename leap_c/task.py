from abc import ABC, abstractmethod
from typing import Any, Optional

import gymnasium as gym
import torch

from leap_c.mpc import MPC, MPCBatchedState, MPCInput


class Task(ABC):
    """A task is a wrapper around an gym environment and an MPC planner."""
    def __init__(self, mpc: MPC, env: gym.Env):
        """A task is a wrapper around an gym environment and an MPC planner.

        Args:
            mpc: The learnable MPC planner.
            env: The gym environment.
        """
        super().__init__()
        self.mpc = mpc
        self.env = env

    @abstractmethod
    def prepare_nn_input(self, obs: Any) -> torch.Tensor:
        """Prepare the neural network input from the observation.

        Args:
            obs: The observation from the environment.
        """
        ...

    @abstractmethod
    def prepare_mpc_input(
        self, mpc_state: MPCBatchedState, obs: Any, param_nn: Optional[torch.Tensor]
    ) -> MPCInput:
        """Prepare the MPC input from the observation.

        Args:
            mpc_state: The current state of the MPC.
            obs: The observation from the environment.
            param_nn: The neural network output.
        """
        ...

