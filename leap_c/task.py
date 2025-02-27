from functools import cached_property
from typing import Any, Callable, Optional

import gymnasium as gym
import torch
from torch.utils.data._utils.collate import collate

from leap_c.collate import create_collate_fn_map, pytree_tensor_to
from leap_c.mpc import MPCInput
from leap_c.nn.extractor import Extractor, IdentityExtractor
from leap_c.nn.modules import MPCSolutionModule

EnvFactory = Callable[[], gym.Env]


class Task:
    """A task describes a concrete problem to be solved by a learning problem.

    This class serves as a base class for tasks that involve a combination of
    a gymnasium environment and a model predictive control (MPC) planner. It
    provides an interface for preparing neural network inputs in the forms of
    extractors and MPC inputs based on environment observations and states.

    Attributes:
        mpc (MPC): The Model Predictive Control planner to be used for this task.
        env_factory (EnvFactory): A factory function to create a gymnasium en-
            vironment for the task.
    """

    def __init__(
        self,
        mpc: MPCSolutionModule | None,
        env_factory: EnvFactory,
    ):
        """Initializes the Task with an MPC planner and a gymnasium environment.

        Args:
            mpc (MPCSolutionModule): The Model Predictive Control planner to be used
                for this task.
            env_factory (EnvFactory): A factory function to create a gymnasium en-
                vironment for the task.
        """
        super().__init__()
        self.mpc = mpc
        self.env_factory = env_factory
        self.collate_fn_map = create_collate_fn_map()
        self._seed = None

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> MPCInput:
        """Prepares the MPC input from the state and observation for the MPC class.

        Args:
            obs (Any): The observation from the environment.
            param_nn (Optional[torch.Tensor]): Optional parameters predicted
                by a neural network to assist in planning.
            action (Optional[torch.Tensor]): The action taken by the policy
                can be used for MPC as a critic formulations.

        Returns:
            MPCInput: The processed input for the MPC planner.
        """
        raise NotImplementedError

    def prepare_nn_input(self, obs: Any) -> torch.Tensor:
        """Prepares the neural network input from the observation.

        Args:
            obs (Any): The observation from the environment.

        Returns:
            torch.Tensor: The processed input for the neural network.
        """
        return obs

    @property
    def param_space(self) -> gym.spaces.Box | None:
        """Returns the parameter space for the task.

        If the task has no parameters, this method should return None.

        Returns:
            gym.spaces.Box: The parameter space for the task or None if there are
                are no parameters.
        """
        return None

    def create_extractor(self) -> Extractor:
        """Creates an extractor for the task.

        Returns:
            Extractor: The extractor for the task.
        """
        return IdentityExtractor(self.train_env)

    @property
    def seed(self) -> int:
        """Returns the seed for the task.

        Returns:
            int: The seed for the task.
        """
        if self._seed is None:
            raise ValueError("Seed has not been set.")
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        """Sets the seed for the task.

        Args:
            value (int): The seed for the task.
        """
        self._seed = value

    @cached_property
    def train_env(self) -> gym.Env:
        """Returns a gymnasium environment for training.

        Returns:
            gym.Env: The environment for training.
        """
        env = self.env_factory()
        env.reset(seed=self.seed)
        env.observation_space.seed(self.seed)
        env.action_space.seed(self.seed)
        return env

    @cached_property
    def eval_env(self) -> gym.Env:
        """Returns a gymnasium environment for evaluation.

        Returns:
            gym.Env: The environment for evaluation.
        """
        env = self.env_factory()
        env.reset(seed=self.seed)
        env.observation_space.seed(self.seed)
        env.action_space.seed(self.seed)
        return env

    def collate(self, data, device):
        return pytree_tensor_to(
            collate(data, collate_fn_map=self.collate_fn_map),
            device=device,
            tensor_dtype=torch.float32,
        )
