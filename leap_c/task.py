from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import gymnasium as gym
import torch
from torch.utils.data._utils.collate import collate

from leap_c.collate import create_collate_fn_map, pytree_tensor_to
from leap_c.env_wrapper import ActionStatsWrapper
from leap_c.mpc import MpcInput
from leap_c.nn.extractor import Extractor, IdentityExtractor
from leap_c.nn.modules import MpcSolutionModule

EnvFactory = Callable[[], gym.Env]


class Task(ABC):
    """A task describes a problem to be solved by a trainer.

    This class serves as a base class for tasks that involve a combination of
    a gymnasium environment and a model predictive control (MPC) planner. It
    provides an interface for preparing neural network inputs in the forms of
    extractors and MPC inputs based on environment observations and states.

    Attributes:
        mpc (MPC): The Model Predictive Control planner to be used for this task.
        collate_fn_map (dict[type, Callable]): A dictionary mapping types to collate
            functions. This is used to collate data into a tensor. If None, the default
            collate function map is used, which is sufficient for most tasks and contains
            some extensions to the default PyTorch collate function to handle acados
            objects.
    """

    def __init__(
        self,
        mpc: MpcSolutionModule | None,
        collate_fn_map: dict[type, Callable] | None = None,
    ):
        """Initializes the Task with an MPC planner and a gymnasium environment.

        Args:
            mpc (MPCSolutionModule): The Model Predictive Control planner to be used
                for this task.
            collate_fn_map (dict[type, Callable]): A dictionary mapping types to collate
                functions. If None, the default collate function map is used, which is
                sufficient for most tasks.
        """
        super().__init__()
        self.mpc = mpc
        self.collate_fn_map = (
            create_collate_fn_map() if collate_fn_map is None else collate_fn_map
        )

    @abstractmethod
    def create_env(self, train: bool) -> gym.Env:
        """Creates a gymnasium environment for the task.

        Args:
            train (bool): Whether the environment is for training or evaluation.

        Returns:
            gym.Env: The environment for the task.
        """
        ...

    @abstractmethod
    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> MpcInput:
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
        ...

    def create_extractor(self, env: gym.Env) -> Extractor:
        """Creates an extractor for the task.

        This could be used to extract features from images or other complex
        observations.

        Args:
            env (gym.Env): The environment for the task.

        Returns:
            Extractor: The extractor for the task.
        """
        return IdentityExtractor(env)

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

    def create_train_env(self, seed: int = 0, record_action: bool = False) -> gym.Env:
        """Returns a gymnasium environment for training.

        Args:
            seed: The seed for the environment.
            record_action: Whether to add the actions taken by the agent as stats.

        Returns:
            gym.Env: The environment for training.
        """
        env = self.create_env(train=True)

        if record_action:
            env = ActionStatsWrapper(env)

        env.reset(seed=seed)
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env

    def create_eval_env(self, seed: int = 1, record_action: bool = False) -> gym.Env:
        """Returns a gymnasium environment for evaluation.

        Args:
            seed: The seed for the environment.
            record_action: Whether to add the actions taken by the agent as stats.
        """
        env = self.create_env(train=False)

        if record_action:
            env = ActionStatsWrapper(env)

        env.reset(seed=seed)
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env

    def collate(self, data, device):
        """Collates the data into a tensor.

        This is the central functionality of leap_c to build batches. In most cases
        you are not required to override this method.

        Args:
            data: The data to be collated.
            device: The device to move the tensor to.
        """
        return pytree_tensor_to(
            collate(data, collate_fn_map=self.collate_fn_map),  # type: ignore
            device=device,
            tensor_dtype=torch.float32,
        )
