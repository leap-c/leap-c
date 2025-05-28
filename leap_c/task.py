from abc import ABC, abstractmethod
from typing import Callable, Optional

import gymnasium as gym
from gymnasium.wrappers import OrderEnforcing, RecordEpisodeStatistics

from leap_c.controller import ParameterizedController
from leap_c.torch.nn.extractor import Extractor, IdentityExtractor


class Task(ABC):
    """A task describes a problem to be solved by a trainer.

    Attributes:
        collate_fn_map (dict[type, Callable]): A dictionary mapping types to collate
            functions. This is used to collate data into a tensor. If None, the default
            collate function map is used, which is sufficient for most tasks and contains
            some extensions to the default PyTorch collate function to handle acados
            objects.
    """

    @abstractmethod
    def _create_env(self, train: bool) -> gym.Env:
        """Creates a gymnasium environment for the task.

        Args:
            train (bool): Whether the environment is for training or evaluation.

        Returns:
            gym.Env: The environment for the task.
        """
        pass

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

    def create_eval_env(self, seed: int = 1) -> gym.Env:
        """Returns a gymnasium environment for evaluation.

        Args:
            seed: The seed for the environment.
        """
        env = self._create_env(train=False)
        env = RecordEpisodeStatistics(env, buffer_length=1)
        env = OrderEnforcing(env)

        env.reset(seed=seed)
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env


class ReinforcementLearningMixin(ABC):
    """A reinforcement learning mixin provides a gymnasium environment for training and evaluation."""

    def create_train_env(self, seed: int = 0) -> gym.Env:
        """Creates a gymnasium environment for training.

        Args:
            seed: The seed for the environment.
        """
        env = self._create_env(train=True)
        env = RecordEpisodeStatistics(env, buffer_length=1)
        env = OrderEnforcing(env)

        env.reset(seed=seed)
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env


class HierarchicalControllerMixin(ABC):
    """A hierarchical controller mixin provides a parameterized controller
    to plan the actions (control inputs) for the task.
    """

    @abstractmethod
    def create_parameterized_controller(
        self, collate_state_fn: Optional[Callable] = None
    ) -> ParameterizedController:
        """Creates a parameterized controller for the hierarchical controller.

        Args:
            collate_state_fn (Optional[Callable]): A function to collate the state.

        Returns:
            ParameterizedController: The parameterized controller.
        """
        pass
