from typing import Any

import gymnasium as gym
import numpy as np
from leap_c.task import Task
from leap_c.mpc import MPCInput
from leap_c.registry import register_task


@register_task("half_cheetah")
class HalfCheetahTask(Task):
    def __init__(self):
        env_factory = lambda: gym.make("HalfCheetah-v5")
        super().__init__(None, env_factory)  # type: ignore

    def prepare_nn_input(self, obs: Any) -> np.ndarray:
        return obs

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: None | np.ndarray = None,
    ) -> MPCInput:
        raise NotImplementedError
