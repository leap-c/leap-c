from typing import Any, Optional

import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces

from leap_c.examples.quadrotor.env import QuadrotorStop
from leap_c.examples.quadrotor.mpc import QuadrotorMPC
from leap_c.nn.modules import MPCSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task
from .utils import read_from_yaml
from functools import cached_property

from ...mpc import MPCInput, MPCParameter


@register_task("quadrotor_stop")
class QuadrotorStopTask(Task):

    def __init__(self):
        params = read_from_yaml("./examples/quadrotor/model_params.yaml")
        learnable_params = ["m"]

        mpc = QuadrotorMPC(learnable_params=learnable_params)
        mpc_layer = MPCSolutionModule(mpc)

        self.param_low = 0.01 * mpc.ocp.p_global_values
        self.param_high = 10. * mpc.ocp.p_global_values

        # TODO: Handle params that are nominally zero
        for i, p in enumerate(mpc.ocp.p_global_values):
            if p == 0:
                Exception("This should not happen")
                self.param_low[i] = -10.0
                self.param_high[i] = 10.0

        super().__init__(mpc_layer)

    @property
    def param_space(self) -> spaces.Box:
        low = self.param_low
        high = self.param_high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def prepare_mpc_input(self, obs: Any, param_nn: Optional[torch.Tensor] = None, ) -> MPCInput:
        mpc_param = MPCParameter(p_global=param_nn)  # type: ignore
        return MPCInput(x0=obs, parameters=mpc_param)

    def create_env(self, train: bool) -> gym.Env:
        return QuadrotorStop()

