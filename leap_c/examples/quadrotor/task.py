from typing import Any, Optional

import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces

from leap_c.examples.quadrotor.env import QuadrotorStop
from leap_c.examples.quadrotor.mpc import QuadrotorMpc
from leap_c.acados.layer import MpcSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task

from leap_c.acados.mpc import MpcInput, MpcParameter


@register_task("quadrotor_terminal")
class QuadrotorStopTask(Task):

    def __init__(self):
        mpc = QuadrotorMpc(N_horizon=4, params_learnable=["terminal_cost"])
        mpc_layer = MpcSolutionModule(mpc)

        self.param_low = mpc.ocp.p_global_values
        self.param_low[14:17] = -1
        self.param_high = mpc.ocp.p_global_values
        self.param_high[14:17] = 1

        super().__init__(mpc_layer)

    @property
    def param_space(self) -> spaces.Box:
        low = self.param_low
        high = self.param_high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def prepare_mpc_input(self, obs: Any, param_nn: Optional[torch.Tensor] = None, ) -> MpcInput:
        mpc_param = MpcParameter(p_global=param_nn)  # type: ignore
        return MpcInput(x0=obs, parameters=mpc_param)

    def create_env(self, train: bool) -> gym.Env:
        return QuadrotorStop()


@register_task("quadrotor_mass")
class QuadrotorStopTask(Task):

    def __init__(self):
        mpc = QuadrotorMpc(N_horizon=4, params_learnable=["m"])
        mpc_layer = MpcSolutionModule(mpc)

        self.param_low = 0.1 * mpc.ocp.p_global_values
        self.param_high = 10. * mpc.ocp.p_global_values

        super().__init__(mpc_layer)

    @property
    def param_space(self) -> spaces.Box:
        low = self.param_low
        high = self.param_high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def prepare_mpc_input(self, obs: Any, param_nn: Optional[torch.Tensor] = None, ) -> MpcInput:
        mpc_param = MpcParameter(p_global=param_nn)  # type: ignore
        return MpcInput(x0=obs, parameters=mpc_param)

    def create_env(self, train: bool) -> gym.Env:
        return QuadrotorStop()
