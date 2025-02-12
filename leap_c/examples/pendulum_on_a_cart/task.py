from typing import Any

import gymnasium as gym
import numpy as np
import torch
from leap_c.examples.pendulum_on_a_cart.env import PendulumOnCartSwingupEnv
from leap_c.examples.pendulum_on_a_cart.mpc import PendulumOnCartMPC
from leap_c.nn.modules import MPCSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task
from leap_c.util import tensor_to_numpy

from ...mpc import MPCInput, MPCParameter


@register_task("pendulum_swingup")
class PendulumOnCart(Task):
    """Swing-up task for the pendulum on a cart system.
    The task is to swing up the pendulum from a downward position to the upright position
    (and balance it there)."""

    def __init__(self):
        mpc = PendulumOnCartMPC(N_horizon=5, T_horizon=0.25, learnable_params=["xref2"])
        mpc_layer = MPCSolutionModule(mpc)
        super().__init__(mpc_layer, PendulumOnCartSwingupEnv)

    @property
    def param_space(self) -> gym.spaces.Box | None:
        return gym.spaces.Box(low=-2.0 * torch.pi, high=2.0 * torch.pi, shape=(1,))

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: torch.Tensor,
    ) -> MPCInput:
        numpy_param = tensor_to_numpy(param_nn).astype(np.float64)
        mpc_param = MPCParameter(p_global=numpy_param)

        return MPCInput(x0=obs, parameters=mpc_param)
