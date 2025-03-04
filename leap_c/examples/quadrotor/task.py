from typing import Any
from collections import OrderedDict

import gymnasium as gym
import torch
import numpy as np
from leap_c.examples.quadrotor.env import QuadrotorStop
from leap_c.examples.quadrotor.mpc import QuadrotorMPC
from leap_c.nn.modules import MPCSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task

from ...mpc import MPCInput, MPCParameter


PARAMS_QUADROTOR = OrderedDict(
    [
        ("m", np.array([0.6])),  # mass of the ball [kg]
        ("g", np.array([9.81])),  # gravity constant [m/s^2]
    ]
)


@register_task("quadrotor_stop")
class PendulumOnCart(Task):

    def __init__(self):
        params = PARAMS_QUADROTOR
        learnable_params = ["m"]

        mpc = QuadrotorMPC(learnable_params=learnable_params, given_default_param_dict=params)
        mpc_layer = MPCSolutionModule(mpc)
        super().__init__(mpc_layer, PendulumOnCartSwingupEnv)

        y_ref_stage = np.array(
            [v.item() for k, v in mpc.given_default_param_dict.items() if "xref" in k or "uref" in k]
        )
        y_ref_stage_e = np.array(
            [v.item() for k, v in mpc.given_default_param_dict.items() if "xref" in k]
        )
        self.y_ref = np.tile(y_ref_stage, (5, 1))
        self.y_ref_e = y_ref_stage_e

    @property
    def param_space(self) -> gym.spaces.Box | None:
        return gym.spaces.Box(low=-2.0 * torch.pi, high=2.0 * torch.pi, shape=(1,))

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: torch.Tensor,
    ) -> MPCInput:
        # get batch dim
        batch_size = param_nn.shape[0]

        # prepare y_ref
        param_y_ref = np.tile(self.y_ref, (batch_size, 1, 1))
        param_y_ref[:, :, 1] = param_nn.detach().cpu().numpy()

        # prepare y_ref_e
        param_y_ref_e = np.tile(self.y_ref_e, (batch_size, 1))
        param_y_ref_e[:, 1] = param_nn.detach().cpu().numpy().squeeze()

        mpc_param = MPCParameter(p_global=param_nn, p_yref=param_y_ref, p_yref_e=param_y_ref_e)

        return MPCInput(x0=obs, parameters=mpc_param)
