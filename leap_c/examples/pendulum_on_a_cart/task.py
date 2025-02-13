from typing import Any
from collections import OrderedDict

import gymnasium as gym
import torch
import numpy as np
from leap_c.examples.pendulum_on_a_cart.env import PendulumOnCartSwingupEnv
from leap_c.examples.pendulum_on_a_cart.mpc import PendulumOnCartMPC, PARAMS
from leap_c.nn.modules import MPCSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task

from ...mpc import MPCInput, MPCParameter


PARAMS_SWINGUP = OrderedDict(
    [
        ("M", np.array([1.0])),  # mass of the cart [kg]
        ("m", np.array([0.1])),  # mass of the ball [kg]
        ("g", np.array([9.81])),  # gravity constant [m/s^2]
        ("l", np.array([0.8])),  # length of the rod [m]
        # The quadratic cost matrix is calculated according to L@L.T
        ("L11", np.array([np.sqrt(2e3)])),
        ("L22", np.array([np.sqrt(2e3)])),
        ("L33", np.array([np.sqrt(1e-2)])),
        ("L44", np.array([np.sqrt(1e-2)])),
        ("L55", np.array([np.sqrt(2e-1)])),
        ("Lloweroffdiag", np.array([0.0] * (4 + 3 + 2 + 1))),
        (
            "c1",
            np.array([0.0]),
        ),  # position linear cost, only used for non-LS (!) cost
        (
            "c2",
            np.array([0.0]),
        ),  # theta linear cost, only used for non-LS (!) cost
        (
            "c3",
            np.array([0.0]),
        ),  # v linear cost, only used for non-LS (!) cost
        (
            "c4",
            np.array([0.0]),
        ),  # thetadot linear cost, only used for non-LS (!) cost
        (
            "c5",
            np.array([0.0]),
        ),  # u linear cost, only used for non-LS (!) cost
        (
            "xref1",
            np.array([0.0]),
        ),  # reference position, only used for LS cost
        (
            "xref2",
            np.array([0.0]),
        ),  # reference theta, only used for LS cost
        (
            "xref3",
            np.array([0.0]),
        ),  # reference v, only used for LS cost
        (
            "xref4",
            np.array([0.0]),
        ),  # reference thetadot, only used for LS cost
        (
            "uref",
            np.array([0.0]),
        ),  # reference u, only used for LS cost
    ]
)


@register_task("pendulum_swingup")
class PendulumOnCart(Task):
    """Swing-up task for the pendulum on a cart system.
    The task is to swing up the pendulum from a downward position to the upright position
    (and balance it there)."""

    def __init__(self):
        params = PARAMS_SWINGUP
        learnable_params = ["xref2"]

        mpc = PendulumOnCartMPC(N_horizon=6, T_horizon=0.25, learnable_params=learnable_params, params=params)
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
