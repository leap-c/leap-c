from collections import OrderedDict
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from leap_c.examples.pendulum_on_a_cart.env import (
    PendulumOnCartBalanceEnv,
    PendulumOnCartSwingupEnv,
)
from leap_c.examples.pendulum_on_a_cart.mpc import PendulumOnCartMPC
from leap_c.nn.modules import MpcSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task

from ...mpc import MpcInput, MpcParameter

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
class PendulumOnCartSwingup(Task):
    """Swing-up task for the pendulum on a cart system.
    The task is to swing up the pendulum from a downward position to the upright position
    (and balance it there)."""

    def __init__(self):
        params = PARAMS_SWINGUP
        learnable_params = ["xref2"]

        mpc = PendulumOnCartMPC(
            N_horizon=6,
            T_horizon=0.25,
            learnable_params=learnable_params,
            params=params,  # type: ignore
        )
        mpc_layer = MpcSolutionModule(mpc)
        super().__init__(mpc_layer)

    def create_env(self, train: bool) -> gym.Env:
        return PendulumOnCartSwingupEnv()

    @property
    def param_space(self) -> gym.spaces.Box | None:
        return gym.spaces.Box(low=-2.0 * torch.pi, high=2.0 * torch.pi, shape=(1,))

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> MpcInput:
        if param_nn is None:
            raise ValueError("Parameter tensor is required for MPC task.")

        mpc_param = MpcParameter(p_global=param_nn)  # type: ignore

        return MpcInput(x0=obs, parameters=mpc_param)


@register_task("pendulum_balance")
class PendulumOnCartBalance(PendulumOnCartSwingup):
    """The same as PendulumOnCartSwingup, but the starting position of the pendulum is upright, making the task a balancing task."""

    def create_env(self, train: bool) -> gym.Env:
        return PendulumOnCartBalanceEnv()


@register_task("pendulum_swingup_long_horizon")
class PendulumOnCartSwingupLong(Task):
    """Swing-up task for the pendulum on a cart system,
    like PendulumOnCartSwingup, but with a much longer horizon.
    """

    def __init__(self):
        params = PARAMS_SWINGUP
        learnable_params = ["xref2"]

        mpc = PendulumOnCartMPC(
            N_horizon=20,
            T_horizon=1,
            learnable_params=learnable_params,
            params=params,  # type: ignore
        )
        mpc_layer = MpcSolutionModule(mpc)
        super().__init__(mpc_layer)
