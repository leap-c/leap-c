from collections import OrderedDict

import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.cartpole.mpc import CartPoleMPC
from leap_c.ocp.acados.layer import MpcSolutionModule
from leap_c.ocp.acados.mpc import (
    MpcCtx,
    MpcInput,
    MpcParameter,
)

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


class CartPoleController(ParameterizedController):
    def __init__(self):
        super().__init__()
        self.collate_state_fn = None

        params = PARAMS_SWINGUP
        learnable_params = ["xref2"]

        mpc = CartPoleMPC(
            N_horizon=5,
            T_horizon=0.25,
            learnable_params=learnable_params,
            params=params,
        )
        self.mpc_layer = MpcSolutionModule(mpc)

    def param_space(self) -> gym.spaces.Box | None:
        return gym.spaces.Box(low=-2.0 * torch.pi, high=2.0 * torch.pi, shape=(1,))

    def default_param(self) -> np.ndarray:
        # TODO: where to use this default parameter? and how should it be initialized?
        raise NotImplementedError

    def jacobian_action_param(self, ctx: MpcCtx) -> np.ndarray:
        return ctx.output.du0_dp_global

    def forward(self, obs, param, ctx: MpcCtx | None = None) -> tuple[MpcCtx, np.ndarray]:
        if ctx is None:
            ctx = MpcCtx()

        mpc_param = MpcParameter(p_global=param)
        mpc_input = MpcInput(x0=obs, parameters=mpc_param)

        mpc_output, mpc_state, mpc_stats = self.mpc_layer(mpc_input, ctx.state)

        ctx = MpcCtx(
            output=mpc_output,
            state=mpc_state,
            log=mpc_stats,
            status=mpc_output.status if hasattr(mpc_output, "status") else None,
        )

        return ctx, mpc_output.u0
