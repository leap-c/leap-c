from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from leap_c.examples.chain.env import ChainEnv
from leap_c.examples.chain.mpc import ChainMpc
from leap_c.examples.chain.utils import Ellipsoid
from leap_c.mpc import MpcInput, MpcParameter
from leap_c.nn.modules import MpcSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task


@register_task("chain")
class ChainTask(Task):
    def __init__(self):
        n_mass = 3
        params = {}
        # rest length of spring
        params["L"] = np.repeat([0.033, 0.033, 0.033], n_mass - 1)

        # spring constant
        params["D"] = np.repeat([1.0, 1.0, 1.0], n_mass - 1)

        # damping constant
        params["C"] = np.repeat([0.1, 0.1, 0.1], n_mass - 1)

        # mass of the balls
        params["m"] = np.repeat([0.033], n_mass - 1)

        # disturbance on intermediate balls
        params["w"] = np.repeat([0.0, 0.0, 0.0], n_mass - 2)

        # Weight on state
        params["q_sqrt_diag"] = np.ones(3 * (n_mass - 1) + 3 * (n_mass - 2))

        # Weight on control inputs
        params["r_sqrt_diag"] = 1e-1 * np.ones(3)

        fix_point = np.zeros(3)
        ellipsoid = Ellipsoid(center=fix_point, radii=0.033 * (n_mass - 1) * np.ones(3))

        pos_last_mass_ref = ellipsoid.spherical_to_cartesian(phi=0.75 * np.pi, theta=np.pi / 2)

        mpc = ChainMpc(
            params=params,
            learnable_params=[
                "q_sqrt_diag",
                "r_sqrt_diag",
            ],
            fix_point=fix_point,
            pos_last_mass_ref=pos_last_mass_ref,
            n_mass=n_mass,
        )
        mpc_layer = MpcSolutionModule(mpc)

        super().__init__(mpc_layer)

        self.param_low = 0.5 * mpc.ocp.p_global_values
        self.param_high = 1.5 * mpc.ocp.p_global_values

        self.n_mass = n_mass
        self.fix_point = fix_point
        self.pos_last_mass_ref = pos_last_mass_ref
        self.ellipsoid = ellipsoid

    @property
    def param_space(self) -> spaces.Box:
        low = self.param_low
        high = self.param_high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def create_env(self, train: bool) -> gym.Env:
        if train:
            phi_range = (0.5 * np.pi, 1.5 * np.pi)
            theta_range = (-0.1 * np.pi, 0.1 * np.pi)
        else:
            phi_range = (0.9 * np.pi, 1.1 * np.pi)
            theta_range = (-0.1 * np.pi, 0.1 * np.pi)

        return ChainEnv(
            max_time=10.0,
            phi_range=phi_range,
            theta_range=theta_range,
            fix_point=self.fix_point,
            n_mass=self.n_mass,
            pos_last_ref=self.pos_last_mass_ref,
        )

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
    ) -> MpcInput:
        mpc_param = MpcParameter(p_global=param_nn)  # type: ignore

        return MpcInput(x0=obs, parameters=mpc_param)
