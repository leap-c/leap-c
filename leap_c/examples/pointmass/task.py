from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.mpc import MpcInput, MpcParameter
from leap_c.nn.modules import MpcSolutionModule
from leap_c.nn.extractor import ScalingExtractor
from leap_c.registry import register_task
from leap_c.task import Task


@register_task("point_mass_easy")
class PointMassEasyTask(Task):
    def __init__(self):
        mpc = PointMassMPC(
            learnable_params=[
                # "m",
                # "cx",
                # "cy",
                "q_diag",
                # "r_diag",
                # "q_diag_e",
                "xref",
                "uref",
                # "xref_e",
                # "u_wind",
            ]
        )
        mpc_layer = MpcSolutionModule(mpc)

        super().__init__(mpc_layer)

        self.param_low = 0.5 * mpc.ocp.p_global_values
        self.param_high = 1.5 * mpc.ocp.p_global_values

        # TODO: Handle params that are nominally zero
        for i, p in enumerate(mpc.ocp.p_global_values):
            if p == 0:
                self.param_low[i] = -10.0
                self.param_high[i] = 10.0

    @property
    def param_space(self) -> spaces.Box:
        return spaces.Box(low=self.param_low, high=self.param_high, dtype=np.float32)

    def create_env(self, train: bool) -> gym.Env:
        return PointMassEnv(
            max_time=20.0, train=train, render_mode="rgb_array", difficulty="easy"
        )

    def create_extractor(self, env):
        return ScalingExtractor(env)

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> MpcInput:
        mpc_param = MpcParameter(p_global=param_nn)  # type: ignore
        obs = obs[..., :4]

        return MpcInput(x0=obs, parameters=mpc_param)


@register_task("point_mass_hard")
class PointMassHardTask(PointMassEasyTask):
    def create_env(self, train: bool) -> gym.Env:
        return PointMassEnv(
            max_time=20.0, train=train, render_mode="rgb_array", difficulty="hard"
        )