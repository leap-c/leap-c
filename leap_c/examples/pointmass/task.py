from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from functools import cached_property

from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.mpc import MPCInput, MPCParameter
from leap_c.nn.modules import MPCSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task


@register_task("point_mass")
class PointMassTask(Task):
    def __init__(self):
        mpc = PointMassMPC(
            learnable_params=[
                # "m",
                # "cx",
                # "cy",
                "q_diag",
                "r_diag",
                "q_diag_e",
                # "xref",
                # "uref",
                # "xref_e",
            ]
        )
        mpc_layer = MPCSolutionModule(mpc)

        super().__init__(mpc_layer, PointMassEnv)

        self.param_low = 0.5 * mpc.ocp.p_global_values
        self.param_high = 1.5 * mpc.ocp.p_global_values

        # TODO: Handle params that are nominally zero
        for i, p in enumerate(mpc.ocp.p_global_values):
            if p == 0:
                self.param_low[i] = -10.0
                self.param_high[i] = 10.0

    @property
    def param_space(self) -> spaces.Box:
        # low = np.array([0.5, 0.0])
        # high = np.array([2.5, 0.5])
        low = self.param_low
        high = self.param_high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @cached_property
    def train_env(self) -> gym.Env:
        env = PointMassEnv(
            init_state_dist={
                "low": np.array([10.0, -5.0, 0.0, 0.0]),
                "high": np.array([10.0, 5.0, 0.0, 0.0]),
            },
        )
        env.reset(seed=self.seed)
        return env

    @cached_property
    def eval_env(self) -> gym.Env:
        env = PointMassEnv(
            init_state_dist={
                "low": np.array([10.0, -1.0, 0.0, 0.0]),
                "high": np.array([10.0, 1.0, 0.0, 0.0]),
            },
        )
        env.reset(seed=self.seed)
        return env

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
    ) -> MPCInput:
        mpc_param = MPCParameter(p_global=param_nn)  # type: ignore

        return MPCInput(x0=obs, parameters=mpc_param)
