from typing import Any, Optional

from gymnasium import spaces
import numpy as np
import torch

from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.mpc import MPCInput, MPCParameter
from leap_c.nn.modules import MPCSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task


@register_task("point_mass")
class PointMassTask(Task):
    def __init__(self):
        mpc = PointMassMPC(learnable_params=["m", "c", "q_diag", "r_diag", "q_diag_e"])
        mpc_layer = MPCSolutionModule(mpc)

        super().__init__(mpc_layer, PointMassEnv)

        self.param_low = 0.9 * mpc.ocp.p_global_values
        self.param_high = 1.1 * mpc.ocp.p_global_values

    @property
    def param_space(self) -> spaces.Box:
        # low = np.array([0.5, 0.0])
        # high = np.array([2.5, 0.5])
        low = self.param_low
        high = self.param_high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
    ) -> MPCInput:
        mpc_param = MPCParameter(p_global=param_nn)  # type: ignore

        return MPCInput(x0=obs, parameters=mpc_param)
