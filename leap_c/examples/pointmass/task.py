from typing import Any, Optional

from gymnasium import spaces
import torch

from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.mpc import MPCInput, MPCParameter
from leap_c.nn.modules import MPCSolutionModule
from leap_c.task import Task
from leap_c.registry import register_task


@register_task("point_mass")
class PointMassTask(Task):
    def __init__(self):
        mpc = PointMassMPC()
        mpc_layer = MPCSolutionModule(mpc)
        super().__init__(mpc_layer, PointMassEnv)

    @property
    def param_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(4,))

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: Optional[torch.Tensor] = None,
    ) -> MPCInput:

        mpc_param = MPCParameter(p_global=param_nn)  # type: ignore

        return MPCInput(x0=obs, parameters=mpc_param)
