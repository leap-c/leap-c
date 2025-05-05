from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from leap_c.examples.mujoco.reacher.env import ReacherEnv
from leap_c.examples.mujoco.reacher.mpc import ReacherMpc
from leap_c.mpc import MpcInput, MpcParameter
from leap_c.nn.extractor import ScalingExtractor
from leap_c.nn.modules import MpcSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task


@register_task("reacher")
class ReacherTask(Task):
    def __init__(self):
        mpc = ReacherMpc(
            learnable_params=[
                "xy_ee_ref",
                "q_sqrt_diag",
                "r_sqrt_diag",
            ]
        )
        mpc_layer = MpcSolutionModule(mpc)

        super().__init__(mpc_layer)

        self.param_low = 0.5 * mpc.ocp.p_global_values
        self.param_high = 1.5 * mpc.ocp.p_global_values

    @property
    def param_space(self) -> spaces.Box:
        return spaces.Box(low=self.param_low, high=self.param_high, dtype=np.float32)

    def create_env(self, train: bool) -> gym.Env:
        return ReacherEnv(
            train=train,
            render_mode="rgb_array",
            xml_file="reacher.xml",
            reference_path="ellipse",
        )

    def create_extractor(self, env):
        return ScalingExtractor(env)

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> MpcInput:
        mpc_param = MpcParameter(p_global=param_nn)  # type: ignore
        """
            | Num | Observation                                     |
            | --- | ------------------------------------------------|
            | 0   | cosine of the angle of the first arm            |
            | 1   | cosine of the angle of the second arm           |
            | 2   | sine of the angle of the first arm              |
            | 3   | sine of the angle of the second arm             |
            | 4   | x-coordinate of the target                      |
            | 5   | y-coordinate of the target                      |
            | 6   | angular velocity of the first arm               |
            | 7   | angular velocity of the second arm              |
            | 8   | x-value of position_fingertip - position_target |
            | 9   | y-value of position_fingertip - position_target |
        """

        # Assume obs is torch.Tensor. Apply arcsin to elements 2 and 3
        # to get the angles of the first and second arms.
        x0 = obs[..., [2, 3, 6, 7]]
        x0[..., 0] = torch.arcsin(x0[..., 0])
        x0[..., 1] = torch.arcsin(x0[..., 1])

        return MpcInput(x0=x0, parameters=mpc_param)
