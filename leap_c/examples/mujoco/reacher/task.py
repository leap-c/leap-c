from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from leap_c.examples.mujoco.reacher.env import ReacherEnv
from leap_c.examples.mujoco.reacher.mpc import ReacherMpc
from leap_c.examples.mujoco.reacher.util import ReferencePath, PathGeometry
from leap_c.mpc import MpcInput, MpcParameter
from leap_c.nn.extractor import ScalingExtractor
from leap_c.nn.modules import MpcSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task


def prepare_mpc_input_cosq_sinq(
    obs: Any,
    param_nn: torch.Tensor | None = None,
    action: torch.Tensor | None = None,  # noqa: ARG001
) -> MpcInput:
    """
    Prepare the MPC input for the Reacher task.

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
    obs = (
        torch.tensor(obs, dtype=torch.float32)
        if not isinstance(obs, torch.Tensor)
        else obs
    )

    # Target position. We want the NN to set offsets from the target position
    p_global = param_nn
    p_global[..., 0:2] = obs[..., 4:6] + param_nn[..., 0:2]
    mpc_param = MpcParameter(p_global=p_global)

    x0 = obs[..., [0, 1, 2, 3, 6, 7]]

    return MpcInput(x0=x0, parameters=mpc_param)


def prepare_mpc_input_q(
    obs: Any,
    param_nn: torch.Tensor | None = None,
    action: torch.Tensor | None = None,  # noqa: ARG001
    offset_target: bool = True,
) -> MpcInput:
    """
    Prepare the MPC input for the Reacher task.

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
    # TODO: offset_target is used here to enable setting offsets from the target.
    # Can be removed when we allow for incremental parameter updates from the NN.
    if param_nn is not None and offset_target:
        p_global = param_nn.clone()
        adjusted_pos = obs[..., 4:6] + p_global[..., 0:2]
        # Create a new tensor with the adjusted values
        p_global = torch.cat([adjusted_pos, p_global[..., 2:]], dim=-1)
    else:
        p_global = param_nn

    mpc_param = MpcParameter(p_global=p_global)

    # Extract the angles from the observation
    x0 = torch.stack(
        [
            torch.atan2(obs[..., 2], obs[..., 0]),
            torch.atan2(obs[..., 3], obs[..., 1]),
            obs[..., 6],
            obs[..., 7],
        ],
        dim=-1,
    )

    return MpcInput(x0=x0, parameters=mpc_param)


@register_task("reacher")
class ReacherTask(Task):
    def __init__(self) -> None:
        mpc = ReacherMpc(
            learnable_params=[
                "xy_ee_ref",
                "q_sqrt_diag",
                "r_sqrt_diag",
            ],
            params={
                "xy_ee_ref": np.array([0.21, 0.0]),
                "q_sqrt_diag": np.array([np.sqrt(10.0)] * 2),
                "r_sqrt_diag": np.array([np.sqrt(1.0)] * 2),
            },
            state_representation="q",
        )
        mpc_layer = MpcSolutionModule(mpc)

        super().__init__(mpc_layer)

        self.param_low = 0.5 * mpc.ocp.p_global_values
        self.param_high = 1.5 * mpc.ocp.p_global_values

        # Special treatment for the first two parameters (target offset)
        self.param_low[:2] = -0.01
        self.param_high[:2] = +0.01

    @property
    def param_space(self) -> spaces.Box:
        return spaces.Box(low=self.param_low, high=self.param_high, dtype=np.float32)

    def create_env(self, train: bool) -> gym.Env:
        return ReacherEnv(
            train=train,
            render_mode="rgb_array",
            xml_file="reacher.xml",
            reference_path=ReferencePath(
                geometry=PathGeometry(
                    type="ellipse",
                    origin=(0, 0.1, 0.01),
                    orientation=(0.0, 0.0, 0.0),
                    length=0.1,
                    width=0.1,
                    direction=+1,
                ),
                max_reach=0.21,
            ),
        )

    def create_extractor(self, env: gym.Env) -> ScalingExtractor:
        return ScalingExtractor(env)

    def prepare_mpc_input(
        self,
        obs: Any,
        param_nn: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> MpcInput:
        return prepare_mpc_input_q(
            obs=obs,
            param_nn=param_nn,
            action=action,
            offset_target=True,
        )
