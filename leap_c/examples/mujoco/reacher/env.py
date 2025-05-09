from typing import Any

import numpy as np
from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv as ReacherEnvV5
from gymnasium import spaces
from scipy.spatial.transform import Rotation
import mujoco
from leap_c.examples.mujoco.reacher.util import ReferencePath, PathGeometry

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class InvalidReferencePathError(ValueError):
    """Custom exception for invalid reference path."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ReacherEnv(ReacherEnvV5):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        reference_path: ReferencePath,
        train: bool = False,
        xml_file: str = "reacher.xml",
        frame_skip: int = 1,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 1,
        delta_path_var: float = 0.01,
        **kwargs,
    ):
        # gymnasium setup
        super().__init__(
            xml_file=xml_file,
            frame_skip=frame_skip,
            default_camera_config=default_camera_config,
            reward_dist_weight=reward_dist_weight,
            reward_control_weight=reward_control_weight,
            **kwargs,
        )
        self.reference_path = reference_path

        # TODO: Add out-of-reach error
        # msg = f"Invalid reference position: {reference_path}. Choose either \
        #     'ellipse' or 'lemniscate'."
        # raise InvalidReferencePathError(msg)

        # Set reasonable bounds for the observation space
        q_min_0 = self.model.jnt_range[0, 0]
        q_max_0 = self.model.jnt_range[0, 1]
        q_min_1 = self.model.jnt_range[1, 0]
        q_max_1 = self.model.jnt_range[1, 1]
        target_x_min = self.model.jnt_range[2, 0]
        target_x_max = self.model.jnt_range[2, 1]
        target_y_min = self.model.jnt_range[3, 0]
        target_y_max = self.model.jnt_range[3, 1]

        # NOTE: In the following, we set angular velocity limits. The simulation
        # actually does not clip velocity, but uses the damping of the motor
        # actuators to limit the velocity.
        # NOTE: First joint has infinite position range

        low = np.array(
            [
                -1.0,  # cosine of the angle of the first arm
                np.cos(q_min_1),  # cosine of the angle of the second arm
                -1.0,  # sine of the angle of the first arm
                -1.0,  # sine of the angle of the second arm
                target_x_min,  # x-coordinate of the target
                target_y_min,  # y-coordinate of the target
                -8.0,  # angular velocity of the first arm
                -8.0,  # angular velocity of the second arm
                2 * target_x_min,  # x-value of position_fingertip - position_target
                2 * target_y_min,  # y-value of position_fingertip - position_target
            ],
            dtype=np.float32,
        )

        high = np.array(
            [
                1.0,  # cosine of the angle of the first arm
                1.0,  # cosine of the angle of the second arm
                1.0,  # sine of the angle of the first arm
                1.0,  # sine of the angle of the second arm
                target_x_max,  # x-coordinate of the target
                target_y_max,  # y-coordinate of the target
                8.0,  # angular velocity of the first arm
                8.0,  # angular velocity of the second arm
                2 * target_x_max,  # x-value of position_fingertip - position_target
                2 * target_y_max,  # y-value of position_fingertip - position_target
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.delta_path_var = delta_path_var

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = super().step(action)

        # Update path variable
        self.path_var += self.delta_path_var
        self.goal = self.reference_path(self.path_var)
        self.data.qpos[2:] = self.goal

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        self.goal = np.random.uniform(low=0.05, high=0.15, size=2)
        qpos[-2:] = self.goal

        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        observation, info = super().reset(seed=seed, options=options)

        self.path_var = self.np_random.uniform(low=0.0, high=1.0, size=(1,))

        return observation, info
