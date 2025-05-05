from typing import Any

import numpy as np
from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv as ReacherEnvV5
from gymnasium import spaces
from scipy.spatial.transform import Rotation

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


def compute_reference_position_lemniscate(
    theta: float,
    origin: list = (0.0, 0.1, 0.0),
    orientation: int = (0.0, 0.0, 0.0),
    length: float = 0.4,
    width: float = 0.2,
    direction: int = +1,
) -> np.ndarray:
    s = direction * 2 * np.pi * (1 + theta)

    pos_ref = np.zeros(3)
    pos_ref[0] = (length / 2) * np.cos(s) / (1 + np.sin(s) ** 2)
    pos_ref[1] = (width / 2) * np.sqrt(2) * np.sin(2 * s) / (1 + np.sin(s) ** 2)

    pos_ref = pos_ref @ Rotation.from_euler("xyz", orientation).as_matrix() + np.array(
        origin
    )

    return pos_ref[:2]


def compute_reference_ellipse(
    theta: float,
    origin: list = (0.0, 0.0, 0.0),
    orientation: int = (0.0, 0.0, 0.0),
    length: float = 0.4,
    width: float = 0.2,
) -> np.ndarray:
    pos_ref = np.zeros(3)
    pos_ref[0] = (length / 2) * np.cos(2 * np.pi * (1 + theta))
    pos_ref[1] = (width / 2) * np.sin(2 * np.pi * (1 + theta))

    pos_ref = pos_ref @ Rotation.from_euler("xyz", orientation).as_matrix() + np.array(
        origin
    )

    return pos_ref[:2]


class InvalidReferencePathError(ValueError):
    """Custom exception for invalid reference path."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ReacherEnv(ReacherEnvV5):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        train: bool = False,
        xml_file: str = "reacher.xml",
        frame_skip: int = 2,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 1,
        reference_path: str = "lemniscate",
        delta_path_var: float = 0.025,
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
        if reference_path == "ellipse":
            self.compute_reference_position = compute_reference_ellipse
        elif reference_path == "lemniscate":
            self.compute_reference_position = compute_reference_position_lemniscate
        else:
            msg = f"Invalid reference position: {reference_path}. Choose either \
                'ellipse' or 'lemniscate'."
            raise InvalidReferencePathError(msg)

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
                np.cos(q_min_0),  # cosine of the angle of the second arm
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
                np.cos(q_max_1),  # cosine of the angle of the second arm
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

        # Set the reference position
        self.data.qpos[2:] = self.compute_reference_position(self.path_var)

        # Update path variable
        self.path_var += self.delta_path_var

        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        observation, info = super().reset(seed=seed, options=options)

        self.path_var = 0.0
        return observation, info
