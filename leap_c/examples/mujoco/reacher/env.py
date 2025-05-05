from typing import Any

import numpy as np
from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv as ReacherEnvV5
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

        if reference_path == "circle":
            self.compute_reference_position = compute_reference_ellipse
        elif reference_path == "lemniscate":
            self.compute_reference_position = compute_reference_position_lemniscate
        else:
            raise ValueError(
                f"Invalid reference position: {reference_path}. "
                "Choose either 'circle' or 'lemniscate'."
            )

        self.delta_path_var = delta_path_var

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = super().step(action)

        # Random goal position
        self.path_var += self.delta_path_var
        self.data.qpos[2:] = self.compute_reference_position(self.path_var)

        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        observation, info = super().reset(seed=seed, options=options)

        self.path_var = 0.0
        return observation, info
