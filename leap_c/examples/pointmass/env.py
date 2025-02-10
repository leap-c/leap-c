from typing import Any
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import casadi as ca


def _A_disc(
    m: float | ca.SX, c: float | ca.SX, dt: float | ca.SX
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [m, c, dt]):
        a = ca.exp(-c * dt / m)
        return ca.vertcat(
            ca.horzcat(1, 0, dt, 0),
            ca.horzcat(0, 1, 0, dt),
            ca.horzcat(0, 0, a, 0),
            ca.horzcat(0, 0, 0, a),
        )
    else:
        a = np.exp(-c * dt / m)
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, a, 0],
                [0, 0, 0, a],
            ]
        )


def _B_disc(
    m: float | ca.SX, c: float | ca.SX, dt: float | ca.SX
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [m, c, dt]):
        b = (m / c) * (1 - ca.exp(-c * dt / m))
        return ca.vertcat(
            ca.horzcat(0, 0),
            ca.horzcat(0, 0),
            ca.horzcat(b, 0),
            ca.horzcat(0, b),
        )
    else:
        b = (m / c) * (1 - np.exp(-c * dt / m))
        return np.array(
            [
                [0, 0],
                [0, 0],
                [b, 0],
                [0, b],
            ]
        )


def _A_cont(
    m: float | ca.SX, c: float | ca.SX, dt: float | ca.SX
) -> np.ndarray | ca.SX:
    if isinstance(m, float):
        return np.array(
            [
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0],
                [0, 0, -(c / m), 0],
                [0, 0, 0, -(c / m)],
            ]
        )
    else:
        return ca.vertcat(
            ca.horzcat(0, 0, 1.0, 0),
            ca.horzcat(0, 0, 0, 1.0),
            ca.horzcat(0, 0, -(c / m), 0),
            ca.horzcat(0, 0, 0, -(c / m)),
        )


def _B_cont(
    m: float | ca.SX, c: float | ca.SX, dt: float | ca.SX
) -> np.ndarray | ca.SX:
    if isinstance(m, float):
        return np.array([[0, 0], [0, 0], [1.0 / m, 0], [0, 1.0 / m]])
    else:
        return ca.vertcat(
            ca.horzcat(0, 0),
            ca.horzcat(0, 0),
            ca.horzcat(1.0 / m, 0),
            ca.horzcat(0, 1.0 / m),
        )


@dataclass
class PointMassParam:
    dt: float
    m: float
    c: float


class PointMassEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        dt: float | None = None,
        max_time: float = 10.0,
        render_mode: str | None = None,
        param: PointMassParam = PointMassParam(dt=0.1, m=1.0, c=0.1),
    ):
        super().__init__()

        self.init_state_dist = {
            "mean": np.array([1.5, 1.5, 0.0, 0.0]),
            "cov": np.diag([0.1, 0.1, 0.05, 0.05]),
        }

        self.input_noise_dist = {
            "low": -0.1,
            "high": 0.1,
        }

        self.state_space = spaces.Box(
            low=np.array([-10.0, -np.inf, -5.0, -5.0]),
            high=np.array([10.0, 10.0, 5.0, 5.0]),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # Will be added after doing a step.
        self.input_noise = 0.0
        self._np_random = None

        if dt is not None:
            param.dt = dt

        self.A = _A_disc(param.m, param.c, param.dt)
        self.B = _B_disc(param.m, param.c, param.dt)

        # For rendering
        if render_mode is not None:
            raise NotImplementedError("Rendering is not implemented yet.")

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        self.action_to_take = action

        u = action

        self.state = self.A @ self.state + self.B @ u

        # Add an input disturbance that acts in the direction of u
        norm_u = np.linalg.norm(u)
        disturbane = self.B @ (self.input_noise * (u / norm_u))
        self.state += disturbane

        self.input_noise = self._get_input_noise()

        o = self._current_observation()
        r = self._calculate_reward()

        if self.state not in self.state_space:
            r -= 1e2

        term = self._is_done()

        trunc = False
        info = {}

        return o, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:  # type: ignore
        self._np_random = np.random.RandomState(seed)
        self.state_trajectory = None
        self.action_to_take = None
        self.state = self._init_state()
        return self.state

    def _current_observation(self):
        return self.state

    def _init_state(self):
        return self._np_random.multivariate_normal(
            mean=self.init_state_dist["mean"],
            cov=self.init_state_dist["cov"],
        )

    def _get_input_noise(self) -> float:
        """Return the next noise to be added to the state."""
        if self._np_random is None:
            raise ValueError("First, reset needs to be called with a seed.")
        return self._np_random.uniform(
            low=self.input_noise_dist["low"],
            high=self.input_noise_dist["high"],
            size=1,
        )

    def _calculate_reward(self):
        # Reward is higher the closer the position is to (0,0) and the lower the velocity
        distance = np.linalg.norm(self.state)
        velocity = np.linalg.norm(self.state[2:])

        reward = -distance - 0.1 * velocity
        return reward

    def _is_done(self):
        # Episode is done if the position is very close to (0,0) at low velocity

        distance = np.linalg.norm(self.state[:2])
        velocity = np.linalg.norm(self.state[2:])

        close_to_zero = distance < 0.1 and velocity < 0.1

        outside_bounds = self.state not in self.state_space

        return close_to_zero or outside_bounds
