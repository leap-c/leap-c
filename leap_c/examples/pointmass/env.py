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
        render_mode: str | None = None,
        init_state: np.ndarray | None = None,
        param: PointMassParam = PointMassParam(dt=0.1, m=1.0, c=0.1),
    ):
        super().__init__()

        # Will be added after doing a step.
        self.current_noise = 0.0
        self._np_random = None

        if init_state is None:
            self.s0 = np.array([0.5, 0.5, 0.0, 0.0]).astype(dtype=np.float32)
        else:
            self.s0 = init_state

        self.A = _A_disc(param.m, param.c, param.dt)
        self.B = _B_disc(param.m, param.c, param.dt)

        # For rendering
        if render_mode is not None:
            raise NotImplementedError("Rendering is not implemented yet.")

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        self.action_to_take = action

        u = action

        self.state = self.A @ self.state + self.B @ u

        # Compute the norm of u
        norm_u = np.linalg.norm(u)
        disturbane = self.B @ (self.current_noise * (u / norm_u))
        self.state += disturbane

        self.current_noise = self.next_noise()

        # frame = None
        # if self.render_mode == "human" or self.render_mode == "rgb_array":
        #     frame = self.render()
        # o, r, term, trunc, info = super().step(
        #     action
        # )  # o is the next state as np.ndarray, next parameters as MPCParameter
        # info["frame"] = frame

        # state = o[0].copy()
        # state[-2:] += self.current_noise
        # self.x = state
        # o = (state, o[1])

        # if state not in self.state_space:
        #     r -= 1e2
        #     term = True

        o = self.state
        r = 0
        term = False
        trunc = False
        info = {}

        return o, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:  # type: ignore
        self.state = self.init_state()
        self.state_trajectory = None
        self.action_to_take = None
        self._np_random = np.random.RandomState(seed)
        return self.state

    def next_noise(self) -> float:
        """Return the next noise to be added to the state."""
        if self._np_random is None:
            raise ValueError("First, reset needs to be called with a seed.")
        return self._np_random.uniform(-0.1, +0.1, size=1)

    def init_state(self):
        return self.s0
