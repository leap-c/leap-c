from typing import Any
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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
        dt: float = 2 / 20,
        max_time: float = 10.0,
        render_mode: str | None = None,
        param: PointMassParam = PointMassParam(dt=0.1, m=2.0, c=0.4),
    ):
        super().__init__()

        self.init_state_dist = {
            "mean": np.array([1.5, 1.5, 0.0, 0.0]),
            "cov": np.diag([0.1, 0.1, 0.05, 0.05]),
        }

        self.input_noise_dist = {
            "low": -1.0,
            "high": 1.0,
        }

        self.observation_space = spaces.Box(
            low=np.array([-2.0, 0.0, -5.0, -5.0]),
            high=np.array([2.0, 2.0, 5.0, 5.0]),
            dtype=np.float64,
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

        self.dt = dt
        self.max_time = max_time

        self.A = _A_disc(param.m, param.c, param.dt)
        self.B = _B_disc(param.m, param.c, param.dt)

        self.trajectory = []

        self._set_canvas()

        # For rendering
        if render_mode is not None:
            raise NotImplementedError("Rendering is not implemented yet.")

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        self.action_to_take = action

        u = action
        # TODO(Jasper): Quickfix
        if u.ndim > 1:
            u = u.squeeze()

        self.state = self.A @ self.state + self.B @ u

        # Add an input disturbance that acts in the direction of u
        self.disturbance = self.input_noise * u
        self.state += self.B @ self.disturbance

        self.u = u

        self.input_noise = self._get_input_noise()

        o = self._current_observation()
        r = self._calculate_reward()

        if self.state not in self.observation_space:
            r -= 1e2

        term = self._is_done()

        trunc = False
        info = {}

        self.time += self.dt

        self.trajectory.append(o)

        return o, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:  # type: ignore
        self._np_random = np.random.RandomState(seed)
        self.state_trajectory = None
        self.action_to_take = None
        self.state = self._init_state()
        self.time = 0.0
        return self.state, {}

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

        outside_bounds = self.state not in self.observation_space

        time_exceeded = self.time > self.max_time

        return close_to_zero or outside_bounds or time_exceeded

    def _set_canvas(self):
        # Create a figure
        fig = plt.figure()
        plt.ylabel("y")
        plt.xlabel("x")
        plt.xlim(-5.1, 5.1)
        plt.ylim(-5.1, 5.1)
        plt.axis("equal")
        plt.grid()

        self.canvas = FigureCanvas(fig)

        # Draw trajectory
        (self.line,) = plt.plot(
            self.canvas.figure.get_axes()[0].get_xlim(),
            self.canvas.figure.get_axes()[0].get_xlim(),
            "k",
            alpha=0.5,
        )

        # Draw position
        (self.point,) = plt.plot([0], [0], "ko")

        # Draw arrow for action
        self.input_arrow = plt.arrow(
            0,
            0,
            0,
            0,
            head_width=0.1,
            head_length=0.1,
            fc="g",
            ec="g",
            alpha=0.75,
        )

        # Draw arrow for action
        self.disturbance_arrow = plt.arrow(
            0,
            0,
            0,
            0,
            head_width=0.1,
            head_length=0.1,
            fc="r",
            ec="r",
            alpha=0.75,
        )

        # Draw constraint boundary
        rect = plt.Rectangle(
            (self.observation_space.low[0], self.observation_space.low[1]),
            width=(self.observation_space.high[0] - self.observation_space.low[0]),
            height=(self.observation_space.high[1] - self.observation_space.low[1]),
            fill=False,  # Set to True if you want a filled rectangle
            color="k",
            linewidth=2,
            linestyle="--",
        )
        self.canvas.figure.get_axes()[0].add_patch(rect)

        # Set the axis limits with some padding
        self.canvas.figure.get_axes()[0].set_xlim(
            self.observation_space.low[0] - 1, self.observation_space.high[0] + 1
        )
        self.canvas.figure.get_axes()[0].set_ylim(
            self.observation_space.low[1] - 1, self.observation_space.high[1] + 1
        )

    def render(self):
        self.line.set_xdata([x[0] for x in self.trajectory])
        self.line.set_ydata([x[1] for x in self.trajectory])

        self.point.set_xdata([self.state[0]])
        self.point.set_ydata([self.state[1]])

        self.input_arrow.set_data(
            x=self.state[0],
            y=self.state[1],
            dx=self.u[0],
            dy=self.u[1],
        )

        self.disturbance_arrow.set_data(
            x=self.state[0],
            y=self.state[1],
            dx=self.disturbance[0],
            dy=self.disturbance[1],
        )
        self.canvas.draw()

        # Convert the plot to an RGB string
        s, (width, height) = self.canvas.print_to_buffer()

        # Convert the RGB string to a NumPy array
        return np.frombuffer(s, np.uint8).reshape((height, width, 4))[:, :, :3]
