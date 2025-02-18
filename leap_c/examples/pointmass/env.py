from typing import Any, Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


@dataclass(kw_only=True)
class WindTunnelParam:
    magnitude: tuple[float, float] = (0, 1.5)
    decay: tuple[float, float] = (0.0, 0.1)
    center: tuple[float, float] = (0, 0)


@dataclass(kw_only=True)
class VortexParam:
    center: tuple[float, float] = (2.5, 0.0)
    magnitude: float = 8.0
    decay: float = 0.5
    direction: float = 1.0


@dataclass(kw_only=True)
class BaseWindParam:
    magnitude: tuple[float, float] = (-1.0, 1.0)


@dataclass(kw_only=True)
class VariationWindParam:
    scale: float = 0.3
    magnitude: tuple[float, float] = (2, 1.5)


@dataclass(kw_only=True)
class RandomWindParam:
    magnitude: float = 0.1
    seed: int = 1000
    dx: int = 10
    dy: int = 6


class VortexWind:
    p: VortexParam

    def __init__(self, param: VortexParam):
        self.p = param

    def __call__(self, x, y):
        dx = x - self.p.center[0]
        dy = y - self.p.center[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        # Tangential velocity decreases with radius
        v_theta = self.p.magnitude * np.exp(-self.p.decay * r)

        return v_theta * np.array(
            [-self.p.direction * np.sin(theta), self.p.direction * np.cos(theta)]
        )


class InverseVortexWind:
    p: VortexParam

    def __init__(self, param: VortexParam):
        self.p = param

    def __call__(self, x, y):
        dx = x - self.p.center[0]
        dy = y - self.p.center[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        # Tangential velocity decreases with radius
        v_theta = self.p.magnitude * np.exp(-self.p.decay * r)

        return v_theta * np.array(
            [self.p.direction * np.sin(theta), self.p.direction * np.cos(theta)]
        )


class BaseWind:
    p: BaseWindParam

    def __init__(self, param: BaseWindParam):
        self.p = param

    def __call__(self, x, y):
        return np.array(self.p.magnitude)


class VariationWind:
    p: VariationWindParam

    # def __init__(self, scale: float, magnitude: tuple[float, float] = (2, 1.5)):
    def __init__(self, param: VariationWindParam):
        self.p = param

    def __call__(self, x, y):
        u = self.p.magnitude[0] * np.sin(self.p.scale * y)
        v = self.p.magnitude[1] * np.cos(self.p.scale * x)
        return np.array([u, v])


class RandomWind:
    p: RandomWindParam

    def __init__(
        self,
        param: RandomWindParam,
    ):
        self.p = param

    def __call__(self, x, y):
        np.random.seed(
            int((x + self.p.dx) * self.p.seed) + int((y + self.p.dy) * self.p.seed)
        )
        u = np.random.randn() * self.p.magnitude
        v = np.random.randn() * self.p.magnitude
        return np.array([u, v])


class WindTunnel:
    p: WindTunnelParam

    def __init__(
        self,
        param: WindTunnelParam,
    ):
        self.p = param

    def __call__(self, x, y):
        return np.array(
            [
                self.p.magnitude[1]
                * np.exp(-self.p.decay[1] * (y - self.p.center[1]) ** 2),
                self.p.magnitude[0]
                * np.exp(-self.p.decay[0] * (x - self.p.center[0]) ** 2),
            ]
        )


class WindField:
    def __init__(self, wind_partial: list[Callable]):
        self.wind_partial = wind_partial

    def __call__(self, x, y):
        return np.sum([wind(x, y) for wind in self.wind_partial], axis=0)


@dataclass
class PointMassParam:
    dt: float
    m: float
    cx: float
    cy: float


@dataclass
class WindParam:
    scale: float = 0.3
    vortex_center: tuple[float, float] = (5.0, 6.0)
    vortex_strength: float = 8.0


def get_wind_velocity(
    x: float | ca.SX,
    y: float | ca.SX,
    scale=0.1,
    vortex_center=(5, 5),
    vortex_strength=1.0,
):
    """
    Compute wind velocity components at a given position.

    Args:
        x (float): X-coordinate of the position
        y (float): Y-coordinate of the position
        scale (float): Wind variation scale
        vortex_center (tuple): Center coordinates of vortex (x, y)
        vortex_strength (float): Strength of the vortex

    Returns:
        tuple: (u, v) wind velocity components at the given position
    """
    if isinstance(x, ca.SX) or isinstance(y, ca.SX):
        u = ca.SX(0.0)
        v = ca.SX(0.0)
        dx = x - vortex_center[0]
        dy = y - vortex_center[1]
        r = ca.sqrt(dx**2 + dy**2)
        theta = ca.arctan2(dy, dx)

        # Tangential velocity decreases with radius
        v_theta = vortex_strength * ca.exp(-0.5 * r)

        # # Add vortex components
        u += -v_theta * ca.sin(theta)
        v += v_theta * ca.cos(theta)

    if True:
        # Base south-west wind
        # u = -1.0
        # v = +1.0

        u = 0.0
        v = 0.0

        # # Add variation
        # u += 2 * np.sin(scale * y)
        # v = 1.5 * np.cos(scale * x)

        # Add vortex
        dx = x - vortex_center[0]
        dy = y - vortex_center[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        # Tangential velocity decreases with radius
        v_theta = vortex_strength * np.exp(-0.5 * r)

        # # Add vortex components
        u += -v_theta * np.sin(theta)
        v += v_theta * np.cos(theta)

        # Add random turbulence component
        # Note: Using a fixed seed for reproducibility
        # np.random.seed(int((x + 10) * 1000) + int((y + 6) * 1000))
        # u += np.random.randn() * 0.5
        # v += np.random.randn() * 0.5
    else:
        u, v = 0.0, 0.0

    return u, v


def map_wind_field(
    wind_field: WindField,
    xlim=[0, 10],
    ylim=[0, 10],
    nx=100,
    ny=100,
):
    """
    Test the wind velocity function using a grid of positions.
    Returns the complete wind field arrays U, V for comparison.

    Args:
        nx, ny (int): Grid dimensions
        scale (float): Wind variation scale
        vortex_center (tuple): Center coordinates of vortex (x, y)
        vortex_strength (float): Strength of the vortex

    Returns:
        tuple: (X, Y, U, V) arrays containing the grid coordinates and wind components
    """
    # Create grid
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(x, y)

    # Initialize velocity arrays
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    # Compute velocities for each grid point
    for i in range(ny):
        for j in range(nx):
            U[i, j], V[i, j] = wind_field(X[i, j], Y[i, j])

    # Compute and print some statistics
    wind_mag = np.sqrt(U**2 + V**2)

    return X, Y, U, V, wind_mag


def _A_disc(
    m: float | ca.SX,
    cx: float | ca.SX,
    cy: float | ca.SX,
    dt: float | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [m, cx, cy, dt]):
        return ca.vertcat(
            ca.horzcat(1, 0, dt, 0),
            ca.horzcat(0, 1, 0, dt),
            ca.horzcat(0, 0, ca.exp(-cx * dt / m), 0),
            ca.horzcat(0, 0, 0, ca.exp(-cy * dt / m)),
        )
    else:
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, np.exp(-cx * dt / m), 0],
                [0, 0, 0, np.exp(-cy * dt / m)],
            ]
        )


def _B_disc(
    m: float | ca.SX,
    cx: float | ca.SX,
    cy: float | ca.SX,
    dt: float | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [m, cx, cy, dt]):
        return ca.vertcat(
            ca.horzcat(0, 0),
            ca.horzcat(0, 0),
            ca.horzcat((m / cx) * (1 - ca.exp(-cx * dt / m)), 0),
            ca.horzcat(0, (m / cy) * (1 - ca.exp(-cy * dt / m))),
        )
    else:
        return np.array(
            [
                [0, 0],
                [0, 0],
                [(m / cx) * (1 - np.exp(-cx * dt / m)), 0],
                [0, (m / cy) * (1 - np.exp(-cy * dt / m))],
            ]
        )


def _A_cont(
    m: float | ca.SX, cx: float | ca.SX, cy: float | ca.SX, dt: float | ca.SX
) -> np.ndarray | ca.SX:
    if isinstance(m, float):
        return np.array(
            [
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0],
                [0, 0, -(cx / m), 0],
                [0, 0, 0, -(cy / m)],
            ]
        )
    else:
        return ca.vertcat(
            ca.horzcat(0, 0, 1.0, 0),
            ca.horzcat(0, 0, 0, 1.0),
            ca.horzcat(0, 0, -(cx / m), 0),
            ca.horzcat(0, 0, 0, -(cy / m)),
        )


def _B_cont(
    m: float | ca.SX, cx: float | ca.SX, cy: float | ca.SX, dt: float | ca.SX
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


class PointMassEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        dt: float = 2 / 20,
        max_time: float = 10.0,
        render_mode: str | None = None,
        param: PointMassParam = PointMassParam(dt=0.1, m=1.0, cx=0.1, cy=0.1),
        Fmax: float = 10.0,
        observation_space: spaces.Box = spaces.Box(
            low=np.array([0.0, -8.0, -50.0, -50.0]),
            high=np.array([6.0, +8.0, 50.0, 50.0]),
            dtype=np.float64,
        ),
        init_state_dist: dict[str, np.ndarray] = {
            "low": np.array([5.0, -5.0, 0.0, 0.0]),
            "high": np.array([5.0, 5.0, 0.0, 0.0]),
        },
        wind_field=WindField(
            [
                # BaseWind(param=BaseWindParam(magnitude=(-1.0, 1.0))),
                # RandomWind(param=RandomWindParam()),
                # VariationWind(param=VariationWindParam()),
                # VortexWind(param=VortexParam(center=(6, 5.0))),
                WindTunnel(
                    param=WindTunnelParam(
                        center=(0, 0), magnitude=(0, 3.0), decay=(0.0, 0.1)
                    )
                ),
            ]
        ),
    ):
        super().__init__()

        # self.wind_param = wind_param
        self.wind_field = wind_field

        self.init_state_dist = init_state_dist

        self.input_noise_dist = {
            "low": -1.0,
            "high": 1.0,
        }

        self.observation_space = observation_space

        self.action_space = spaces.Box(
            low=np.array([-Fmax, -Fmax]),
            high=np.array([Fmax, Fmax]),
            dtype=np.float32,
        )

        # Will be added after doing a step.
        self.input_noise = 0.0
        self._np_random = None

        if dt is not None:
            param.dt = dt

        self.dt = dt
        self.max_time = max_time

        self.A = _A_disc(param.m, param.cx, param.cy, param.dt)
        self.B = _B_disc(param.m, param.cx, param.cy, param.dt)

        self.terminal_radius = 0.5

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

        self.u = u

        self.u_wind = self.wind_field(self.state[0], self.state[1])

        self.u_dist = 0.0 * self.input_noise * u

        self.state = self.A @ self.state + self.B @ (self.u + self.u_wind + self.u_dist)

        self.input_noise = self._get_input_noise()

        o = self._current_observation()
        r = self._calculate_reward()

        if self.state not in self.observation_space:
            r -= 50

        term = self._is_done()

        info = {}

        self.time += self.dt

        trunc = self.time > self.max_time

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

        self.trajectory = []
        plt.close("all")
        self.canvas = None
        self.line = None

        self._set_canvas()

        return self.state, {}

    def _current_observation(self):
        return self.state

    def _init_state(self):
        return np.random.uniform(
            low=self.init_state_dist["low"], high=self.init_state_dist["high"]
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

        distance = np.linalg.norm(self.state[:2])

        # velocity = np.linalg.norm(self.state[2:])
        # power = np.dot(self.u, self.state[2:])

        # power = np.linalg.norm(self.u)
        power = np.dot(self.u, self.state[2:])

        if len(self.trajectory) > 1:
            path_increment = self.state[:2] - self.trajectory[-1][:2]
        else:
            path_increment = np.array([0.0, 0.0])
        work = np.dot(path_increment, self.u)

        # print(
        #     f"Position: {self.state[:2]}, Velocity: {self.state[2:]}, Distance: {distance}, Force: {self.u}, Power: {power}, Work: {work}"
        # )

        reward = -self.dt * (distance + 5 * power)

        distance = np.linalg.norm(self.state[:2])
        velocity = np.linalg.norm(self.state[2:])

        close_to_zero = distance < self.terminal_radius and velocity < 0.5

        if close_to_zero:
            reward += 50
        # reward = -distance - 10 * work
        return reward

    def _is_done(self):
        # Episode is done if the position is very close to (0,0) at low velocity

        distance = np.linalg.norm(self.state[:2])
        velocity = np.linalg.norm(self.state[2:])

        close_to_zero = distance < self.terminal_radius and velocity < 0.5

        outside_bounds = self.state not in self.observation_space

        time_exceeded = self.time > self.max_time

        done = close_to_zero or outside_bounds or time_exceeded

        # if done:
        #     print(
        #         f"Close to zero: {close_to_zero}, Outside bounds: {outside_bounds}, Time exceeded: {time_exceeded}"
        #     )

        return done

    def _set_canvas(self):
        fig = plt.figure(figsize=(10, 10))
        plt.xlabel("x")
        plt.ylabel("y")

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

        # Draw velocity field
        X, Y, U, V, wind_mag = map_wind_field(
            wind_field=self.wind_field,
            xlim=(self.observation_space.low[0], self.observation_space.high[0]),
            ylim=(self.observation_space.low[1], self.observation_space.high[1]),
            nx=30,
            ny=30,
        )

        contour = plt.contourf(X, Y, wind_mag, levels=20, cmap="viridis")
        plt.colorbar(contour, label="Wind Speed")

        # Plot wind vectors
        quiver = plt.quiver(
            X, Y, U, V, color="black", alpha=0.8, scale=10, scale_units="xy"
        )

        self.quiver = quiver

        # Draw constraint boundary
        # rect = plt.Rectangle(
        #     (self.observation_space.low[0], self.observation_space.low[1]),
        #     width=(self.observation_space.high[0] - self.observation_space.low[0]),
        #     height=(self.observation_space.high[1] - self.observation_space.low[1]),
        #     fill=False,  # Set to True if you want a filled rectangle
        #     color="k",
        #     linewidth=2,
        #     linestyle="--",
        # )
        # self.canvas.figure.get_axes()[0].add_patch(rect)

        # Set axis limits tight
        # plt.tight_layout()

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
            alpha=1.0,
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

        xwind = 0
        ywind = 0
        uv = self.wind_field(xwind, ywind)

        self.wind_arrow = plt.arrow(
            xwind,
            ywind,
            uv[0] / quiver.scale,
            uv[1] / quiver.scale,
            head_width=0.1,
            head_length=0.1,
            fc="b",
            ec="b",
            alpha=0.75,
        )

        # Draw radius with self.terminal_radius at (0,0)
        self.circle = plt.Circle(
            xy=(0, 0),
            radius=self.terminal_radius,
            fill=False,
            color="k",
            linewidth=2,
            linestyle="--",
        )

        self.canvas.figure.get_axes()[0].add_patch(self.circle)

        # Draw radius for initial state distribution
        self.rectangle_init = plt.Rectangle(
            (self.init_state_dist["low"][0], self.init_state_dist["low"][1]),
            width=(self.init_state_dist["high"][0] - self.init_state_dist["low"][0]),
            height=(self.init_state_dist["high"][1] - self.init_state_dist["low"][1]),
            fill=False,  # Set to True if you want a filled rectangle
            color="k",
            linewidth=2,
            linestyle="-",
            alpha=0.5,
        )

        self.canvas.figure.get_axes()[0].add_patch(self.rectangle_init)

        # Set the axis limits with some padding
        self.canvas.figure.get_axes()[0].set_xlim(
            self.observation_space.low[0], self.observation_space.high[0]
        )
        self.canvas.figure.get_axes()[0].set_ylim(
            self.observation_space.low[1], self.observation_space.high[1]
        )

        # Set axis equal
        # plt.axis("equal")
        # plt.show()
        # exit(0)

    def render(self):
        self.line.set_xdata([x[0] for x in self.trajectory])
        self.line.set_ydata([x[1] for x in self.trajectory])

        self.point.set_xdata([self.state[0]])
        self.point.set_ydata([self.state[1]])

        self.input_arrow.set_data(
            x=self.state[0],
            y=self.state[1],
            dx=self.u[0] / self.quiver.scale,
            dy=self.u[1] / self.quiver.scale,
        )

        self.disturbance_arrow.set_data(
            x=self.state[0],
            y=self.state[1],
            dx=self.u_dist[0] / self.quiver.scale,
            dy=self.u_dist[1] / self.quiver.scale,
        )

        self.wind_arrow.set_data(
            x=self.state[0],
            y=self.state[1],
            dx=self.u_wind[0] / self.quiver.scale,
            dy=self.u_wind[1] / self.quiver.scale,
        )

        self.canvas.draw()

        # Convert the plot to an RGB string
        s, (width, height) = self.canvas.print_to_buffer()

        # Convert the RGB string to a NumPy array
        return np.frombuffer(s, np.uint8).reshape((height, width, 4))[:, :, :3]
