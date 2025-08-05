from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

import casadi as ca
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import FancyArrowPatch

from leap_c.utils.latexify import latex_plot_context


class Circle:
    def __init__(self, pos: np.ndarray, radius: float):
        self.pos = pos
        self.radius = radius

    def __contains__(self, item):
        # Check only position (first 2 elements)
        if len(item) >= 2:
            return np.linalg.norm(item[:2] - self.pos) <= self.radius
        return False  # Cannot check if item is not position-like

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        theta = rng.uniform(0, 2 * np.pi)
        r = self.radius * np.sqrt(rng.uniform(0, 1))
        x = self.pos[0] + r * np.cos(theta)
        y = self.pos[1] + r * np.sin(theta)
        return np.array([x, y])


class WindField(ABC):
    @abstractmethod
    def __call__(self, pos: np.ndarray) -> np.ndarray: ...

    def plot_XY(
        self, xlim: tuple[float, float], ylim: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        nx = ny = 20
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def plot_wind_field(
        self,
        ax: Axes,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        scale: float = 80.0,
    ):
        # Get x, y
        X, Y = self.plot_XY(xlim, ylim)

        # Initialize velocity arrays
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # Compute velocities for each grid point
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = np.array([X[i, j], Y[i, j]])
                U[i, j], V[i, j] = self(pos)

        # Compute and print some statistics
        wind_mag = np.sqrt(U**2 + V**2)

        # Filter out zero wind vectors for better visualization
        mask = wind_mag > 0

        # Plot the wind Field
        ax.quiver(
            X[mask],
            Y[mask],
            U[mask],
            V[mask],
            wind_mag[mask],
            scale=scale,
            scale_units="xy",
            width=0.0035,
            pivot="mid",
        )


class WindParcour(WindField):
    def __init__(self, magnitude: float = 10.0, difficulty: str = "easy"):
        self.magnitude = magnitude
        if difficulty == "easy":
            self.boxes = [
                [np.array([0.5, 0.2]), np.array([1.5, 1.0])],
                [np.array([2.5, 0.0]), np.array([3.5, 0.8])],
            ]
        elif difficulty == "hard":
            self.boxes = [
                [np.array([0.5, 0.1]), np.array([1.5, 1.0])],
                [np.array([2.5, 0.0]), np.array([3.5, 0.9])],
            ]
        else:
            raise ValueError(f"Unknown difficulty level: {difficulty}")

    def plot_XY(
        self, xlim: tuple[float, float], ylim: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        parts_X = []
        parts_Y = []

        for box in self.boxes:
            # create intersection of the box with the xlim and set_ylim
            xlim_box = (max(xlim[0], box[0][0]), min(xlim[1], box[1][0]))
            ylim_box = (max(ylim[0], box[0][1]), min(ylim[1], box[1][1]))
            delta = 0.1999999
            num_x = int((xlim_box[1] - xlim_box[0]) // delta) - 1
            num_y = int((ylim_box[1] - ylim_box[0]) // delta) - 1
            mid_x = (xlim_box[0] + xlim_box[1]) / 2
            mid_y = (ylim_box[0] + ylim_box[1]) / 2
            left_x = mid_x - num_x * delta / 2
            left_y = mid_y - num_y * delta / 2
            x = np.linspace(left_x, left_x + num_x * delta, num_x)
            y = np.linspace(left_y, left_y + num_y * delta, num_y)

            parts_X.append(x)
            parts_Y.append(y)

        parts_X = np.concatenate(parts_X)
        parts_Y = np.concatenate(parts_Y)

        X, Y = np.meshgrid(parts_X, parts_Y)
        return X, Y

    def plot_wind_field(
        self,
        ax: Axes,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        scale: float = 80.0,
    ):
        # plot rectangles for each boxes
        for box in self.boxes:
            rect = plt.Rectangle(
                box[0],
                box[1][0] - box[0][0],
                box[1][1] - box[0][1],
                color="gray",
                alpha=0.1,
            )
            ax.add_patch(rect)

        # plot wind field
        super().plot_wind_field(ax, xlim, ylim, scale=scale)

    def __call__(self, pos: np.ndarray) -> np.ndarray:
        for box in self.boxes:
            if np.all(box[0] <= pos) and np.all(pos <= box[1]):
                return np.array([-self.magnitude, 0.0])
        return np.array([0.0, 0.0])


@dataclass
class PointMassParam:
    dt: float  # time discretization
    m: float  # mass
    cx: float  # damping coefficient in x direction
    cy: float


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
        )  # type: ignore

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
        )  # type: ignore

    return np.array(
        [
            [0, 0],
            [0, 0],
            [(m / cx) * (1 - np.exp(-cx * dt / m)), 0],
            [0, (m / cy) * (1 - np.exp(-cy * dt / m))],
        ]
    )


class PointMassEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        param: PointMassParam = PointMassParam(dt=0.1, m=1.0, cx=15, cy=15),
        Fmax: float = 10,
        max_time: float = 10.0,
        render_mode: str | None = None,
        difficulty: str = "easy",
    ):
        # gymnasium setup
        max_v = 20
        max_wind_force = 10.5
        self.state_low = np.array([0.0, 0.0, -max_v, -max_v], dtype=np.float32)
        self.state_high = np.array([4, 1.0, max_v, max_v], dtype=np.float32)
        self.wind_low = np.array([-max_wind_force, -max_wind_force], dtype=np.float32)
        self.wind_high = np.array([max_wind_force, max_wind_force], dtype=np.float32)
        self.obs_high = np.concatenate([self.state_high, self.wind_high])
        self.obs_low = np.concatenate([self.state_low, self.wind_low])
        self.action_low = np.array([-Fmax, -Fmax], dtype=np.float32)
        self.action_high = np.array([Fmax, Fmax], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)
        self.render_mode = render_mode

        # env logic
        self.max_time = max_time
        self.dt = param.dt
        self.Fmax = Fmax
        self.A = _A_disc(param.m, param.cx, param.cy, param.dt)
        self.B = _B_disc(param.m, param.cx, param.cy, param.dt)
        self.start = Circle(pos=np.array([0.25, 0.8]), radius=0.15)
        self.goal = Circle(pos=np.array([3.75, 0.2]), radius=0.15)
        self.wind_field = WindParcour(magnitude=max_wind_force, difficulty=difficulty)

        # env state
        self.state: np.ndarray | None = None
        self.action: np.ndarray | None = None
        self.time: float = 0.0

        # plotting attributes (initialize to None)
        self.fig: plt.Figure | None = None  # type: ignore
        self.ax: plt.Axes | None = None  # type: ignore
        self.trajectory_plot: plt.Line2D | None = None  # type: ignore
        self.agent_plot: plt.Line2D | None = None  # type: ignore
        self.action_arrow_patch: FancyArrowPatch | None = None
        self.trajectory: List[np.ndarray] = []

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        if self.state is None:
            raise ValueError("Environment must be reset before stepping.")

        # transition
        action = np.clip(action, self.action_low, self.action_high)  # type: ignore
        force_wind = self.wind_field(self.state[:2])
        self.state = self.A @ self.state + self.B @ (action + force_wind)  # type: ignore
        self.action = action  # Store the action taken
        self.time += self.dt

        # observation
        self.trajectory.append(self.state.copy())  # type: ignore

        # termination and truncation
        out_of_bounds = (self.state_high < self.state).any() or (
            self.state_low > self.state
        ).any()
        reached_goal = self.state[:2] in self.goal  # type: ignore
        term = out_of_bounds or reached_goal
        trunc = self.time >= self.max_time
        if term or trunc:
            info = {"task": {"violation": bool(out_of_bounds), "success": reached_goal}}
        else:
            info = {}

        # reward
        dist_to_goal_x = np.abs(self.state[0] - self.goal.pos[0])  # type: ignore
        r_dist = 1 - dist_to_goal_x / (self.state_high[0] - self.state_low[0])
        # dist_max = np.linalg.norm(self.goal.pos - self.state_low[:2])
        # r_dist = 1.0 - np.linalg.norm(self.state[:2] - self.goal.pos) / dist_max
        r_goal = 60 * (1.0 - 0.5 * self.time / self.max_time) if reached_goal else 0.0
        r = 0.1 * r_dist + r_goal

        return self._observation(), float(r), bool(term), bool(trunc), info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        super().reset(seed=seed)
        self.time = 0.0
        self.state = self._init_state(options=options)
        self.action = np.zeros(self.action_space.shape, dtype=np.float32)  # type: ignore
        self.trajectory = [self.state.copy()]

        # Close existing figure if resetting during run
        if self.render_mode == "human" and self.fig is not None:
            self._close_fig()

        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        ode_state = self.state.copy().astype(np.float32)  # type: ignore
        wind_field = self.wind_field(self.state[:2]).astype(np.float32)  # type: ignore
        return np.concatenate([ode_state, wind_field])

    def _init_state(self, num_tries: int = 100, options=None) -> np.ndarray:
        if num_tries <= 0:
            raise ValueError("Could not find a valid initial state.")

        if options is not None and "mode" in options and options["mode"] == "train":
            low = np.array([0.1, 0.1, 0.0, 0.0])
            high = np.array([3.9, 0.9, 0.0, 0.0])
            state = self.np_random.uniform(low=low, high=high)
        else:
            pos = self.start.sample(self.np_random)
            state = np.array([*pos, 0.0, 0.0])

        # check if the state is in the wind field
        if (self.wind_field(state[:2]) != 0).any():
            return self._init_state(num_tries - 1)

        return state

    def render(self) -> np.ndarray | None:
        with latex_plot_context():
            return self._render()

    def _render(self) -> np.ndarray | None:
        if self.render_mode is None:
            gym.logger.warn("Cannot render environment without a render_mode set.")
            return None
        if self.state is None or self.action is None:
            gym.logger.warn("Cannot render environment before reset() is called.")
            return None

        if self.fig is None:
            if self.render_mode == "human":
                plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 4))

            # Set limits based on observation space position with padding
            self.ax.set_xlim(self.state_low[0], self.state_high[0])
            self.ax.set_ylim(self.state_low[1], self.state_high[1])
            self.ax.set_yticks(np.arange(0, 1.1, 0.5))

            self.ax.set_aspect(
                "equal", adjustable="box"
            )  # Ensure aspect ratio is visually correct
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")

            self.ax.text(
                self.start.pos[0] + 0.02,
                self.start.pos[1],
                r"$\odot$",
                fontsize=60,
                color="black",
                horizontalalignment="center",
                verticalalignment="center_baseline",
                zorder=3,
                label="Start ($\odot$)",
            )
            self.ax.plot(
                [], [], "ko", marker=r"$\odot$", markersize=10, label="Start", zorder=3
            )
            self.ax.text(
                self.goal.pos[0] + 0.02,  # x-coordinate from self.goal.pos
                self.goal.pos[1],  # y-coordinate from self.goal.pos
                r"$\otimes$",  # The LaTeX symbol (otimes) in math mode
                fontsize=60,  # Adjust size for visibility
                color="black",  # Choose a color (e.g., green, lime)
                horizontalalignment="center",  # Center the symbol horizontally
                verticalalignment="center_baseline",  # Center the symbol vertically
                zorder=3,  # Ensure it's drawn prominently
                label="Goal ($\otimes$)",  # Optional: Update label for legend
            )
            self.ax.plot(
                [], [], "ko", marker=r"$\otimes$", markersize=10, label="Goal", zorder=3
            )

            if self.wind_field:
                self.wind_field.plot_wind_field(
                    self.ax,
                    xlim=(self.state_low[0], self.state_high[0]),
                    ylim=(self.state_low[1], self.state_high[1]),
                )

            (self.trajectory_plot,) = self.ax.plot(
                [],
                [],
                "b-",
                alpha=0.5,
                label="Trajectory",
                zorder=1,
                lw=2.5,
            )  # Blue line
            (self.agent_plot,) = self.ax.plot(
                [], [], "ro", markersize=8, label="Agent", zorder=3
            )  # Red circle
            # Action arrow will be created/removed dynamically

            # add goal to legend below plot with three columns
            self.ax.legend(
                loc="upper center",
                fontsize=10,
                frameon=True,
                ncol=4,
                bbox_to_anchor=(0.5, -0.25),
            )

        # Update trajectory
        traj_x = [s[0] for s in self.trajectory]
        traj_y = [s[1] for s in self.trajectory]
        self.trajectory_plot.set_data(traj_x, traj_y)  # type: ignore

        # Update agent position
        self.agent_plot.set_data([self.state[0]], [self.state[1]])  # type: ignore

        # Update action arrow (remove old, add new)
        if self.action_arrow_patch is not None:
            self.action_arrow_patch.remove()
            self.action_arrow_patch = None

        # Calculate arrow properties (scale for visibility)
        # Might need adjustment due to different axis scales, but let's try the same first
        arrow_scale = 0.03  # Keep same scale relative to action magnitude
        dx = self.action[0] * arrow_scale
        dy = self.action[1] * arrow_scale

        # Only draw arrow if it has significant length
        if np.linalg.norm([dx, dy]) > 1e-4:
            # Create a new arrow patch
            self.action_arrow_patch = FancyArrowPatch(
                (self.state[0], self.state[1]),  # start point
                (self.state[0] + dx, self.state[1] + dy),  # end point
                color="darkorange",
                mutation_scale=15,
                alpha=0.9,
                zorder=2,  # Above trajectory, below agent
            )
            self.ax.add_patch(self.action_arrow_patch)  # type: ignore

        self.fig.tight_layout()  # Adjust layout to fit all elements
        if self.render_mode == "human":
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
            return None
        elif self.render_mode == "rgb_array":
            canvas = FigureCanvas(self.fig)
            canvas.draw()  # Draw the canvas
            self.fig.subplots_adjust(bottom=0.25)
            # Get RGB data
            image_shape = canvas.get_width_height()[::-1] + (4,)
            buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            img = buf.reshape(image_shape)[:, :, :3]  # Get RGB, discard alpha

            # Important for rgb_array mode to avoid artifacts on next render call
            if self.action_arrow_patch:
                self.action_arrow_patch.remove()
                self.action_arrow_patch = None
            return img
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _close_fig(self):
        """Helper to close the Matplotlib figure."""
        if self.fig is not None:
            if self.render_mode == "human":
                plt.ioff()  # Turn off interactive mode
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.trajectory_plot = None
            self.agent_plot = None
            self.action_arrow_patch = None

    def close(self):
        """Close the rendering window."""
        self._close_fig()


if __name__ == "__main__":
    env = PointMassEnv(
        render_mode="human",
        max_time=15.0,
        train=False,
    )  # Longer time
    obs, info = env.reset(seed=44)  # Changed seed slightly

    terminated = False
    truncated = False
    total_reward = 0
    env.render()

    for i in range(300):  # Increase steps for longer visualization
        action = env.action_space.sample()

        goal_dir = env.goal.pos - obs[:2]
        goal_dir_norm = np.linalg.norm(goal_dir)
        if goal_dir_norm > 1e-3:
            proportional_force = (goal_dir / goal_dir_norm) * env.Fmax
        else:
            proportional_force = np.zeros(2)

        action = proportional_force

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()

        if terminated or truncated:
            print(f"Episode finished after {i + 1} timesteps.")
            print(f"Termination: {terminated}, Truncation: {truncated}")
            print(f"Final state (pos): {obs[:2]}")
            print(f"Goal position: {env.goal.pos}")
            print(f"Distance to goal: {np.linalg.norm(obs[:2] - env.goal.pos):.3f}")
            print(f"Total reward: {total_reward:.2f}")
            if env.render_mode == "human":
                plt.pause(5.0)  # Increased pause to see final state
            break  # Stop after one episode for this example

    # Close the environment rendering window
    env.close()
    print("Environment closed.")
