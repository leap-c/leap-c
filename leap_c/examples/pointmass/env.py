from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from leap_c.examples.utils.matplotlib_env import MatplotlibRenderEnv

DifficultyLevel = Literal["easy", "hard"]


@dataclass(kw_only=True)
class PointMassEnvConfig:
    """Configuration for the PointMass environment."""

    dt: float = 0.1  # time discretization [s]
    m: float = 1.0  # mass [kg]
    cx: float = 15  # damping coefficient in x direction [kg/s]
    cy: float = 15  # damping coefficient in y direction [kg/s]
    Fmax: float = 10.0  # maximum force [N]
    max_time: float = 10.0  # maximum time for an episode [s]
    difficulty: DifficultyLevel = "hard"  # difficulty level for the wind field
    max_wind_force: float = 10.5  # maximum wind force [N]
    max_v: float = 20.0  # maximum velocity [m/s] (only used for obs space limits)


class PointMassEnv(MatplotlibRenderEnv):
    """A 2D point mass navigation environment with wind disturbances.

    Observation Space:
    ------------------
    The observation is a `ndarray` with shape `(6,)` and dtype `np.float32` representing:
    | Num | Observation         | Min        | Max        |
    |-----|---------------------|------------|------------|
    | 0   | x-position          | 0.0        | 4.0        |
    | 1   | y-position          | 0.0        | 1.0        |
    | 2   | x-velocity          | -max_v     | max_v      |
    | 3   | y-velocity          | -max_v     | max_v      |
    | 4   | wind force (x)      | -max_wind  | max_wind   |
    | 5   | wind force (y)      | -max_wind  | max_wind   |

    Action Space:
    -------------
    The action is a `ndarray` with shape `(2,)` and dtype `np.float32`
    representing the force applied in x and y directions.
    Each component is bounded by [-Fmax, Fmax] which will be enforced by
    clipping the input action in each step.

    Reward:
    -------
    The agent receives a small reward for progressing towards the goal in the x-direction
    and a large bonus for reaching the goal quickly.
    - Distance reward: r_dist = 1 - abs(x - goal_x) / (x_max - x_min)
    - Goal reward: r_goal = 60 * (1 - 0.5 * t / max_time) if goal reached, else 0
    - Total reward: r = 0.1 * r_dist + r_goal

    Termination:
    ------------
    The episode terminates if:
    - The agent leaves the allowed state space (out of bounds, also including velocity limits)
    - The agent reaches the goal region

    Truncation:
    -----------
    The episode is truncated if the maximum time is exceeded.

    Info:
    -----
    The info dictionary contains:
    - "task": {"violation": bool, "success": bool}
      - violation: True if out of bounds
      - success: True if goal reached

    Attributes:
        cfg: Configuration object for the environment.
        state_low: Lower bounds for the Pointmass position and velocity.
        state_high: Upper bounds for the Pointmass position and velocity.
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        A: State transition matrix for the point mass dynamics.
        B: Control input matrix for the point mass dynamics.
        start: The starting position will be sampled in this Circle.
        goal: The goal is reached if the pointmass enters this Circle.
        wind_field: WindField object representing the wind disturbances.
        state: Current state of the environment (position and velocity).
        action: Last action taken (used in rendering).
        time: Elapsed time in the current episode.
        trajectory: List of states visited during the episode (used in rendering).
        trajectory_plot: object representing the trajectory in the rendering.
        agent_plot: object representing the agent in the rendering.
        action_arrow_patch: object representing the action arrow in the rendering.
    """

    cfg: PointMassEnvConfig
    state_low: np.ndarray
    state_high: np.ndarray
    observation_space: spaces.Box
    action_space: spaces.Box
    A: np.ndarray
    B: np.ndarray
    start: "Circle"
    goal: "Circle"
    wind_field: "WindField"
    state: np.ndarray | None
    action: np.ndarray | None
    time: float
    trajectory_plot: Line2D | None
    agent_plot: Line2D | None
    action_arrow_patch: FancyArrowPatch | None

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        cfg: PointMassEnvConfig | None = None,
    ):
        """Initialize the PointMass environment.

        Args:
            render_mode: The mode to render with. Supported modes are: human, rgb_array, None.
            cfg: Configuration for the environment. If None, default configuration is used.
        """
        super().__init__(render_mode=render_mode)

        self.cfg = PointMassEnvConfig() if cfg is None else cfg

        # gymnasium setup
        self.state_low = np.array([0.0, 0.0, -self.cfg.max_v, -self.cfg.max_v], dtype=np.float32)
        self.state_high = np.array([4, 1.0, self.cfg.max_v, self.cfg.max_v], dtype=np.float32)
        wind_low = np.array([-self.cfg.max_wind_force, -self.cfg.max_wind_force], dtype=np.float32)
        wind_high = np.array([self.cfg.max_wind_force, self.cfg.max_wind_force], dtype=np.float32)
        obs_high = np.concatenate([self.state_high, wind_high])
        obs_low = np.concatenate([self.state_low, wind_low])
        self.action_low = np.array([-self.cfg.Fmax, -self.cfg.Fmax], dtype=np.float32)
        self.action_high = np.array([self.cfg.Fmax, self.cfg.Fmax], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)

        # env logic
        self.A, self.B = define_transition_matrices(
            m=self.cfg.m, cx=self.cfg.cx, cy=self.cfg.cy, dt=self.cfg.dt
        )
        self.start = Circle(pos=np.array([0.25, 0.8]), radius=0.15)
        self.goal = Circle(pos=np.array([3.75, 0.2]), radius=0.15)
        self.wind_field = WindParcour(
            magnitude=self.cfg.max_wind_force, difficulty=self.cfg.difficulty
        )

        # env state
        self.state = None
        self.action = None
        self.time = 0.0

        # plotting attributes (initialize to None)
        self.trajectory_plot = None
        self.agent_plot = None
        self.action_arrow_patch = None
        self.trajectory: list[np.ndarray] = []

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        if self.state is None:
            raise ValueError("Environment must be reset before stepping.")

        # transition
        action = np.clip(action, self.action_low, self.action_high)  # type: ignore
        force_wind = self.wind_field(self.state[:2])
        self.state = self.A @ self.state + self.B @ (action + force_wind)  # type: ignore
        self.action = action  # Store the action taken
        self.time += self.cfg.dt

        # observation
        self.trajectory.append(self.state.copy())  # type: ignore

        # termination and truncation
        out_of_bounds = (self.state_high < self.state).any() or (self.state_low > self.state).any()
        reached_goal = self.state[:2] in self.goal  # type: ignore
        term = out_of_bounds or reached_goal
        trunc = self.time >= self.cfg.max_time
        if term or trunc:
            info = {"task": {"violation": bool(out_of_bounds), "success": reached_goal}}
        else:
            info = {}

        # reward
        dist_to_goal_x = np.abs(self.state[0] - self.goal.pos[0])
        r_dist = 1 - dist_to_goal_x / (self.state_high[0] - self.state_low[0])
        r_goal = 60 * (1.0 - 0.5 * self.time / self.cfg.max_time) if reached_goal else 0.0
        r = 0.1 * r_dist + r_goal

        return self._observation(), float(r), bool(term), bool(trunc), info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        super().reset(seed=seed)
        self.time = 0.0
        self.state = self._init_state(options=options)
        self.action = np.zeros(self.action_space.shape, dtype=np.float32)  # type: ignore
        self.trajectory = [self.state.copy()]

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
            state = np.array([pos[0], pos[1], 0.0, 0.0])

        # check if the state is in the wind field
        if (self.wind_field(state[:2]) != 0).any():
            return self._init_state(num_tries - 1)

        return state

    def _render_setup(self):
        """Initializes the Matplotlib figure and axes for rendering."""
        if self.render_mode == "human":
            plt.ion()

        self._fig, self._ax = plt.subplots(figsize=(10, 4))

        # Set limits based on observation space position with padding
        self._ax.set_xlim(self.state_low[0], self.state_high[0])
        self._ax.set_ylim(self.state_low[1], self.state_high[1])
        self._ax.set_yticks(np.arange(0, 1.1, 0.5))

        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")

        self._ax.text(
            self.start.pos[0] + 0.02,
            self.start.pos[1],
            r"$\odot$",
            fontsize=60,
            color="black",
            horizontalalignment="center",
            verticalalignment="center_baseline",
            zorder=3,
            label=r"Start ($\odot$)",
        )
        self._ax.plot([], [], "ko", marker=r"$\odot$", markersize=10, label="Start", zorder=3)
        self._ax.text(
            self.goal.pos[0] + 0.02,  # x-coordinate from self.goal.pos
            self.goal.pos[1],  # y-coordinate from self.goal.pos
            r"$\otimes$",  # The LaTeX symbol (otimes) in math mode
            fontsize=60,  # Adjust size for visibility
            color="black",  # Choose a color (e.g., green, lime)
            horizontalalignment="center",  # Center the symbol horizontally
            verticalalignment="center_baseline",  # Center the symbol vertically
            zorder=3,  # Ensure it's drawn prominently
            label=r"Goal ($\otimes$)",  # Optional: Update label for legend
        )
        self._ax.plot([], [], "ko", marker=r"$\otimes$", markersize=10, label="Goal", zorder=3)

        if self.wind_field:
            self.wind_field.plot_wind_field(
                self._ax,
                xlim=(self.state_low[0], self.state_high[0]),
                ylim=(self.state_low[1], self.state_high[1]),
            )

        (self.trajectory_plot,) = self._ax.plot(
            [],
            [],
            "b-",
            alpha=0.5,
            label="Trajectory",
            zorder=1,
            lw=2.5,
        )  # Blue line
        (self.agent_plot,) = self._ax.plot(
            [], [], "ro", markersize=8, label="Agent", zorder=3
        )  # Red circle

        # add goal to legend below plot with three columns
        self._ax.legend(
            loc="upper center",
            fontsize=10,
            frameon=True,
            ncol=4,
            bbox_to_anchor=(0.5, -0.25),
        )

    def _render_frame(self) -> np.ndarray | None:
        # update trajectory
        traj_x = [s[0] for s in self.trajectory]
        traj_y = [s[1] for s in self.trajectory]
        self.trajectory_plot.set_data(traj_x, traj_y)  # type: ignore

        # update agent position
        self.agent_plot.set_data([self.state[0]], [self.state[1]])  # type: ignore

        # update action arrow (remove old, add new)
        if self.action_arrow_patch is not None:
            self.action_arrow_patch.remove()
            self.action_arrow_patch = None
        arrow_scale = 0.03
        dx = self.action[0] * arrow_scale  # type: ignore
        dy = self.action[1] * arrow_scale  # type: ignore

        if np.linalg.norm([dx, dy]) > 1e-4:
            # Create a new arrow patch
            self.action_arrow_patch = FancyArrowPatch(
                (self.state[0], self.state[1]),  # type: ignore
                (self.state[0] + dx, self.state[1] + dy),  # type: ignore
                color="darkorange",
                mutation_scale=15,
                alpha=0.9,
                zorder=2,  # Above trajectory, below agent
            )
            self._ax.add_patch(self.action_arrow_patch)  # type: ignore


@dataclass
class Circle:
    pos: np.ndarray
    radius: float

    def __contains__(self, item):
        if len(item) >= 2:
            return np.linalg.norm(item[:2] - self.pos) <= self.radius
        return False

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        theta = rng.uniform(0, 2 * np.pi)
        r = self.radius * np.sqrt(rng.uniform(0, 1))
        x = self.pos[0] + r * np.cos(theta)
        y = self.pos[1] + r * np.sin(theta)
        return np.array([x, y])


def define_transition_matrices(
    m: float, cx: float, cy: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Define the transition matrices A and B for the point mass dynamics.

    Args:
        m: Mass of the point mass.
        cx: Damping coefficient in x direction.
        cy: Damping coefficient in y direction.
        dt: Time step for the discrete dynamics.

    Returns:
        A: State transition matrix.
        B: Control input matrix.
    """
    A = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, np.exp(-cx * dt / m), 0],
            [0, 0, 0, np.exp(-cy * dt / m)],
        ]
    )
    B = np.array(
        [
            [0, 0],
            [0, 0],
            [(m / cx) * (1 - np.exp(-cx * dt / m)), 0],
            [0, (m / cy) * (1 - np.exp(-cy * dt / m))],
        ]
    )
    return A, B


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
        max_len=0.15,
    ):
        X, Y = self.plot_XY(xlim, ylim)

        # flatten
        XY = np.stack([X.ravel(), Y.ravel()], axis=1)
        UV = np.apply_along_axis(self.__call__, 1, XY)
        U, V = UV[:, 0], UV[:, 1]

        wind_mag = np.hypot(U, V)
        mask = wind_mag > 0

        scale = np.max(wind_mag) / max_len

        ax.quiver(
            X.ravel()[mask],
            Y.ravel()[mask],
            U[mask],
            V[mask],
            wind_mag[mask],
            scale=scale,
            scale_units="xy",
            width=0.0035,
            pivot="mid",
        )


class WindParcour(WindField):
    def __init__(self, magnitude: float = 10.0, difficulty: DifficultyLevel = "easy"):
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
        Xs, Ys = [], []
        delta = 0.2

        for box in self.boxes:
            # intersection with plot limits
            x0 = max(xlim[0], box[0][0])
            x1 = min(xlim[1], box[1][0])
            y0 = max(ylim[0], box[0][1])
            y1 = min(ylim[1], box[1][1])

            num_x = (x1 - x0) // delta
            delta_x = (x1 - x0) / num_x
            num_y = (y1 - y0) // delta
            delta_y = (y1 - y0) / num_y

            xs = x0 + delta_x * (0.5 + np.arange(num_x))
            ys = y0 + delta_y * (0.5 + np.arange(num_y))
            X, Y = np.meshgrid(xs, ys)

            Xs.append(X.ravel())
            Ys.append(Y.ravel())

        # concatenate all boxes
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        return X, Y

    def plot_wind_field(
        self,
        ax: Axes,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ):
        # plot rectangles for each boxes
        for box in self.boxes:
            rect = patches.Rectangle(
                box[0],  # type: ignore
                box[1][0] - box[0][0],
                box[1][1] - box[0][1],
                color="gray",
                alpha=0.1,
            )
            ax.add_patch(rect)

        # plot wind field
        super().plot_wind_field(ax, xlim, ylim)

    def __call__(self, pos: np.ndarray) -> np.ndarray:
        for box in self.boxes:
            if np.all(box[0] <= pos) and np.all(pos <= box[1]):
                return np.array([-self.magnitude, 0.0])
        return np.array([0.0, 0.0])
