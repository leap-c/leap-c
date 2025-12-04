from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.lines import Line2D

from leap_c.examples.utils.matplotlib_env import MatplotlibRenderEnv


@dataclass(kw_only=True)
class LqrEnvConfig:
    """Configuration for the LQR environment.

    Attributes:
        dt: Time step in seconds.
        Q: State cost matrix (2x2).
        R: Control cost matrix (1x1).
        x_min: Minimum position.
        x_max: Maximum position.
        v_min: Minimum velocity.
        v_max: Maximum velocity.
        F_min: Minimum force.
        F_max: Maximum force.
        max_time: Maximum time for an episode in seconds.
    """

    dt: float = 0.1
    Q: np.ndarray | None = None
    R: np.ndarray | None = None
    x_min: float = -2.0
    x_max: float = 2.0
    v_min: float = -2.0
    v_max: float = 2.0
    F_min: float = -0.5
    F_max: float = 0.5
    max_time: float = 10.0

    def __post_init__(self):
        """Set default Q and R matrices if not provided."""
        if self.Q is None:
            self.Q = np.diag([1.0, 0.1])
        if self.R is None:
            self.R = np.array([[0.01]])


class LqrEnv(MatplotlibRenderEnv):
    """A simple 1D LQR environment with position and velocity states.

    The dynamics follow a discrete-time double integrator:
        x[k+1] = x[k] + dt * v[k] + 0.5 * dt^2 * F[k]
        v[k+1] = v[k] + dt * F[k]

    Observation Space:
    ------------------
    The observation is a `ndarray` with shape `(2,)` and dtype `np.float32` representing:
    | Num | Observation         | Min        | Max        |
    |-----|---------------------|------------|------------|
    | 0   | position (x)        | x_min      | x_max      |
    | 1   | velocity (v)        | v_min      | v_max      |

    Action Space:
    -------------
    The action is a `ndarray` with shape `(1,)` and dtype `np.float32`
    representing the applied force, bounded by [F_min, F_max].

    Reward:
    -------
    The reward is the negative LQR cost:
        r = -(x^T Q x + u^T R u)
    where Q and R are the state and control cost matrices.

    Termination:
    ------------
    The episode terminates if:
    - The agent leaves the allowed state space (position or velocity out of bounds)

    Truncation:
    -----------
    The episode is truncated if the maximum time is exceeded.

    Info:
    -----
    The info dictionary contains:
    - "task": {"violation": bool, "success": bool}
      - violation: True if out of bounds
      - success: True if close to origin with low velocity

    Attributes:
        cfg: Configuration object for the environment.
        state_low: Lower bounds for position and velocity.
        state_high: Upper bounds for position and velocity.
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        A: State transition matrix (2x2).
        B: Control input matrix (2x1).
        state: Current state [x, v].
        time: Elapsed time in the current episode.
        trajectory: List of states visited during the episode (used in rendering).
        trajectory_x_plot: Line object for position trajectory.
        trajectory_v_plot: Line object for velocity trajectory.
        action_plot: Line object for action trajectory.
    """

    cfg: LqrEnvConfig
    state_low: np.ndarray
    state_high: np.ndarray
    observation_space: spaces.Box
    action_space: spaces.Box
    A: np.ndarray
    B: np.ndarray
    state: np.ndarray | None
    time: float
    trajectory: list[np.ndarray]
    trajectory_x_plot: Line2D | None
    trajectory_v_plot: Line2D | None
    action_plot: Line2D | None
    actions: list[np.ndarray]

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        cfg: LqrEnvConfig | None = None,
    ):
        """Initialize the LQR environment.

        Args:
            render_mode: The mode to render with. Supported modes are: human, rgb_array, None.
            cfg: Configuration for the environment. If None, default configuration is used.
        """
        super().__init__(render_mode=render_mode)

        self.cfg = LqrEnvConfig() if cfg is None else cfg

        # Gymnasium setup
        self.state_low = np.array([self.cfg.x_min, self.cfg.v_min], dtype=np.float32)
        self.state_high = np.array([self.cfg.x_max, self.cfg.v_max], dtype=np.float32)
        self.action_low = np.array([self.cfg.F_min], dtype=np.float32)
        self.action_high = np.array([self.cfg.F_max], dtype=np.float32)

        self.observation_space = spaces.Box(low=self.state_low, high=self.state_high)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)

        # Define discrete-time dynamics matrices
        dt = self.cfg.dt
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.B = np.array([[0.5 * dt**2], [dt]])

        # Environment state
        self.state = None
        self.time = 0.0
        self.trajectory = []
        self.actions = []

        # Plotting attributes (initialize to None)
        self.trajectory_x_plot = None
        self.trajectory_v_plot = None
        self.action_plot = None

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step within the environment.

        Args:
            action: Control input (force).

        Returns:
            Tuple containing:
                - observation: Next observation [x, v].
                - reward: Reward for this step (negative LQR cost).
                - terminated: Whether the episode terminated.
                - truncated: Whether the episode was truncated.
                - info: Dictionary with task information.
        """
        if self.state is None:
            raise ValueError("Environment must be reset before stepping.")

        # Clip action to bounds
        action = np.clip(action, self.action_low, self.action_high)

        # State transition
        self.state = self.A @ self.state + self.B @ action.reshape(-1)
        self.time += self.cfg.dt

        # Store trajectory for rendering
        self.trajectory.append(self.state.copy())
        self.actions.append(action.copy())

        # Check termination conditions
        out_of_bounds = (self.state_high < self.state).any() or (self.state_low > self.state).any()

        # Success if close to origin with low velocity
        close_to_origin = np.abs(self.state[0]) < 0.1 and np.abs(self.state[1]) < 0.1

        terminated = bool(out_of_bounds)
        truncated = self.time >= self.cfg.max_time

        # Compute reward (negative LQR cost)
        state_cost = self.state @ self.cfg.Q @ self.state
        control_cost = action.reshape(-1) @ self.cfg.R @ action.reshape(-1)
        reward = float(-(state_cost + control_cost))

        info = {}
        if terminated or truncated:
            info = {
                "task": {
                    "violation": bool(out_of_bounds),
                    "success": bool(close_to_origin and not out_of_bounds),
                }
            }

        return self._observation(), reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.
                - "mode": "train" samples from a wider range, otherwise samples near origin.

        Returns:
            Tuple containing:
                - observation: Initial observation [x, v].
                - info: Empty dictionary.
        """
        super().reset(seed=seed)
        self.time = 0.0
        self.trajectory = []
        self.actions = []

        # Sample initial state
        if options is not None and "mode" in options and options["mode"] == "train":
            # Training: sample from wider range
            low = np.array([self.cfg.x_min * 0.8, self.cfg.v_min * 0.8])
            high = np.array([self.cfg.x_max * 0.8, self.cfg.v_max * 0.8])
            self.state = self.np_random.uniform(low=low, high=high)
        else:
            # Testing: sample from narrower range near origin
            low = np.array([self.cfg.x_min * 0.5, self.cfg.v_min * 0.5])
            high = np.array([self.cfg.x_max * 0.5, self.cfg.v_max * 0.5])
            self.state = self.np_random.uniform(low=low, high=high)

        self.trajectory.append(self.state.copy())

        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        """Get the current observation.

        Returns:
            Current state [x, v] as float32 array.
        """
        return self.state.copy().astype(np.float32)

    def _render_setup(self):
        """Initialize the Matplotlib figure and axes for rendering."""
        if self.render_mode == "human":
            plt.ion()

        self._fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        self._ax = axes

        # Position subplot
        ax = axes[0]
        (self.trajectory_x_plot,) = ax.plot([], [], "b-", label="Position x", linewidth=2)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, label="Target")
        ax.set_xlim(0, self.cfg.max_time / self.cfg.dt)
        ax.set_ylim(self.cfg.x_min * 1.1, self.cfg.x_max * 1.1)
        ax.set_ylabel("Position x")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Velocity subplot
        ax = axes[1]
        (self.trajectory_v_plot,) = ax.plot([], [], "r-", label="Velocity v", linewidth=2)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, label="Target")
        ax.set_xlim(0, self.cfg.max_time / self.cfg.dt)
        ax.set_ylim(self.cfg.v_min * 1.1, self.cfg.v_max * 1.1)
        ax.set_ylabel("Velocity v")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Action subplot
        ax = axes[2]
        (self.action_plot,) = ax.step([], [], "g-", where="post", label="Force F", linewidth=2)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlim(0, self.cfg.max_time / self.cfg.dt)
        ax.set_ylim(self.cfg.F_min * 1.1, self.cfg.F_max * 1.1)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Force F")
        ax.grid(True, alpha=0.3)
        ax.legend()

        self._fig.suptitle("LQR System Trajectory", fontsize=14)
        self._fig.tight_layout()

    def _render_frame(self) -> np.ndarray | None:
        """Update the rendering with current trajectory data."""
        if len(self.trajectory) == 0:
            return None

        time_steps = np.arange(len(self.trajectory))
        trajectory_array = np.array(self.trajectory)

        # Update position plot
        self.trajectory_x_plot.set_data(time_steps, trajectory_array[:, 0])

        # Update velocity plot
        self.trajectory_v_plot.set_data(time_steps, trajectory_array[:, 1])

        # Update action plot
        if len(self.actions) > 0:
            actions_array = np.array(self.actions)
            self.action_plot.set_data(np.arange(len(actions_array)), actions_array[:, 0])
