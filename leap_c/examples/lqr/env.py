from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box
from matplotlib.lines import Line2D
from scipy.linalg import solve_discrete_are

from leap_c.examples.utils.matplotlib_env import MatplotlibRenderEnv


@dataclass(kw_only=True)
class LqrEnvConfig:
    """Configuration for the LQR environment.

    Attributes:
        dt: Time step in seconds.
        q_diag_sqrt: Square root of diagonal Q matrix elements.
        r_diag_sqrt: Square root of diagonal R matrix elements.
        x_min: Minimum position.
        x_max: Maximum position.
        v_min: Minimum velocity.
        v_max: Maximum velocity.
        F_min: Minimum force.
        F_max: Maximum force.
        max_time: Maximum time for an episode in seconds.
        mass: Mass of the system (kg).
        damping: Damping coefficient (Ns/m).
        stiffness: Spring stiffness coefficient (N/m).
    """

    dt: float = 0.1
    q_diag_sqrt: np.ndarray = field(default_factory=lambda: np.sqrt(np.array([1.0, 0.1])))
    r_diag_sqrt: np.ndarray = field(default_factory=lambda: np.sqrt(np.array([0.01])))
    x_min: float = -2.0
    x_max: float = 2.0
    v_min: float = -2.0
    v_max: float = 2.0
    F_min: float = -0.5
    F_max: float = 0.5
    max_time: float = 5.0
    mass: float = 1.0
    damping: float = 0.1
    stiffness: float = 0.5

    def __post_init__(self):
        """Compute `Q` and `R` from sqrt diagonal representation using Cholesky factorization."""
        self.Q = np.diag(np.square(self.q_diag_sqrt))
        self.R = np.diag(np.square(self.r_diag_sqrt))


class LqrEnv(MatplotlibRenderEnv):
    """A simple 1D LQR environment with position and velocity states.

    The dynamics follow a discrete-time mass-spring-damper system:
        x[k+1] = x[k] + dt * v[k]
        v[k+1] = v[k] + dt * (F[k]/m - (b/m)*v[k] - (k/m)*x[k])

    where m is mass, b is damping coefficient, and k is spring stiffness.

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
    observation_space: Box
    action_space: Box
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
        state_low = np.array([self.cfg.x_min, self.cfg.v_min], np.float32)
        state_high = np.array([self.cfg.x_max, self.cfg.v_max], np.float32)
        self.observation_space = Box(state_low, state_high)

        action_low = np.array([self.cfg.F_min], np.float32)
        action_high = np.array([self.cfg.F_max], np.float32)
        self.action_space = Box(action_low, action_high)

        # Define discrete-time dynamics matrices for mass-spring-damper system
        # Continuous dynamics: x_dot = v, v_dot = F/m - (b/m)*v - (k/m)*x
        # Discretized using Euler method
        dt = self.cfg.dt
        m = self.cfg.mass
        b = self.cfg.damping
        k = self.cfg.stiffness

        self.A = np.array([[1.0, dt], [-dt * k / m, 1.0 - dt * b / m]])
        self.B = np.array([[0.0], [dt / m]])

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
        act_space = self.action_space
        action = np.clip(action, act_space.low, act_space.high, dtype=act_space.dtype)

        # State transition
        self.state = self.A @ self.state + self.B @ action.reshape(-1)
        self.time += self.cfg.dt

        # Store trajectory for rendering
        self.trajectory.append(self.state.copy())
        self.actions.append(action.copy())

        # Check termination conditions
        obs_space = self.observation_space
        out_of_bounds = (obs_space.high < self.state).any() or (obs_space.low > self.state).any()

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


def plot_optimal_value_and_policy(
    env: LqrEnv | None = None,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    x_min: float = -2.0,
    x_max: float = 2.0,
    v_min: float = -2.0,
    v_max: float = 2.0,
    n_points: int = 101,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot heatmaps of the optimal LQR value function and policy.

    This function solves the discrete-time algebraic Riccati equation (DARE)
    to compute the optimal value function V(x) = x^T P x and the optimal
    policy u*(x) = -K x, then visualizes them as 2D heatmaps.

    Args:
        env: LQR environment to extract matrices from. If None, must provide A, B, Q, R.
        A: State transition matrix (2x2). If None, extracted from env.
        B: Control input matrix (2x1). If None, extracted from env.
        Q: State cost matrix (2x2). If None, extracted from env.
        R: Control cost matrix (1x1). If None, extracted from env.
        x_min: Minimum position value for the grid.
        x_max: Maximum position value for the grid.
        v_min: Minimum velocity value for the grid.
        v_max: Maximum velocity value for the grid.
        n_points: Number of grid points in each dimension.

    Returns:
        Tuple of (figure, axes) containing the heatmap plots.

    Raises:
        ValueError: If neither env nor all matrices (A, B, Q, R) are provided.
    """
    # Extract matrices from environment or use provided ones
    if env is not None:
        A = env.A if A is None else A
        B = env.B if B is None else B
        Q = env.cfg.Q if Q is None else Q
        R = env.cfg.R if R is None else R
        x_min = env.cfg.x_min if x_min == -2.0 else x_min
        x_max = env.cfg.x_max if x_max == 2.0 else x_max
        v_min = env.cfg.v_min if v_min == -2.0 else v_min
        v_max = env.cfg.v_max if v_max == 2.0 else v_max
    elif A is None or B is None or Q is None or R is None:
        raise ValueError("Must provide either env or all matrices (A, B, Q, R)")

    # Solve discrete-time algebraic Riccati equation
    P = solve_discrete_are(A, B, Q, R)

    # Compute optimal feedback gain K
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

    # Create grid in state space
    x_vals = np.linspace(x_min, x_max, n_points)
    v_vals = np.linspace(v_min, v_max, n_points)

    # Initialize arrays for value function and policy
    V = np.zeros((n_points, n_points))  # Value function
    U = np.zeros((n_points, n_points))  # Optimal control

    # Compute value function and optimal policy at each grid point
    for i, v in enumerate(v_vals):
        for j, x in enumerate(x_vals):
            state = np.array([[x], [v]])
            # Value function: V(x) = x^T P x
            V[i, j] = float(state.T @ P @ state)
            # Optimal policy: u*(x) = -K x
            U[i, j] = float(-K @ state)

    # Create figure with horizontal subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot value function heatmap
    im1 = axes[0].imshow(
        V,
        extent=[x_min, x_max, v_min, v_max],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_xlabel("Position x")
    axes[0].set_ylabel("Velocity v")
    axes[0].set_title("Optimal Value Function V(x) = x^T P x")
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label("Value")

    # Plot optimal policy heatmap
    im2 = axes[1].imshow(
        U,
        extent=[x_min, x_max, v_min, v_max],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.max(np.abs(U)),
        vmax=np.max(np.abs(U)),
    )
    axes[1].set_xlabel("Position x")
    axes[1].set_ylabel("Velocity v")
    axes[1].set_title("Optimal Policy u*(x) = -K x")
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label("Control Force")

    # Add zero contours
    axes[0].contour(
        x_vals,
        v_vals,
        V,
        levels=[1.0, 2.0, 5.0, 10.0],
        colors="white",
        alpha=0.4,
        linewidths=1,
    )
    axes[1].contour(x_vals, v_vals, U, levels=[0.0], colors="black", linewidths=2)

    fig.suptitle("LQR Optimal Value Function and Policy", fontsize=14, y=1.00)
    fig.tight_layout()

    return fig, axes


if __name__ == "__main__":
    env = LqrEnv()

    fig, axes = plot_optimal_value_and_policy(env=env)

    plt.show()
