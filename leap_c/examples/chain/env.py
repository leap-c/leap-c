from dataclasses import dataclass, field
from typing import Any, Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from leap_c.examples.chain.dynamics import (
    create_discrete_casadi_dynamics,
    define_f_expl_expr,
)
from leap_c.examples.chain.utils.ellipsoid import Ellipsoid
from leap_c.examples.chain.utils.resting_chain_solver import RestingChainSolver
from leap_c.examples.utils.matplotlib_env import MatplotlibRenderEnv


@dataclass
class ChainEnvConfig:
    """Configuration for the Chain environment."""

    n_mass: int = 5  # number of masses in the chain
    dt: float = 0.05  # simulation time step [s]
    max_time: float = 10.0  # maximum simulation time after which the episode is truncated [s]
    vmax: float = 1.0  # maximum velocity of the last mass [m/s]

    # dynamics parameters (dependent defaults)
    L: list[float] = field(default_factory=list)  # rest length of spring [m]
    D: list[float] = field(default_factory=list)  # spring stiffness [N/m]
    C: list[float] = field(default_factory=list)  # damping coefficient [Ns/m]
    m: list[float] = field(default_factory=list)  # mass of the balls [kg]
    w: list[float] = field(default_factory=list)  # disturbance on intermediate balls [N]

    def __post_init__(self):
        if not self.L:
            self.L = [0.033, 0.033, 0.033] * (self.n_mass - 1)
        if not self.D:
            self.D = [1.0, 1.0, 1.0] * (self.n_mass - 1)
        if not self.C:
            self.C = [0.1, 0.1, 0.1] * (self.n_mass - 1)
        if not self.m:
            self.m = [0.033] * (self.n_mass - 1)
        if not self.w:
            self.w = [0.0, 0.0, 0.0] * (self.n_mass - 2)


class ChainEnv(MatplotlibRenderEnv, gym.Env):
    """An environment of a chain of masses, connected by "springs".

    The first mass is fixed at a given point,
    and the last mass can be controlled by setting its velocity.
    The goal is to move the chain to a target position while minimizing mass velocity.

    The system corresponds to the example in
    L. Wirsching, H. G. Bock, and M. Diehl, “Fast nmpc of a chain of masses
    connected by springs,” in 2006 IEEE Conference on Computer Aided Control
    System Design, 2006 IEEE International Conference on Control Applications,
    2006 IEEE International Symposium on Intelligent Control, pp. 591–596, IEEE,
    2006.
    NOTE: An additoinal damping term has been added to the dynamics.

    Observation Space:
    ------------------

    The observation is of shape `(6 * n_mass - 9,)` representing the state of the
    system. It consists of the 3D positions of the `n_mass - 1` free masses and
    the 3D velocities of the `n_mass - 2` masses (the free masses, except the last mass,
    whose velocity is being controlled).

    The state is structured as `[p_2, ..., p_{n_mass}, v_2, ..., v_{n_mass-1}]`, where `p_i` is the
    position of the i-th mass and `v_i` is its velocity.

    Action Space:
    -------------

    The action is a `ndarray` with shape `(3,)` which can take values in the range `(-vmax, vmax)`.
    It represents the velocity of the last mass.

    Reward:
    -------

    The reward is designed to encourage the agent to move the last mass to the target position
    while minimizing velocity. It is calculated as:
    `r = 10 * (r_dist + r_vel)`
    where:
    - `r_dist` is the negative l1 norm of the distance between the last mass
        and the target position.
    - `r_vel` is the negative l2 norm of the velocities of the masses, scaled by 0.1.

    Termination and truncation:
    ---------------------------
    The system never terminates, but the episode is truncated after
    `max_time` seconds simulation time.

    Info:
    -----
    The info dictionary contains:
    - "task": {"violation": bool, "success": bool}
      - violation: Always False.
      - success: True if goal reached

    Attributes:
        cfg: Configuration for the environment.
        fix_point: Fixed point where the first mass is anchored.
        pos_last_ref: Target state for the chain.
        nx_pos: Number of position states in the observation.
        nx_vel: Number of velocity states in the observation.
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        trajectory: List to store the trajectory of states during an episode.
        dyn_param_dict: Dictionary of dynamics parameters.
        discrete_dynamics: Function to compute the next state given the current state and action.
        resting_chain_solver: Solver to compute the target resting state of the chain.
        init_phi_range: Sampling range of initial azimuthal angles for the last mass.
        init_theta_range: Sampling range of initial polar angles for the last mass.
        ellipsoid: Ellipsoid class for translating the sampled initial spherical coordinates into
            the initial cartesian coordinates.
    """

    cfg: ChainEnvConfig
    reset_needed: bool
    t: float
    pos_last_ref: np.ndarray
    nx_pos: int
    nx_vel: int
    observation_space: gym.spaces.Box
    action_space: gym.spaces.Box
    trajectory: list[np.ndarray]
    dyn_param_dict: dict[str, np.ndarray]
    discrete_dynamics: Callable
    resting_chain_solver: RestingChainSolver
    init_phi_range: np.ndarray
    init_theta_range: np.ndarray
    ellipsoid: Ellipsoid

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        cfg: ChainEnvConfig | None = None,
    ):
        """Initialize the Chain environment.

        Args:
            render_mode: The mode to render with. Supported modes are: human, rgb_array, None.
            cfg: Configuration for the environment. If None, default configuration is used.
        """
        super().__init__(render_mode=render_mode)
        self.cfg = ChainEnvConfig() if cfg is None else cfg

        self.fix_point = np.zeros(3)

        self.init_phi_range = np.array([np.pi / 6, np.pi / 3])
        self.init_theta_range = np.array([-np.pi / 4, np.pi / 4])

        length = self.cfg.L[0]
        pos_last_mass_ref = self.fix_point + np.array([length * (self.cfg.n_mass - 1), 0.0, 0.0])

        self.nx_pos = 3 * (self.cfg.n_mass - 1)
        self.nx_vel = 3 * (self.cfg.n_mass - 2)

        self.pos_last_ref = pos_last_mass_ref

        # Compute observation space
        pos_max = np.array(self.cfg.L) * (self.cfg.n_mass - 1)
        pos_min = -pos_max
        vel_max = np.array([self.cfg.vmax, self.cfg.vmax, self.cfg.vmax] * (self.cfg.n_mass - 2))
        vel_min = -vel_max
        self.observation_space = spaces.Box(
            low=np.concatenate([pos_min, vel_min], dtype=np.float32),
            high=np.concatenate([pos_max, vel_max], dtype=np.float32),
        )

        self.action_space = spaces.Box(
            low=np.array([-self.cfg.vmax, -self.cfg.vmax, -self.cfg.vmax], dtype=np.float32),
            high=np.array([self.cfg.vmax, self.cfg.vmax, self.cfg.vmax], dtype=np.float32),
        )

        self.trajectory = []

        self.dyn_param_dict = {
            "L": np.array(self.cfg.L),
            "D": np.array(self.cfg.D),
            "C": np.array(self.cfg.C),
            "m": np.array(self.cfg.m),
            "w": np.array(self.cfg.w),
        }

        self.discrete_dynamics = create_discrete_casadi_dynamics(self.cfg.n_mass, self.cfg.dt)

        self.resting_chain_solver = RestingChainSolver(
            n_mass=self.cfg.n_mass,
            f_expl=define_f_expl_expr,
            fix_point=self.fix_point,
            **self.dyn_param_dict,
        )

        self.x_ref, self.u_ref = self.resting_chain_solver(p_last=self.pos_last_ref)

        self.ellipsoid = Ellipsoid(
            center=self.fix_point,
            radii=np.sum(self.dyn_param_dict["L"].reshape(-1, 3), axis=0),
        )

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        u = action
        self.action = action

        self.u = u

        self.state = self.discrete_dynamics(
            x=self.state,
            u=self.action,
            **self.dyn_param_dict,
        )["x_next"].full()[:, 0]

        o = self.state.copy()

        # Calculate reward directly
        pos_last = self.state[self.nx_pos - 3 : self.nx_pos]
        vel = self.state[self.nx_pos :]
        r_dist = -np.linalg.norm(pos_last - self.pos_last_ref, axis=0, ord=1)
        r_vel = -0.1 * np.linalg.norm(vel, axis=0, ord=2)
        r = 10 * (r_dist + r_vel)

        reached_goal_pos = bool(np.linalg.norm(self.x_ref - self.state, axis=0, ord=2) < 1e-1)
        term = False

        self.time += self.cfg.dt
        trunc = self.time > self.cfg.max_time

        info = {}
        if trunc:
            info["task"] = {"success": reached_goal_pos, "violations": False}

        self.trajectory.append(o)

        return o, r, term, trunc, info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        self.state_trajectory = None
        self.state, self.action = self._init_state_and_action()
        self.time = 0.0
        self.trajectory = []

        return self.state.copy(), {}

    def _init_state_and_action(self):
        phi = self.np_random.uniform(low=self.init_phi_range[0], high=self.init_phi_range[1])
        theta = self.np_random.uniform(low=self.init_theta_range[0], high=self.init_theta_range[1])
        p_last = self.ellipsoid.spherical_to_cartesian(phi=phi, theta=theta)
        x_ss, u_ss = self.resting_chain_solver(p_last=p_last)

        return x_ss, u_ss

    def _render_setup(self):
        """One-time setup for the rendering."""
        self._fig, self._ax = plt.subplots(3, 1, figsize=(8, 10))

        # Plot reference
        ref_pos = np.vstack([self.fix_point, self.x_ref[: self.nx_pos].reshape(-1, 3)])
        # Ensure we scale each axis independently
        min_y = np.min(ref_pos, axis=0)
        max_y = np.max(ref_pos, axis=0)
        mid_y = (min_y + max_y) / 2
        max_delta = np.max(np.abs(max_y - mid_y)) * 1.1  # 10% margin
        low_ylim = mid_y - max_delta
        high_ylim = mid_y + max_delta

        labels = ["x", "y", "z"]
        self.lines = []
        for k, ax_k in enumerate(self._ax):
            ax_k.plot(ref_pos[:, k], "ro--", label="Reference")
            ax_k.grid()
            ax_k.set_xticks(range(self.cfg.n_mass + 1))
            ax_k.set_xlim(0, self.cfg.n_mass + 1)
            ax_k.set_ylim(low_ylim[k], high_ylim[k])
            ax_k.set_ylabel(labels[k])
            self.lines.append(
                ax_k.plot(range(ref_pos[:, k].shape[0]), ref_pos[:, k], ".-", label="Current")[0]
            )
            ax_k.legend()

    def _render_frame(self):
        """Update the plot with the current environment state."""
        pos = np.vstack([self.fix_point, self.state[: self.nx_pos].reshape(-1, 3)])
        for k, line in enumerate(self.lines):
            line.set_ydata(pos[:, k])
