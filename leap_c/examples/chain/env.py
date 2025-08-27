from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from leap_c.examples.chain.dynamics import (
    define_f_expl_expr,
    create_discrete_casadi_dynamics,
)
from leap_c.examples.chain.utils.ellipsoid import Ellipsoid
from leap_c.examples.chain.utils.resting_chain_solver import RestingChainSolver


@dataclass
class ChainEnvConfig:
    """Configuration for the Chain environment."""

    n_mass: int = 5  # number of masses in the chain
    dt: float = 0.05  # simulation time step [s]
    max_time: float = 10.0  # maximum simulation time [s]
    vmax: float = 1.0  # maximum velocity of the last mass [m/s]

    # dynamics parameters (dependent defaults)
    L: list[float] = field(default_factory=list)  # rest length of spring [m]
    D: list[float] = field(default_factory=list)  # spring stiffness [N/m]
    C: list[float] = field(default_factory=list)  # damping coefficient [Ns/m]
    m: list[float] = field(default_factory=list)  # mass of the balls [kg]
    w: list[float] = field(
        default_factory=list
    )  # disturbance on intermediate balls [N]

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


class ChainEnv(gym.Env):
    """An environment of a chain of masses.

    The first mass is fixed at a given point,
    and the last mass can be controlled by setting its velocity. The goal is to
    move the last mass to a target position.

    Observation Space:
    ------------------

    The observation is of shape `(6 * n_mass - 9,)` representing the state of the
    system. It consists of the 3D positions of the `n_mass - 1` free masses and
    the 3D velocities of the first `n_mass - 2` free masses.

    The state is structured as `[p_2, ..., p_{n_mass}, v_2, ..., v_{n_mass-1}]`, where `p_i` is the
    position of the i-th mass and `v_i` is its velocity.

    Action Space:
    -------------

    The action is a `ndarray` with shape `(3,)` which can take values in the range `(-vmax, vmax)`.
    It represents the velocity of the second mass (the first free mass).

    Reward:
    -------

    The reward is designed to encourage the agent to move the last mass to the target position
    while minimizing velocity. It is calculated as:
    `r = 10 * (r_dist + r_vel)`
    where:
    - `r_dist` is the negative l1 norm of the distance between the last mass and the target position.
    - `r_vel` is the negative l2 norm of the velocities of the masses, scaled by -0.1.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        cfg: ChainEnvConfig | None = None,
    ):
        super().__init__()
        self.cfg = ChainEnvConfig() if cfg is None else cfg

        self.fix_point = np.zeros(3)

        self.init_phi_range = np.array([np.pi / 6, np.pi / 3])
        self.init_theta_range = np.array([-np.pi / 4, np.pi / 4])

        length = self.cfg.L[0]
        pos_last_mass_ref = self.fix_point + np.array(
            [length * (self.cfg.n_mass - 1), 0.0, 0.0]
        )

        self.nx_pos = 3 * (self.cfg.n_mass - 1)
        self.nx_vel = 3 * (self.cfg.n_mass - 2)

        self.pos_last_ref = pos_last_mass_ref

        # Compute observation space
        pos_max = np.array(self.cfg.L) * (self.cfg.n_mass - 1)
        pos_min = -pos_max
        vel_max = np.array(
            [self.cfg.vmax, self.cfg.vmax, self.cfg.vmax] * (self.cfg.n_mass - 2)
        )
        vel_min = -vel_max
        self.observation_space = spaces.Box(
            low=np.concatenate([pos_min, vel_min], dtype=np.float32),
            high=np.concatenate([pos_max, vel_max], dtype=np.float32),
        )

        self.action_space = spaces.Box(
            low=np.array(
                [-self.cfg.vmax, -self.cfg.vmax, -self.cfg.vmax], dtype=np.float32
            ),
            high=np.array(
                [self.cfg.vmax, self.cfg.vmax, self.cfg.vmax], dtype=np.float32
            ),
        )

        self.render_mode = render_mode
        self.trajectory = []

        self.dyn_param_dict = {
            "L": np.array(self.cfg.L),
            "D": np.array(self.cfg.D),
            "C": np.array(self.cfg.C),
            "m": np.array(self.cfg.m),
            "w": np.array(self.cfg.w),
        }

        self.discrete_dynamics = create_discrete_casadi_dynamics(
            self.cfg.n_mass, self.cfg.dt
        )

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

        self._set_canvas()

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

        reached_goal_pos = bool(
            np.linalg.norm(self.x_ref - self.state, axis=0, ord=2) < 1e-1
        )
        term = False

        self.time += self.cfg.dt
        trunc = self.time > self.cfg.max_time

        info = {}
        if trunc:
            info["task"] = {"success": reached_goal_pos, "violations": False}

        self.trajectory.append(o)

        return o, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        self.state_trajectory = None
        self.state, self.action = self._init_state_and_action()
        self.time = 0.0
        self.trajectory = []
        plt.close("all")
        self.canvas = None
        self.line = None

        self._set_canvas()

        return self.state.copy(), {}

    def _init_state_and_action(self):
        phi = self.np_random.uniform(
            low=self.init_phi_range[0], high=self.init_phi_range[1]
        )
        theta = self.np_random.uniform(
            low=self.init_theta_range[0], high=self.init_theta_range[1]
        )
        p_last = self.ellipsoid.spherical_to_cartesian(phi=phi, theta=theta)
        x_ss, u_ss = self.resting_chain_solver(p_last=p_last)

        return x_ss, u_ss

    def _set_canvas(self):
        plt.figure()
        ax = [plt.subplot(3, 1, i) for i in range(1, 4)]

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
        for k, ax_k in enumerate(ax):
            ax_k.plot(ref_pos[:, k], "ro--")
            ax_k.grid()
            ax_k.set_xticks(range(self.cfg.n_mass + 1))
            ax_k.set_xlim(0, self.cfg.n_mass + 1)
            ax_k.set_ylim(low_ylim[k], high_ylim[k])
            ax_k.set_ylabel(labels[k])
            self.lines.append(
                ax_k.plot(range(ref_pos[:, k].shape[0]), ref_pos[:, k], ".-")[0]
            )

        self.canvas = FigureCanvas(plt.gcf())

    def render(self):
        if self.render_mode is None:
            return None

        if self.render_mode in ["rgb_array", "human"]:
            pos = np.vstack([self.fix_point, self.state[: self.nx_pos].reshape(-1, 3)])
            for k, line in enumerate(self.lines):
                line.set_ydata(pos[:, k])

            # Convert the plot to an RGB string
            s, (width, height) = self.canvas.print_to_buffer()
            # Convert the RGB string to a NumPy array
            return np.frombuffer(s, np.uint8).reshape((height, width, 4))[:, :, :3]
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")
