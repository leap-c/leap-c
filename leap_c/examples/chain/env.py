from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from leap_c.examples.chain.config import ChainParams, make_default_chain_params
from leap_c.examples.chain.utils.dynamics import (
    create_discrete_numpy_dynamics,
    get_f_expl_expr,
)
from leap_c.examples.chain.utils.ellipsoid import Ellipsoid
from leap_c.examples.chain.utils.resting_chain_solver import RestingChainSolver


class ChainEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        params: ChainParams | None = None,
        n_mass: int = 5,
        pos_last_mass_ref: np.ndarray | None = None,
    ):
        super().__init__()
        # Create default chain params
        if params is None:
            params = make_default_chain_params(n_mass)

        if pos_last_mass_ref is None:
            pos_last_mass_ref = params.fix_point.value + np.array(
                [0.033 * (n_mass - 1), 0.0, 0.0]
            )

        self.n_mass = n_mass

        # Extract parameter values from chain_params
        self.dyn_param_dict = {
            "L": params.L.value,
            "D": params.D.value,
            "C": params.C.value,
            "m": params.m.value,
            "w": params.w.value,
        }

        # Use ranges from chain_params
        self.phi_range = tuple(params.phi_range.value)
        self.theta_range = tuple(params.theta_range.value)

        self.nx_pos = 3 * (n_mass - 1)
        self.nx_vel = 3 * (n_mass - 2)

        # Set default values
        self.fix_point = np.array([0.0, 0.0, 0.0])
        self.pos_last_ref = pos_last_mass_ref
        self.dt = 0.05  # Default time step
        self.max_time = 10.0  # Default maximum simulation time
        vmax = 1.0  # Default maximum action value

        # Compute observation space
        pos_max = np.array(self.dyn_param_dict["L"]) * (n_mass - 1)
        pos_min = -pos_max
        vel_max = np.array([2.0, 2.0, 2.0] * (n_mass - 2))
        vel_min = -vel_max
        self.observation_space = spaces.Box(
            low=np.concatenate([pos_min, vel_min], dtype=np.float32),
            high=np.concatenate([pos_max, vel_max], dtype=np.float32),
        )

        self.action_space = spaces.Box(
            low=np.array([-vmax, -vmax, -vmax], dtype=np.float32),
            high=np.array([vmax, vmax, vmax], dtype=np.float32),
        )

        self.render_mode = render_mode
        self.trajectory = []

        # Create the discrete dynamics function
        self.discrete_dynamics = create_discrete_numpy_dynamics(n_mass, self.dt)

        self.resting_chain_solver = RestingChainSolver(
            n_mass=n_mass,
            f_expl=get_f_expl_expr,
            params=params,
        )

        self.x_ref, self.u_ref = self.resting_chain_solver(p_last=self.pos_last_ref)

        self.ellipsoid = Ellipsoid(
            center=self.fix_point,
            radii=np.sum(params.L.value.reshape(-1, 3), axis=0),
        )

        self._set_canvas()

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        u = action
        self.action = action

        self.u = u

        self.state = self.discrete_dynamics(
            x=self.state,
            u=self.action,
            p=self.dyn_param_dict,
            fix_point=self.fix_point,
        )

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

        self.time += self.dt
        trunc = self.time > self.max_time

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
        phi = self.np_random.uniform(low=self.phi_range[0], high=self.phi_range[1])  # type:ignore
        theta = self.np_random.uniform(
            low=self.theta_range[0], high=self.theta_range[1]
        )  # type:ignore
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
            ax_k.set_xticks(range(self.n_mass + 1))
            ax_k.set_xlim(0, self.n_mass + 1)
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
