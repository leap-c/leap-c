from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from leap_c.examples.chain.mpc import get_f_expl_expr
from leap_c.examples.chain.utils import (
    Ellipsoid,
    RestingChainSolver,
    nominal_params_to_structured_nominal_params,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def _cont_f_expl(
    x: np.ndarray,
    u: np.ndarray,
    p: dict[str, np.ndarray],
    fix_point: np.ndarray | None = None,
) -> np.ndarray:
    assert all(
        key in p for key in ["D", "L", "C", "m", "w"]
    ), "Not all necessary parameters are in p."

    if fix_point is None:
        fix_point = np.zeros(3, 1)

    n_masses = p["m"].shape[0] + 1

    xpos = x[: 3 * (p["m"].shape[0])]
    xvel = x[3 * (p["m"].shape[0]) :]

    # Force on intermediate masses
    f = np.zeros(3 * (n_masses - 2))

    # Gravity force on intermediate masses
    for i in range(int(f.shape[0] / 3)):
        f[3 * i + 2] = -9.81

    n_link = n_masses - 1

    # Spring force
    for i in range(n_link):
        if i == 0:
            dist = xpos[i * 3 : (i + 1) * 3] - fix_point
        else:
            dist = xpos[i * 3 : (i + 1) * 3] - xpos[(i - 1) * 3 : i * 3]

        F = np.zeros(3)
        for j in range(F.shape[0]):
            F[j] = (
                p["D"][i + j]
                / p["m"][i]
                * (1 - p["L"][i + j] / np.linalg.norm(dist))
                * dist[j]
            )

        # mass on the right
        if i < n_link - 1:
            f[i * 3 : (i + 1) * 3] -= F

        # mass on the left
        if i > 0:
            f[(i - 1) * 3 : i * 3] += F

    # Damping force
    for i in range(n_link):
        if i == 0:
            vel = xvel[i * 3 : (i + 1) * 3]
        elif i == n_link - 1:
            vel = u - xvel[(i - 1) * 3 : i * 3]
        else:
            vel = xvel[i * 3 : (i + 1) * 3] - xvel[(i - 1) * 3 : i * 3]

        F = np.zeros(3)
        for j in range(3):
            F[j] = p["C"][i + j] * abs(vel[j]) * vel[j]

        # mass on the right
        if i < n_masses - 2:
            f[i * 3 : (i + 1) * 3] -= F

        # mass on the left
        if i > 0:
            f[(i - 1) * 3 : i * 3] += F

    # Disturbance on intermediate masses
    for i in range(n_masses - 2):
        f[i * 3 : (i + 1) * 3] += p["w"][i]

    return np.concatenate([xvel, u, f])


def _disc_f_expl(
    x: np.ndarray,
    u: np.ndarray,
    p: dict[str, np.ndarray],
    dt: float,
    fix_point: np.ndarray | None = None,
) -> np.ndarray:
    k1 = _cont_f_expl(x, u, p, fix_point)
    k2 = _cont_f_expl(x + dt / 2 * k1, u, p, fix_point)
    k3 = _cont_f_expl(x + dt / 2 * k2, u, p, fix_point)
    k4 = _cont_f_expl(x + dt * k3, u, p, fix_point)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _compute_observation_space(param: dict[str, np.ndarray]) -> spaces.Box:
    n_mass = param["m"].shape[0] + 1
    pos_max = np.array(param["L"]) * (n_mass - 1)
    pos_min = -pos_max

    vel_max = np.array([2.0, 2.0, 2.0] * (n_mass - 2))
    vel_min = -vel_max

    return spaces.Box(
        low=np.concatenate([pos_min, vel_min]),
        high=np.concatenate([pos_max, vel_max]),
        dtype=np.float32,
    )


def get_params(n_mass: int) -> dict[str, np.ndarray]:
    params = {}

    # rest length of spring
    params["L"] = np.repeat([0.033, 0.033, 0.033], n_mass - 1)

    # spring constant
    params["D"] = np.repeat([1.0, 1.0, 1.0], n_mass - 1)

    # damping constant
    params["C"] = np.repeat([0.1, 0.1, 0.1], n_mass - 1)

    # mass of the balls
    params["m"] = np.repeat([0.033], n_mass - 1)

    # disturbance on intermediate balls
    params["w"] = np.repeat([0.0, 0.0, 0.0], n_mass - 2)

    return params


class ChainEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        dt: float = 0.1,
        max_time: float = 10.0,
        render_mode: str | None = None,
        param: dict[str, np.ndarray] | None = None,
        vmax: float = 1.0,
        n_mass: int = 3,
        fix_point: np.ndarray | None = None,
        pos_last_ref: np.ndarray | None = None,
        phi_range: tuple[float, float] = (np.pi / 6, np.pi / 3),
        theta_range: tuple[float, float] = (-np.pi / 4, np.pi / 4),
    ):
        super().__init__()

        self.n_mass = n_mass

        if param is None:
            param = get_params(n_mass=n_mass)

        self.nx_pos = 3 * (n_mass - 1)
        self.nx_vel = 3 * (n_mass - 2)

        if fix_point is None:
            self.fix_point = np.array([0.0, 0.0, 0.0])
        else:
            self.fix_point = fix_point

        if pos_last_ref is None:
            self.pos_last_ref = np.array([0.033 * (n_mass - 1), 0.0, 0.0])
        else:
            self.pos_last_ref = pos_last_ref

        self.param = param
        self.structured_param = nominal_params_to_structured_nominal_params(param)

        self.observation_space = _compute_observation_space(self.param)

        self.action_space = spaces.Box(
            low=np.array([-vmax, -vmax, -vmax]),
            high=np.array([vmax, vmax, vmax]),
            dtype=np.float32,
        )

        self.dt = dt
        self.max_time = max_time

        self.trajectory = []

        self.resting_chain_solver = RestingChainSolver(
            n_mass=n_mass, fix_point=self.fix_point, f_expl=get_f_expl_expr
        )

        self.x_ref, self.u_ref = self.resting_chain_solver(p_last=self.pos_last_ref)

        self.ellipsoid = Ellipsoid(
            center=self.fix_point,
            radii=np.sum(np.array(self.structured_param["L"]), axis=0),
        )

        self.phi_range = phi_range
        self.theta_range = theta_range

        # self.ellipsoid_center = self.fix_point
        # self.ellispoid_variability_matrix = np.diag()

        self._set_canvas()

        # For rendering
        # if render_mode is not None:
        # raise NotImplementedError("Rendering is not implemented yet.")

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        u = action
        self.action = action

        self.u = u

        self.state = _disc_f_expl(
            x=self.state,
            u=self.action,
            p=self.param,
            dt=self.dt,
            fix_point=self.fix_point,
        )

        o = self._current_observation()
        r = self._calculate_reward()

        term = self._is_done()

        info = {}

        self.time += self.dt

        trunc = self.time > self.max_time

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

        return self.state, {}

    def set_state(self, state: np.ndarray):
        self.state = state

    def _current_observation(self):
        return self.state

    def _init_state_and_action(self):
        phi = self.np_random.uniform(low=self.phi_range[0], high=self.phi_range[1])  # type:ignore
        theta = self.np_random.uniform(
            low=self.theta_range[0], high=self.theta_range[1]
        )  # type:ignore
        p_last = self.ellipsoid.spherical_to_cartesian(phi=phi, theta=theta)
        x_ss, u_ss = self.resting_chain_solver(p_last=p_last)

        return x_ss, u_ss

    def _calculate_reward(self):
        pos_last = self.state[self.nx_pos - 3 : self.nx_pos]
        vel = self.state[self.nx_pos :]

        return -np.linalg.norm(
            pos_last - self.pos_last_ref, axis=0, ord=2
        ) - 0.1 * np.linalg.norm(vel, axis=0, ord=2)

    def _is_done(self):
        return bool(np.linalg.norm(self.x_ref - self.state, axis=0, ord=2) < 1e-3)

    def _set_canvas(self):
        plt.figure()
        ax = [plt.subplot(3, 1, i) for i in range(1, 4)]

        # Plot reference
        ref_pos = np.vstack([self.fix_point, self.x_ref[: self.nx_pos].reshape(-1, 3)])
        labels = ["x", "y", "z"]
        self.lines = []
        for k, ax_k in enumerate(ax):
            ax_k.plot(ref_pos[:, k], "ro--")
            ax_k.grid()
            ax_k.set_xticks(range(self.n_mass + 1))
            ax_k.set_xlim(0, self.n_mass + 1)
            ax_k.set_ylabel(labels[k])
            self.lines.append(
                ax_k.plot(range(ref_pos[:, k].shape[0]), ref_pos[:, k], ".-")[0]
            )

        self.canvas = FigureCanvas(plt.gcf())

    def render(self):
        pos = np.vstack([self.fix_point, self.state[: self.nx_pos].reshape(-1, 3)])
        for k, line in enumerate(self.lines):
            line.set_ydata(pos[:, k])
        # Convert the plot to an RGB string
        s, (width, height) = self.canvas.print_to_buffer()

        # Convert the RGB string to a NumPy array
        return np.frombuffer(s, np.uint8).reshape((height, width, 4))[:, :, :3]
