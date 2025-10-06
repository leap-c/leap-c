from dataclasses import dataclass
from typing import Optional
from types import SimpleNamespace


import matplotlib.pyplot as plt
from matplotlib import cm
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.interpolate import CubicSpline

from leap_c.examples.utils.matplotlib_env import MatplotlibRenderEnv
from leap_c.examples.race_car.track import Track
from leap_c.examples.race_car.plotFcn import plotTrackProj, plotRes, plotalat

@dataclass(kw_only=True)
class RaceCarEnvConfig:
    m: float = 0.043
    C1: float = 0.5
    C2: float = 15.5
    Cm1: float = 0.28
    Cm2: float = 0.05
    Cr0: float = 0.011
    Cr2: float = 0.006

    # integration / episode
    dt: float = 0.02
    max_time: float = 20.0

    # track properties
    n_min: float = -0.12
    n_max: float = 0.12
    s_init: float = 0.0
    # input/state bounds (rate inputs!)
    ddelta_min: float = -2.0
    ddelta_max: float = 2.0
    dthrottle_min: float = -10.0
    dthrottle_max: float = 10.0

    delta_min: float = -0.40
    delta_max: float = 0.40
    throttle_min: float = -1.0
    throttle_max: float = 1.0

class RaceCarEnv(MatplotlibRenderEnv, gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        cfg: Optional[RaceCarEnvConfig] = None,
        track_file: Optional[str] = None,
    ):
        super().__init__(render_mode=render_mode)
        self.cfg = cfg if cfg is not None else RaceCarEnvConfig()
        self.track_file = track_file if track_file is not None else "LMS_Track.txt"
        self.track = Track(self.track_file)

        self.kappa_interp = CubicSpline(
            self.track.thetaref,
            self.track.kapparef,
            bc_type='periodic'
        )

        # action
        self.action_space = spaces.Box(
            low=np.array([self.cfg.ddelta_min, self.cfg.dthrottle_min], dtype=np.float32),
            high=np.array([self.cfg.ddelta_max, self.cfg.dthrottle_max], dtype=np.float32),
            dtype=np.float32,
        )

        high = np.array(
            [
                np.finfo(np.float32).max,
                max(abs(self.cfg.n_min), abs(self.cfg.n_max)),
                2 * np.pi,
                10.0,
                max(abs(self.cfg.delta_min), abs(self.cfg.delta_max)),
                max(abs(self.cfg.throttle_min), abs(self.cfg.throttle_max)),
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)


        self.integrator = lambda x, u, h: self.rk4_step(x, u, h)

    def rk4_step(self, x, u, h):
        k1 = self.f_explicit(x, u)
        k2 = self.f_explicit(x + 0.5 * h * k1, u)
        k3 = self.f_explicit(x + 0.5 * h * k2, u)
        k4 = self.f_explicit(x + h * k3, u)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _Fxd(self, v, D):
        return (self.cfg.Cm1 - self.cfg.Cm2 * v) * D - self.cfg.Cr2 * v * v - self.cfg.Cr0 * np.tanh(5.0 * v)

    def f_explicit(self, x, u):
        s, n, alpha, v, delta, D = x
        ddelta, dD = u
        kappa = self.kappa_interp(s % self.track.thetaref[-1])

        Fxd = self._Fxd(v, D)
        sdota = (v * np.cos(alpha + self.cfg.C1 * delta)) / (1.0 - kappa * n)

        sdot = sdota
        ndot = v * np.sin(alpha + self.cfg.C1 * delta)
        alphadot = self.cfg.C2 * v * delta - kappa * sdota
        vdot = (Fxd / self.cfg.m) * np.cos(self.cfg.C1 * delta)
        deltadot = ddelta
        Ddot = dD

        return np.array([sdot, ndot, alphadot, vdot, deltadot, Ddot], dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.x = self.integrator(self.x, action, self.cfg.dt)
        self.x_trajectory.append(self.x.copy())
        self.u_trajectory.append(action.copy())
        self.t += self.cfg.dt

        r = self._get_reward(self.x, action)

        term = False; trunc = False; info = {}
        if self.t >= self.cfg.max_time:
            trunc = True
        if not (self.cfg.n_min <= self.x[1] <= self.cfg.n_max):
            term = True

        return self.x, r, term, trunc, info

    def _get_reward(self, x, u):
        s, n, alpha, v, delta, D = x
        v_ref = 2.0 # m/s
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        r_long = -(v - v_ref)**2
        r_lat = -(n**2 + alpha**2) * 3
        r_total = r_long + r_lat
        return float(r_total)

    def init_state(self, options: Optional[dict] = None) -> np.ndarray:
        return np.array([-2, 0, 0, 0, 0, 0], dtype=np.float32)  # Start with v=0.5 m/s

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        self.t = 0.0
        self.x = self.init_state(options)

        self.x_trajectory, self.u_trajectory = [], []
        self.pos_trajectory = None

        # Reset visualization trajectory data
        self._vis_x_traj, self._vis_y_traj, self._vis_velocities = [], [], []
        self._track_filename = self.track_file

        return self.x.copy(), {}

    def _render_setup(self):
        self._fig, self._ax = plt.subplots(figsize=(10, 8))

        Xref, Yref, Psiref = self.track.Xref, self.track.Yref, self.track.psiref
        distance = 0.12

        self._ax.plot(Xref, Yref, '--', color='k', label='Center line')

        Xboundleft = Xref - distance * np.sin(Psiref)
        Yboundleft = Yref + distance * np.cos(Psiref)
        Xboundright = Xref + distance * np.sin(Psiref)
        Yboundright = Yref - distance * np.cos(Psiref)
        self._ax.plot(Xboundleft, Yboundleft, color='k', linewidth=1)
        self._ax.plot(Xboundright, Yboundright, color='k', linewidth=1)

        self._ax.set_xlim(-1.1, 1.6); self._ax.set_ylim(-1.75, 0.35)
        self._ax.set_xlabel('x[m]'); self._ax.set_ylabel('y[m]')
        self._ax.set_aspect('equal', 'box')
        self._ax.grid(True)

    def _render_frame(self):
        if self.x is None:
            return

        s, n, alpha, v, delta, D = self.x

        s_wrapped = s % self.track.thetaref[-1]
        XY = self.track.get_XY(s_wrapped)
        dXY = self.track.get_XY(s_wrapped, 1)
        psi0 = np.arctan2(dXY[1], dXY[0])

        x_pos = XY[0] - n * np.sin(psi0)
        y_pos = XY[1] + n * np.cos(psi0)

        self._vis_x_traj.append(x_pos)
        self._vis_y_traj.append(y_pos)
        self._vis_velocities.append(v)

        [artist.remove() for artist in self._ax.collections if hasattr(artist, '_car_marker')]
        [text.remove() for text in self._ax.texts if hasattr(text, '_info_text')]

        # Plot trajectory
        len(self._vis_x_traj) > 1 and self._ax.scatter(
            self._vis_x_traj, self._vis_y_traj, c=self._vis_velocities,
            cmap=cm.rainbow, edgecolor='none', marker='o', s=20
        )

        # Plot car
        car_scatter = self._ax.scatter([x_pos], [y_pos], c='red', marker='o',
                                        s=100, edgecolor='black', linewidth=2, zorder=10)
        car_scatter._car_marker = True

        # Add info text
        info_text = (
            f'Time: {self.t:.2f}s\n'
            f'Progress: {s:.2f}m\n'
            f'Speed: {v:.2f}m/s\n'
            f'Lateral dev: {n:.3f}m\n'
            f'Heading err: {np.rad2deg(alpha):.1f}°\n'
            f'Steering: {np.rad2deg(delta):.1f}°\n'
            f'Throttle: {D:.2f}'
        )

        text_obj = self._ax.text(
            0.02, 0.98, info_text,
            transform=self._ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10,
            family='monospace'
        )
        text_obj._info_text = True


    def plot_trajectory(self, filename='LMS_Track.txt'):
        if not self.x_trajectory:
            return print("No trajectory data available. Run simulation first.")

        plotTrackProj(np.array(self.x_trajectory), filename)
        plt.title('Race Car Trajectory')
        plt.show()

    def plot_results(self, simU=None):
        if not self.x_trajectory:
            return print("No trajectory data available. Run simulation first.")

        simX = np.array(self.x_trajectory)
        simU = simU or (np.array(self.u_trajectory) if getattr(self, 'u_trajectory', None) else np.zeros((len(simX), 2)))

        plotRes(simX, simU, np.arange(len(simX)) * self.cfg.dt)
        plt.suptitle('Race Car Simulation Results')
        plt.show()

    def plot_acceleration(self, simU=None):
        if not getattr(self, 'x_trajectory', None):
            return print("No trajectory data available. Run simulation first.")

        simX = np.array(self.x_trajectory)
        simU = simU or (np.array(self.u_trajectory) if getattr(self, 'u_trajectory', None) else np.zeros((len(simX), 2)))

        constraint = SimpleNamespace(
            alat=lambda x, u: self.cfg.C2 * x[3]**2 * x[4] + self._Fxd(x[3], x[5]) * np.sin(self.cfg.C1 * x[4]) / self.cfg.m,
            alat_min=-4.0,
            alat_max=4.0
        )

        plotalat(simX, simU, constraint, np.arange(len(simX)) * self.cfg.dt)
        plt.title('Lateral Acceleration')
        plt.show()

    def render_track_overview(self):
        if not self.x_trajectory:
            return print("No trajectory data available. Run simulation first.")

        plt.figure(figsize=(12, 8))
        plotTrackProj(np.array(self.x_trajectory), self._track_filename)
        plt.title('Race Car Environment - Track Overview')
        plt.show()