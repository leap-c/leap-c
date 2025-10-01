from dataclasses import dataclass
from typing import Optional, Union, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import cm

from leap_c.examples.utils.matplotlib_env import MatplotlibRenderEnv

from leap_c.examples.race_cars.tracks.readDataFcn import getTrack
from leap_c.examples.race_cars.time2spatial import transformProj2Orig, transformOrig2Proj
from leap_c.examples.race_cars.plotFcn import plotTrackProj, plotRes, plotalat

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
    """
    Spatial bicycle model environment:
    - State: x = [s, n, alpha, v, D, delta]
      s: station along centerline [m]
      n: lateral deviation [m]
      alpha: heading error wrt centerline [rad]
      v: speed [m/s]
      D: "throttle" state (acts through Fxd)
      delta: steering angle [rad]
    - Action: u = [derD, derDelta] (rates)
    - Observation: state itself (float32)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}


    def __init__(
        self,
        render_mode: Optional[str] = None,
        cfg: Optional[RaceCarEnvConfig] = None,
        kappa_ref: Optional[Callable] = None,  # function kappa_ref(s) -> float
    ):
        super().__init__(render_mode=render_mode)
        self.cfg = cfg if cfg is not None else RaceCarEnvConfig()

        if kappa_ref is not None:
            self.kappa_ref = kappa_ref
        else:
            track_filename = 'LMS_Track.txt'
            [s_ref, _, _, _, kappa_ref_data] = getTrack(track_filename)
            self.kappa_ref = lambda s: np.interp(s, s_ref, kappa_ref_data)

        # action
        self.action_space = spaces.Box(
            low=np.array([self.cfg.dthrottle_min, self.cfg.ddelta_min], dtype=np.float32),
            high=np.array([self.cfg.dthrottle_max, self.cfg.ddelta_max], dtype=np.float32),
            dtype=np.float32,
        )

        # observation (s, n, alpha, v, D, delta)
        high = np.array(
            [
                np.finfo(np.float32).max,
                max(abs(self.cfg.n_min), abs(self.cfg.n_max)),
                2 * np.pi,
                10.0,
                max(abs(self.cfg.throttle_min), abs(self.cfg.throttle_max)),
                max(abs(self.cfg.delta_min), abs(self.cfg.delta_max)),
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.integrator = lambda x, u, h: self._rk4(self.f_explicit, x, u, h)

        self.t = 0
        self.x = None

    def _Fxd(self, v, D):
        return (self.cfg.Cm1 - self.cfg.Cm2 * v) * D - self.cfg.Cr2 * v * v - self.cfg.Cr0 * np.tanh(5.0 * v)

    def f_explicit(self, x, u):
        s, n, alpha, v, D, delta = x
        derD, derDelta = u
        kappa = self.kappa_ref(s)

        Fxd = self._Fxd(v, D)
        sdota = (v * np.cos(alpha + self.cfg.C1 * delta)) / (1.0 - kappa * n)

        sdot = sdota
        ndot = v * np.sin(alpha + self.cfg.C1 * delta)
        alphadot = self.cfg.C2 * v * delta - kappa * sdota
        vdot = (Fxd / self.cfg.m) * np.cos(self.cfg.C1 * delta)
        Ddot = derD
        deltadot = derDelta

        return np.array([sdot, ndot, alphadot, vdot, Ddot, deltadot], dtype=np.float32)

    def _rk4(self, f, x, u, h):
        k1 = f(x, u)
        k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u)
        k4 = f(x + h * k3, u)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

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
        s, n, alpha, v, D, delta = x
        v_ref = 2.0 # m/s
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        r_long = -(v - v_ref)**2
        r_lat = -(n**2 + alpha**2) * 3
        r_total = r_long + r_lat
        return float(r_total)

    def init_state(self, options: Optional[dict] = None) -> np.ndarray:
        return np.array([-2, 0, 0, 0, 0, 0], dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        self.t = 0.0
        self.x = self.init_state(options)

        self.x_trajectory = []
        self.pos_trajectory = None
        self.u_trajectory = []  # Store control inputs for plotting

        # Reset visualization trajectory data
        self._vis_x_traj = []
        self._vis_y_traj = []
        self._vis_velocities = []
        self._track_filename = 'LMS_Track.txt'

        return self.x.copy(), {}

    def _render_setup(self):
        """Setup the matplotlib figure for real-time rendering."""
        self._fig, self._ax = plt.subplots(figsize=(10, 8))

        [Sref, Xref, Yref, Psiref, _] = getTrack(self._track_filename)
        distance = 0.12  # track width

        # Plot center line
        self._ax.plot(Xref, Yref, '--', color='k', label='Center line')

        # Draw track boundaries
        Xboundleft = Xref - distance * np.sin(Psiref)
        Yboundleft = Yref + distance * np.cos(Psiref)
        Xboundright = Xref + distance * np.sin(Psiref)
        Yboundright = Yref - distance * np.cos(Psiref)
        self._ax.plot(Xboundleft, Yboundleft, color='k', linewidth=1)
        self._ax.plot(Xboundright, Yboundright, color='k', linewidth=1)

        # Set plot limits and labels
        self._ax.set_ylim(bottom=-1.75, top=0.35)
        self._ax.set_xlim(left=-1.1, right=1.6)
        self._ax.set_ylabel('y[m]')
        self._ax.set_xlabel('x[m]')
        self._ax.set_aspect('equal', 'box')
        self._ax.grid(True)

    def _render_frame(self):
        if self.x is None:
            return

        s, n, alpha, v = self.x[0], self.x[1], self.x[2], self.x[3]
        [x_pos, y_pos, _, _] = transformProj2Orig(s, n, alpha, v, self._track_filename)

        # Store trajectory data
        self._vis_x_traj.append(x_pos)
        self._vis_y_traj.append(y_pos)
        self._vis_velocities.append(v)

        # Clear previous car position
        for artist in self._ax.collections:
            if hasattr(artist, '_car_marker'):
                artist.remove()

        # Plot trajectory with velocity colormap
        if len(self._vis_x_traj) > 1:
            scatter = self._ax.scatter(self._vis_x_traj, self._vis_y_traj,
                                        c=self._vis_velocities, cmap=cm.rainbow,
                                        edgecolor='none', marker='o', s=20)

        # Plot current car position
        car_scatter = self._ax.scatter([x_pos], [y_pos], c='red', marker='o', s=100,
                                        edgecolor='black', linewidth=2, zorder=10)
        car_scatter._car_marker = True  # Mark for removal next frame


    def plot_trajectory(self, filename='LMS_Track.txt'):
        """Plot the complete trajectory on track using acados visualization."""
        if not self.x_trajectory:
            print("No trajectory data available. Run simulation first.")
            return

        simX = np.array(self.x_trajectory)
        plotTrackProj(simX, filename)
        plt.title('Race Car Trajectory')
        plt.show()

    def plot_results(self, simU=None):
        """Plot state and control trajectories over time."""
        if not self.x_trajectory:
            print("No trajectory data available. Run simulation first.")
            return

        simX = np.array(self.x_trajectory)
        if simU is None:
            if hasattr(self, 'u_trajectory') and self.u_trajectory:
                simU = np.array(self.u_trajectory)
            else:
                # Create dummy control inputs if not provided
                simU = np.zeros((len(simX), 2))

        t = np.arange(len(simX)) * self.cfg.dt
        plotRes(simX, simU, t)
        plt.suptitle('Race Car Simulation Results')
        plt.show()

    def plot_acceleration(self, simU=None):
        """Plot lateral acceleration using acados plotalat style."""
        if not hasattr(self, 'x_trajectory') or len(self.x_trajectory) == 0:
            print("No trajectory data available. Run simulation first.")
            return

        simX = np.array(self.x_trajectory)
        if simU is None:
            if hasattr(self, 'u_trajectory') and self.u_trajectory:
                simU = np.array(self.u_trajectory)
            else:
                # Create dummy control inputs if not provided
                simU = np.zeros((len(simX), 2))

        # Create constraint object for acceleration calculation
        from types import SimpleNamespace
        constraint = SimpleNamespace()

        def alat_func(x, u):
            s, n, alpha, v, D, delta = x
            derD, derDelta = u
            Fxd = self._Fxd(v, D)
            return self.cfg.C2 * v * v * delta + Fxd * np.sin(self.cfg.C1 * delta) / self.cfg.m

        constraint.alat = alat_func
        constraint.alat_min = -4.0
        constraint.alat_max = 4.0

        t = np.arange(len(simX)) * self.cfg.dt
        plotalat(simX, simU, constraint, t)
        plt.title('Lateral Acceleration')
        plt.show()

    def render_track_overview(self):
        """Render a static overview of the track with current trajectory."""
        if not self.x_trajectory:
            print("No trajectory data available. Run simulation first.")
            return

        # Use the existing plot_trajectory method but with current data
        simX = np.array(self.x_trajectory)
        plt.figure(figsize=(12, 8))
        plotTrackProj(simX, self._track_filename)
        plt.title('Race Car Environment - Track Overview')
        plt.show()