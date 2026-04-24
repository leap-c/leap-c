"""Frenet-frame race-car Gymnasium environment.

State (matches the OCP exactly): ``[s, n, alpha, v, D, delta]``.
Action: ``[derD, derDelta]`` - control rates that integrate into ``D, delta`` as part of the state.

Dynamics: explicit RK4 with the same continuous expression that the OCP uses for its
``disc_dyn_expr``; this guarantees plant-model alignment up to the integrator choice
(env: 1 RK4 step per ``dt``; OCP: ``integrate_erk4`` per stage).

See ``RaceCarEnv`` below for the full observation / action / reward / termination spec.

References:
----------
- Reiter, R., Nurkanović, A., Frey, J., Diehl, M. (2023).
  "Frenet-Cartesian model representations for automotive obstacle avoidance
  within nonlinear MPC."
  European Journal of Control, Vol. 74, 100847.
  Preprint: https://arxiv.org/abs/2212.13115
  Published: https://www.sciencedirect.com/science/article/pii/S0947358023000766
- Upstream code: ``external/acados/examples/acados_python/race_cars/``
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from leap_c.examples.race_car.bicycle_model import (
    DDELTA_MAX_DEFAULT,
    DDELTA_MIN_DEFAULT,
    DEFAULT_TRACK_FILE,
    DELTA_MAX_DEFAULT,
    DELTA_MIN_DEFAULT,
    DTHROTTLE_MAX_DEFAULT,
    DTHROTTLE_MIN_DEFAULT,
    N_MAX_DEFAULT,
    THROTTLE_MAX_DEFAULT,
    THROTTLE_MIN_DEFAULT,
    VEHICLE_PARAMS_DEFAULT,
    f_explicit_numpy_factory,
    frenet_to_cartesian,
    get_track,
)
from leap_c.examples.utils.matplotlib_env import MatplotlibRenderEnv
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx


@dataclass(kw_only=True)
class RaceCarRewardConfig:
    """Per-term weights for the race-car step reward.

    The reward is a weighted sum of six terms. All weights are non-negative; the
    sign is built into each term. Setting a weight to ``0.0`` disables that term.

    Dense (per-step):
        ``w_progress``  -> ``+ (s_new - s_prev)``               (Option A: true arc-length)
        ``w_time``      -> ``- dt``                             (explicit time cost)
        ``w_lateral``   -> ``- n**2 * dt``                      (stay near centerline)
        ``w_slip``      -> ``- (alpha + C1*delta)**2 * dt``     (discourage drift)

    One-shot (terminal step):
        ``w_bonus``     -> ``+ 1`` when the lap terminates successfully.
        ``w_violation`` -> ``- 1`` when the episode truncates off-track.

    NOTE: Every dense term scales linearly with ``dt``, rescale weights if ``dt`` changes

    Factory methods reproduce the wiki's documented options:
        ``.progress()``   -> Option A (default): ``ds`` per step.
        ``.lap_time(K)``  -> Option B: ``-dt`` per step plus ``+K`` on lap complete.
        ``.hybrid(c,K,Kv)`` -> Option C: ``ds - c*dt`` plus ``+K`` / ``-Kv`` terminals.
    """

    w_progress: float = 1.0
    w_time: float = 0.0
    w_bonus: float = 0.0
    w_violation: float = 0.0
    w_lateral: float = 0.0
    w_slip: float = 0.0

    @classmethod
    def progress(cls) -> "RaceCarRewardConfig":
        """Option A: reward is the actual arc-length progression per step."""
        return cls()

    @classmethod
    def lap_time(cls, bonus: float = 100.0) -> "RaceCarRewardConfig":
        """Option B: sparse lap-time reward (``-dt`` + ``+bonus`` on lap complete)."""
        return cls(w_progress=0.0, w_time=1.0, w_bonus=bonus)

    @classmethod
    def hybrid(
        cls,
        c: float = 0.1,
        bonus: float = 100.0,
        violation: float = 50.0,
    ) -> "RaceCarRewardConfig":
        """Option C: ``ds - c*dt`` plus terminal bonus and off-track penalty."""
        return cls(w_time=c, w_bonus=bonus, w_violation=violation)


@dataclass(kw_only=True)
class RaceCarEnvConfig:
    """Configuration for the race-car environment.

    Bounds mirror the acados example and Reiter et al. (2023). ``init_state = [-2, 0, 0, 0, 0, 0]``
    starts the car 2 m before the start line; the curvature spline is extended backwards to keep
    this region well-defined.

    Attributes:
        track_file: Path to a whitespace-separated track file with columns
            ``(s, x, y, psi, kappa)``.
        dt: Simulation step size [s].
        n_max: Track half-width [m]; ``|n| > n_max + n_violation_margin`` truncates the episode.
        n_violation_margin: Extra tolerance [m] beyond ``n_max`` before the off-track flag fires.
        vehicle_params: Bicycle-model parameters (mass, cornering/motor/rolling coefficients).
        init_state: Initial ``[s, n, alpha, v, D, delta]`` at every ``reset``.
        throttle_min, throttle_max: Hard saturation on ``D`` (throttle / duty cycle, dimensionless).
        delta_min, delta_max: Hard saturation on ``delta`` (steering angle) [rad].
        dthrottle_min, dthrottle_max: Action bounds on ``derD`` (throttle rate) [1/s].
        ddelta_min, ddelta_max: Action bounds on ``derDelta`` (steering rate) [rad/s].
        max_steps: Truncation step budget per episode.
        reward: ``RaceCarRewardConfig`` selecting per-term weights for the step reward.
            Defaults to Option A (arc-length progression).
    """

    track_file: Path = field(default_factory=lambda: DEFAULT_TRACK_FILE)
    dt: float = 0.02
    n_max: float = N_MAX_DEFAULT
    n_violation_margin: float = 0.05
    vehicle_params: dict[str, float] = field(default_factory=lambda: dict(VEHICLE_PARAMS_DEFAULT))
    init_state: np.ndarray = field(
        default_factory=lambda: np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    throttle_min: float = THROTTLE_MIN_DEFAULT
    throttle_max: float = THROTTLE_MAX_DEFAULT
    delta_min: float = DELTA_MIN_DEFAULT
    delta_max: float = DELTA_MAX_DEFAULT
    dthrottle_min: float = DTHROTTLE_MIN_DEFAULT
    dthrottle_max: float = DTHROTTLE_MAX_DEFAULT
    ddelta_min: float = DDELTA_MIN_DEFAULT
    ddelta_max: float = DDELTA_MAX_DEFAULT
    max_steps: int = 4000
    reward: RaceCarRewardConfig = field(default_factory=RaceCarRewardConfig)


class RaceCarEnv(MatplotlibRenderEnv[np.ndarray, np.ndarray, AcadosDiffMpcCtx]):
    """One-lap Frenet-frame race-car environment matching the acados race_cars example.

    The state and observation are identical: a 6-vector ``[s, n, alpha, v, D, delta]``
    expressed in the Frenet frame aligned with the track centerline. The action is a
    2-vector of control *rates* ``[derD, derDelta]`` that integrate into the ``D``
    and ``delta`` components of the state via the bicycle model.

    Observation Space:
    ------------------
    The observation is a ``np.ndarray`` with shape ``(6,)`` and dtype ``np.float64``.
    Bounds are deliberately looser than the OCP / saturation limits so that
    observations outside the safe set remain valid under the Gymnasium spec.

    | Num | Observation                              | Min    | Max    | Units |
    |-----|------------------------------------------|--------|--------|-------|
    | 0   | s     (arc length along centerline)      | -inf   | +inf   | m     |
    | 1   | n     (lateral offset from centerline)   | -1.0   |  1.0   | m     |
    | 2   | alpha (heading offset from tangent)      | -pi    |  pi    | rad   |
    | 3   | v     (longitudinal velocity)            | -inf   | +inf   | m/s   |
    | 4   | D     (throttle / duty cycle)            | -1.5   |  1.5   | -     |
    | 5   | delta (steering angle)                   | -0.6   |  0.6   | rad   |

    NOTE: ``step`` enforces the *physical* limits every step by saturating
    ``D in [throttle_min, throttle_max]`` and ``delta in [delta_min, delta_max]``
    (defaults ``[-1, 1]`` and ``[-0.40, 0.40]`` rad, matching the OCP hard path
    constraints). The observation-space bounds are wider on purpose.

    NOTE: ``s`` starts at ``init_state[0]`` (default ``-2.0``, i.e. 2 m before the
    start line) and grows monotonically; the episode terminates when ``s`` exceeds
    the track length.

    Action Space:
    -------------
    The action is a ``np.ndarray`` with shape ``(2,)`` of control rates:

    | Num | Action                        | Min    | Max    | Units  |
    |-----|-------------------------------|--------|--------|--------|
    | 0   | derD     (throttle rate)      | -10.0  |  10.0  | 1/s    |
    | 1   | derDelta (steering rate)      |  -2.0  |   2.0  | rad/s  |

    Actions are clipped to the action-space bounds inside ``step`` and then integrated
    via one RK4 step of the same continuous dynamics that the OCP uses for its
    ``disc_dyn_expr``, ensuring plant-model alignment.

    Reward:
    -------
    The reward is a weighted sum over ``cfg.reward`` (see ``RaceCarRewardConfig``).
    Default (``progress``) is the arc-length progression ``s_new - s_prev`` per step.
    Other presets: ``lap_time`` (sparse Option B), ``hybrid`` (dense Option C with
    terminal bonus and off-track penalty).
    Individual term weights can be mixed freely for ablation sweeps.

    Terminates:
    -----------
    When ``s > pathlength`` (lap completed, ``info['task']['success'] == True``).

    Truncates:
    ----------
    - When ``|n| > n_max + n_violation_margin`` (off-track,
      ``info['task']['violation'] == True``).
    - When the step counter reaches ``cfg.max_steps``.

    Info:
    -----
    - ``task.violation`` (bool): off-track flag; cause of truncation.
    - ``task.success`` (bool): lap-completed flag; cause of termination.
    - ``step`` (int): current step index within the episode.

    Attributes:
        cfg: ``RaceCarEnvConfig`` with bounds, timestep, vehicle params, init state.
        integrator: RK4 integrator ``(x, u, dt) -> x_next`` from the same continuous
            dynamics as the OCP.
        x: Current state, or ``None`` before the first ``reset``.
        x_trajectory: List of past states recorded since the last ``reset``.
        observation_space: ``gym.spaces.Box`` described above.
        action_space: ``gym.spaces.Box`` described above.
        render_mode: ``"human"`` / ``"rgb_array"`` / ``None``.
        ctx: Last planner ``AcadosDiffMpcCtx`` (used to overlay the MPC plan when rendering).

    References:
        See module docstring for the Reiter et al. (2023) citation.
    """

    cfg: RaceCarEnvConfig
    integrator: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    x: np.ndarray | None
    x_trajectory: list[np.ndarray]

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = None,
        cfg: RaceCarEnvConfig | None = None,
    ) -> None:
        super().__init__(render_mode=render_mode)
        self.cfg = RaceCarEnvConfig() if cfg is None else cfg

        self._sref, self._xref, self._yref, self._psiref, _ = get_track(self.cfg.track_file)
        self._pathlength = float(self._sref[-1])
        self._c1 = float(self.cfg.vehicle_params["C1"])

        f_numpy = f_explicit_numpy_factory(self.cfg.track_file, self.cfg.vehicle_params)

        def rk4_step(f, x, u, h):
            k1 = f(x, u)
            k2 = f(x + 0.5 * h * k1, u)
            k3 = f(x + 0.5 * h * k2, u)
            k4 = f(x + h * k3, u)
            return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        self.integrator = lambda x, u, t: rk4_step(f_numpy, x, u, t)

        # Loose observation bounds — s and v grow during a lap; n and angles are loosely bounded.
        obs_high = np.array(
            [np.inf, 1.0, np.pi, np.inf, 1.5, 0.6],
            dtype=np.float64,
        )
        obs_low = -obs_high
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=np.array([self.cfg.dthrottle_min, self.cfg.ddelta_min], dtype=np.float64),
            high=np.array([self.cfg.dthrottle_max, self.cfg.ddelta_max], dtype=np.float64),
            dtype=np.float64,
        )

        self.x = None
        self.x_trajectory = []
        self._step = 0
        self._reset_needed = True

        # Rendering placeholders
        self._car_dot = None
        self._traj_scatter = None
        self._plan_line = None
        self.ctx: AcadosDiffMpcCtx | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        self.x = self.cfg.init_state.copy()
        self.x_trajectory = [self.x.copy()]
        self._step = 0
        self._reset_needed = False
        return self.x.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._reset_needed:
            raise RuntimeError("Call reset() before step().")

        action = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )

        s_prev = float(self.x[0])
        self.x = self.integrator(self.x, action, self.cfg.dt)
        # Saturate D and delta to mirror the OCP's hard box constraints on those states.
        self.x[4] = float(np.clip(self.x[4], self.cfg.throttle_min, self.cfg.throttle_max))
        self.x[5] = float(np.clip(self.x[5], self.cfg.delta_min, self.cfg.delta_max))
        self.x_trajectory.append(self.x.copy())
        self._step += 1

        s, n, alpha, _, _, delta = self.x
        dt = self.cfg.dt

        terminated = bool(s > self._pathlength)
        off_track = bool(abs(n) > self.cfg.n_max + self.cfg.n_violation_margin)
        truncated = bool(off_track or self._step >= self.cfg.max_steps)

        rw = self.cfg.reward
        slip = alpha + self._c1 * delta
        reward = (
            rw.w_progress * (s - s_prev)
            - rw.w_time * dt
            - rw.w_lateral * (n * n * dt)
            - rw.w_slip * (slip * slip * dt)
        )
        if terminated:
            reward += rw.w_bonus
        if off_track:
            reward -= rw.w_violation
        reward = float(reward)

        info: dict[str, Any] = {
            "task": {
                "violation": off_track,
                "success": terminated,
            },
            "step": self._step,
        }
        self._reset_needed = terminated or truncated
        return self.x.copy(), reward, terminated, truncated, info

    def _render_setup(self) -> None:
        import matplotlib.pyplot as plt

        self._fig, self._ax = plt.subplots(figsize=(8, 8))
        nx = -np.sin(self._psiref)
        ny = np.cos(self._psiref)
        inner_x = self._xref + self.cfg.n_max * nx
        inner_y = self._yref + self.cfg.n_max * ny
        outer_x = self._xref - self.cfg.n_max * nx
        outer_y = self._yref - self.cfg.n_max * ny
        self._ax.plot(self._xref, self._yref, "k--", lw=0.4, label="centerline")
        self._ax.plot(inner_x, inner_y, "k-", lw=0.6)
        self._ax.plot(outer_x, outer_y, "k-", lw=0.6)
        self._ax.set_aspect("equal")
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")
        self._ax.set_title("Race car (Frenet) - Cartesian view")

        # Arc-length markers every 1 m, matching external/acados plotFcn.plotTrackProj.
        for i in range(int(self._pathlength) + 1):
            k = int(np.argmin(np.abs(self._sref - i)))
            s_k = float(self._sref[k])
            lx, ly, _ = frenet_to_cartesian(
                s_k,
                self.cfg.n_max + 0.06,
                self._sref,
                self._xref,
                self._yref,
                self._psiref,
            )
            tin_x, tin_y, _ = frenet_to_cartesian(
                s_k,
                self.cfg.n_max,
                self._sref,
                self._xref,
                self._yref,
                self._psiref,
            )
            tout_x, tout_y, _ = frenet_to_cartesian(
                s_k,
                -self.cfg.n_max,
                self._sref,
                self._xref,
                self._yref,
                self._psiref,
            )
            self._ax.plot(
                [np.asarray(tin_x).item(), np.asarray(tout_x).item()],
                [np.asarray(tin_y).item(), np.asarray(tout_y).item()],
                "k-",
                lw=0.3,
                alpha=0.5,
            )
            self._ax.text(
                np.asarray(lx).item(),
                np.asarray(ly).item(),
                f"{i}m",
                fontsize=6,
                ha="center",
                va="center",
                alpha=0.7,
            )

        # Velocity-colored trajectory scatter
        # vmin/vmax are fixed so the colorbar doesn't jump between video frames.
        self._traj_scatter = self._ax.scatter(
            [],
            [],
            c=[],
            cmap="rainbow",
            s=4,
            edgecolor="none",
            vmin=0.0,
            vmax=2.0,
            label="trajectory",
        )
        self._cbar = self._fig.colorbar(self._traj_scatter, ax=self._ax, fraction=0.035)
        self._cbar.set_label("v [m/s]")

        (self._plan_line,) = self._ax.plot([], [], "g--", lw=1.0, alpha=0.6, label="MPC plan")
        (self._car_dot,) = self._ax.plot([], [], "ro", ms=8, label="car")
        self._ax.legend(loc="upper right", fontsize=8)

    def _render_frame(self) -> None:
        if self.x is None:
            return

        traj_arr = np.asarray(self.x_trajectory)
        x_traj, y_traj, _ = frenet_to_cartesian(
            traj_arr[:, 0], traj_arr[:, 1], self._sref, self._xref, self._yref, self._psiref
        )
        v_traj = traj_arr[:, 3]
        self._traj_scatter.set_offsets(np.c_[x_traj, y_traj])
        self._traj_scatter.set_array(v_traj)

        car_x, car_y, _ = frenet_to_cartesian(
            self.x[0], self.x[1], self._sref, self._xref, self._yref, self._psiref
        )
        self._car_dot.set_data(car_x, car_y)

        if self.ctx is not None and getattr(self.ctx, "iterate", None) is not None:
            try:
                x_plan = np.asarray(self.ctx.iterate.x[0]).reshape(-1, 6)
            except Exception:
                x_plan = None
            if x_plan is not None:
                px, py, _ = frenet_to_cartesian(
                    x_plan[:, 0],
                    x_plan[:, 1],
                    self._sref,
                    self._xref,
                    self._yref,
                    self._psiref,
                )
                self._plan_line.set_data(px, py)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = RaceCarEnv(render_mode="human")
    obs, _ = env.reset(seed=0)
    print("init obs:", obs)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(np.array([5.0, 0.0]))
        if term or trunc:
            print("terminated/truncated", info)
            break
    print("final obs:", obs)
    env.render()
    plt.show()
