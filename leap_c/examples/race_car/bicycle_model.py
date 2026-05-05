"""Spatial (Frenet-frame) bicycle model adapted from the acados race_cars example.

Source:
    external/acados/examples/acados_python/race_cars/{bicycle_model.py, tracks/readDataFcn.py}

Differences vs the source:
    - Uses ca.SX symbols (leap_c convention) instead of ca.MX.
    - Splits the monolithic ``bicycle_model`` factory into composable functions so the
      env can build a numpy callable from the same continuous dynamics that the OCP
      uses, mirroring the cartpole pattern.

State / control:
    x = [s, n, alpha, v, D, delta]    (nx = 6)
    u = [derD, derDelta]              (nu = 2)
    - s     : arc length along the track centerline [m]
    - n     : lateral offset from centerline [m]
    - alpha : heading offset from centerline tangent [rad]
    - v     : longitudinal velocity [m/s]
    - D     : throttle / duty cycle [-]
    - delta : steering angle [rad]

Default constants (match the acados example):

    | Name                    | Value | Units  | Meaning                                    |
    |-------------------------|-------|--------|--------------------------------------------|
    | N_MAX_DEFAULT           |  0.12 | m      | max lateral offset (track half-width)      |
    | THROTTLE_MIN/MAX        | -/+1  | -      | throttle / duty-cycle box constraint       |
    | DELTA_MIN/MAX           | -/+0.40 | rad  | steering angle box constraint              |
    | DTHROTTLE_MIN/MAX       | -/+10 | 1/s    | throttle-rate hard box constraint (on u)   |
    | DDELTA_MIN/MAX          | -/+2  | rad/s  | steering-rate hard box constraint (on u)   |
    | ALAT_MAX_DEFAULT        |  4.0  | m/s^2  | lateral accel path constraint              |
    | ALONG_MAX_DEFAULT       |  4.0  | m/s^2  | longitudinal accel path constraint (soft)  |

``VEHICLE_PARAMS_DEFAULT`` holds the bicycle-model mass, cornering factors (C1, C2),
motor coefficients (Cm1, Cm2), and rolling resistance terms (Cr0, Cr2); values are
taken from the acados example implementation.

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
from pathlib import Path

import casadi as ca
import numpy as np

ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_TRACK_FILE = ASSETS_DIR / "LMS_Track.txt"

VEHICLE_PARAMS_DEFAULT: dict[str, float] = {
    "m": 0.043,
    "C1": 0.5,
    "C2": 15.5,
    "Cm1": 0.28,
    "Cm2": 0.05,
    "Cr0": 0.011,
    "Cr2": 0.006,
}

N_MAX_DEFAULT = 0.12
THROTTLE_MIN_DEFAULT = -1.0
THROTTLE_MAX_DEFAULT = 1.0
DELTA_MIN_DEFAULT = -0.40
DELTA_MAX_DEFAULT = 0.40
DTHROTTLE_MIN_DEFAULT = -10.0
DTHROTTLE_MAX_DEFAULT = 10.0
DDELTA_MIN_DEFAULT = -2.0
DDELTA_MAX_DEFAULT = 2.0
ALAT_MAX_DEFAULT = 4.0
ALONG_MAX_DEFAULT = 4.0


def get_track(track_file: Path = DEFAULT_TRACK_FILE) -> tuple[np.ndarray, ...]:
    """Load a track reference file with columns ``(s, x, y, psi, kappa)``."""
    array = np.loadtxt(track_file)
    return array[:, 0], array[:, 1], array[:, 2], array[:, 3], array[:, 4]


def build_curvature_spline(track_file: Path = DEFAULT_TRACK_FILE) -> tuple[ca.Function, float]:
    """Build a CasADi B-spline curvature interpolant ``kappa(s)`` and return the lap length.

    Mirrors the wrap-around extension trick from the acados example so the spline is
    well-defined for ``s`` values slightly before the start line and beyond a single
    lap, both of which arise during MPC lookahead.
    """
    s0, _, _, _, kapparef = get_track(track_file)
    length = len(s0)
    pathlength = float(s0[-1])

    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    kapparef = np.append(kapparef, kapparef[1:length])
    s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)

    kapparef_s = ca.interpolant("kapparef_s", "bspline", [s0], kapparef)
    return kapparef_s, pathlength


def define_f_expl_expr(
    x: ca.SX,
    u: ca.SX,
    kapparef_s: ca.Function,
    vp: dict[str, float] | None = None,
) -> ca.SX:
    """Continuous Frenet-frame bicycle dynamics. Returns the 6-vector ``dx/dt`` as a CasADi expr."""
    vp = VEHICLE_PARAMS_DEFAULT if vp is None else vp
    m, C1, C2 = vp["m"], vp["C1"], vp["C2"]
    Cm1, Cm2, Cr0, Cr2 = vp["Cm1"], vp["Cm2"], vp["Cr0"], vp["Cr2"]

    s, n, alpha, v, D, delta = x[0], x[1], x[2], x[3], x[4], x[5]
    derD, derDelta = u[0], u[1]

    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * ca.tanh(5 * v)
    sdota = (v * ca.cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)
    return ca.vertcat(
        sdota,
        v * ca.sin(alpha + C1 * delta),
        v * C2 * delta - kapparef_s(s) * sdota,
        Fxd / m * ca.cos(C1 * delta),
        derD,
        derDelta,
    )


def define_a_long_a_lat_exprs(
    x: ca.SX,
    vp: dict[str, float] | None = None,
) -> tuple[ca.SX, ca.SX]:
    """Symbolic longitudinal and lateral accelerations used in the soft path constraints."""
    vp = VEHICLE_PARAMS_DEFAULT if vp is None else vp
    m, C1, C2 = vp["m"], vp["C1"], vp["C2"]
    Cm1, Cm2, Cr0, Cr2 = vp["Cm1"], vp["Cm2"], vp["Cr0"], vp["Cr2"]

    v, D, delta = x[3], x[4], x[5]
    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * ca.tanh(5 * v)
    a_long = Fxd / m
    a_lat = C2 * v * v * delta + Fxd * ca.sin(C1 * delta) / m
    return a_long, a_lat


def f_explicit_numpy_factory(
    track_file: Path = DEFAULT_TRACK_FILE,
    vp: dict[str, float] | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a numpy-callable ``f(x, u) -> dx``.

    Evaluates the same continuous dynamics used inside the OCP.
    """
    x_sym = ca.SX.sym("x", 6)
    u_sym = ca.SX.sym("u", 2)
    kapparef_s, _ = build_curvature_spline(track_file)
    f_expr = define_f_expl_expr(x_sym, u_sym, kapparef_s, vp)
    f = ca.Function("f", [x_sym, u_sym], [f_expr])

    def _f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.asarray(f(x, u)).flatten()

    return _f


def frenet_to_cartesian(
    s: np.ndarray,
    n: np.ndarray,
    sref: np.ndarray,
    xref: np.ndarray,
    yref: np.ndarray,
    psiref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Frenet ``(s, n)`` to Cartesian ``(x, y, psi_track)``.

    Applies linear interpolation between the two nearest reference samples.
    Vectorised over scalar / array inputs.
    Mirrors ``time2spatial.transformProj2Orig`` (linear interpolation along the track polyline).
    """
    tracklength = sref[-1]
    s_mod = np.atleast_1d(s) % tracklength
    n_arr = np.atleast_1d(n)

    idx = np.searchsorted(sref, s_mod, side="right") - 1
    idx = np.clip(idx, 0, sref.size - 2)
    idx_next = idx + 1

    seg_len = sref[idx_next] - sref[idx]
    seg_len = np.where(seg_len > 1e-12, seg_len, 1e-12)
    t = (s_mod - sref[idx]) / seg_len

    x0 = (1 - t) * xref[idx] + t * xref[idx_next]
    y0 = (1 - t) * yref[idx] + t * yref[idx_next]
    psi0 = (1 - t) * psiref[idx] + t * psiref[idx_next]

    x = x0 - n_arr * np.sin(psi0)
    y = y0 + n_arr * np.cos(psi0)
    return x, y, psi0
