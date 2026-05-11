"""Model and reference-path utilities for the MPCC race-car planner.

Implements the Model Predictive Contouring Control formulation described in
``mpcc/main.tex`` and adapted to the 2D bicycle in ``RaceCarEnv``.

The MPCC formulation augments the physical state with a virtual progress state
``theta`` and a virtual control ``v_theta`` (its rate). The cost rewards progress
along a reference path ``p^d(theta) = (x^d(theta), y^d(theta))`` while penalising
position error split into contouring (perpendicular to the path) and lag (along the
path) components. The reference path is parametrised by arc length and built from
the same track file as the existing Frenet planner.

Two frame variants are exposed and share this module:

- ``cartesian``: physical state ``[x, y, psi, v, D, delta]`` and reference-path
  splines ``x^d, y^d, psi^d``. Tangent is ``t(theta) = [cos psi^d, sin psi^d]``.
  This is the paper-style formulation; contouring/lag errors are computed via
  projection onto the path tangent.
- ``frenet``: physical state ``[s, n, alpha, v, D, delta]`` reused from
  ``bicycle_model.py``. Contouring error reduces to ``n`` and lag error to
  ``s - theta``. Useful as an ablation that isolates the role of the virtual
  progress / progress-reward terms without changing the frame.

Both variants share the augmented control ``[derD, derDelta, dv_theta]`` and the
augmented states ``theta`` and ``v_theta``, giving ``nx = 8`` and ``nu = 3``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import casadi as ca
import numpy as np

from leap_c.examples.race_car.bicycle_model import (
    DEFAULT_TRACK_FILE,
    VEHICLE_PARAMS_DEFAULT,
    define_a_long_a_lat_exprs,
    define_f_expl_expr,
    frenet_to_cartesian,
    get_track,
)

__all__ = [
    "DV_THETA_MAX_DEFAULT",
    "DV_THETA_MIN_DEFAULT",
    "MpccFrame",
    "MpccTrackGeometry",
    "NU_MPCC",
    "NX_MPCC",
    "V_THETA_MAX_DEFAULT",
    "build_mpcc_path_splines",
    "define_a_long_a_lat_mpcc",
    "define_contour_lag_exprs",
    "define_f_expl_mpcc_cartesian",
    "define_f_expl_mpcc_frenet",
    "f_explicit_numpy_factory_mpcc",
    "frenet_obs_to_mpcc_state",
    "x0_default",
]

MpccFrame = Literal["cartesian", "frenet"]
"""Reference frame for the MPCC formulation. ``cartesian`` follows mpcc/main.tex;
``frenet`` is the lower-friction ablation that keeps the existing Frenet dynamics."""

# Augmented dimensions; shared by both frames.
NX_MPCC = 8
NU_MPCC = 3

# Default bounds on the virtual progress states / controls. v_theta is the path-speed
# of the virtual progress variable (m/s along arc length); dv_theta is its rate.
V_THETA_MAX_DEFAULT = 5.0
DV_THETA_MIN_DEFAULT = -4.0
DV_THETA_MAX_DEFAULT = 4.0


@dataclass(frozen=True)
class MpccTrackGeometry:
    """Reference-path splines and raw samples used by the MPCC OCP.

    The four interpolants are CasADi ``bspline``s in arc length ``s`` (= ``theta``
    in the MPCC notation). They are extended ~80 m backwards and one full lap
    forwards in arc length so the splines remain well-defined for ``theta``
    values the OCP may probe outside ``[0, pathlength]`` (negative at start,
    beyond ``pathlength`` near lap completion).

    The raw track samples (``s_samples`` etc.) are kept alongside so the
    Frenet-to-Cartesian observation conversion does not have to re-load the
    track file on every solve.

    The Cartesian splines ``xref``, ``yref``, ``psiref`` are only consulted by
    the Cartesian frame variant; the Frenet variant uses only ``kapparef``.

    Attributes:
        kapparef: Track curvature ``kappa(s)``.
        xref: Centerline Cartesian ``x(s)``.
        yref: Centerline Cartesian ``y(s)``.
        psiref: Centerline heading ``psi(s)`` (unwrapped).
        pathlength: Length of one lap, ``float(track_s[-1])``.
        s_samples, x_samples, y_samples, psi_samples: Raw track samples used by
            ``frenet_to_cartesian`` (linear interpolation along the polyline).
    """

    kapparef: ca.Function
    xref: ca.Function
    yref: ca.Function
    psiref: ca.Function
    pathlength: float
    s_samples: np.ndarray
    x_samples: np.ndarray
    y_samples: np.ndarray
    psi_samples: np.ndarray


def build_mpcc_path_splines(track_file: Path = DEFAULT_TRACK_FILE) -> MpccTrackGeometry:
    """Build the four reference-path splines used by the MPCC OCP.

    Replicates the wrap-around trick from ``bicycle_model.build_curvature_spline``:
    appends one full lap forward and ~80 m backwards in arc length so the
    interpolants are well-defined when the MPC horizon probes ``theta`` values
    just before the start line or past the finish line.

    The heading column is ``np.unwrap``-ed before extension; a track that closes
    on itself accumulates a net rotation of (approximately) ``+/-2*pi`` per lap,
    and the forward/backward extensions add/subtract that net rotation so the
    extended ``psi(s)`` remains continuous.
    """
    s0, xref, yref, psiref, kapparef = get_track(track_file)
    psiref = np.unwrap(psiref)

    length = len(s0)
    pathlength = float(s0[-1])
    dpsi_lap = float(psiref[-1] - psiref[0])

    s_ext = np.append(s0, s0[length - 1] + s0[1:length])
    x_ext = np.append(xref, xref[1:length])
    y_ext = np.append(yref, yref[1:length])
    psi_ext = np.append(psiref, psiref[1:length] + dpsi_lap)
    kappa_ext = np.append(kapparef, kapparef[1:length])

    pre_lo = length - 81
    pre_hi = length - 2
    s_full = np.append([-s_ext[length - 2] + s_ext[pre_lo:pre_hi]], s_ext)
    x_full = np.append(xref[length - 80 : length - 1], x_ext)
    y_full = np.append(yref[length - 80 : length - 1], y_ext)
    psi_full = np.append(psiref[length - 80 : length - 1] - dpsi_lap, psi_ext)
    kappa_full = np.append(kapparef[length - 80 : length - 1], kappa_ext)

    return MpccTrackGeometry(
        kapparef=ca.interpolant("kapparef_s_mpcc", "bspline", [s_full], kappa_full),
        xref=ca.interpolant("xref_s_mpcc", "bspline", [s_full], x_full),
        yref=ca.interpolant("yref_s_mpcc", "bspline", [s_full], y_full),
        psiref=ca.interpolant("psiref_s_mpcc", "bspline", [s_full], psi_full),
        pathlength=pathlength,
        s_samples=s0,
        x_samples=xref,
        y_samples=yref,
        psi_samples=psiref,
    )


def _Fxd_expr(v: ca.SX, D: ca.SX, vp: dict[str, float]) -> ca.SX:
    """Longitudinal drive-train force expression, identical to ``bicycle_model``."""
    Cm1, Cm2, Cr0, Cr2 = vp["Cm1"], vp["Cm2"], vp["Cr0"], vp["Cr2"]
    return (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * ca.tanh(5 * v)


def define_f_expl_mpcc_cartesian(
    x_aug: ca.SX,
    u_aug: ca.SX,
    vp: dict[str, float] | None = None,
) -> ca.SX:
    """Continuous augmented Cartesian dynamics for MPCC.

    State ``x_aug = [x, y, psi, v, D, delta, theta, v_theta]``.
    Control ``u_aug = [derD, derDelta, dv_theta]``.

    The bicycle equations match ``define_f_expl_expr`` in the Frenet frame but
    expressed directly in Cartesian: ``xdot = v cos(psi + C1 delta)``,
    ``ydot = v sin(psi + C1 delta)``, ``psidot = v C2 delta``,
    ``vdot = Fxd/m cos(C1 delta)``. The augmented states evolve as
    ``thetadot = v_theta`` and ``v_theta_dot = dv_theta``.
    """
    vp = VEHICLE_PARAMS_DEFAULT if vp is None else vp
    m, C1, C2 = vp["m"], vp["C1"], vp["C2"]

    psi = x_aug[2]
    v = x_aug[3]
    D = x_aug[4]
    delta = x_aug[5]
    v_theta = x_aug[7]

    derD = u_aug[0]
    derDelta = u_aug[1]
    dv_theta = u_aug[2]

    Fxd = _Fxd_expr(v, D, vp)

    return ca.vertcat(
        v * ca.cos(psi + C1 * delta),
        v * ca.sin(psi + C1 * delta),
        v * C2 * delta,
        Fxd / m * ca.cos(C1 * delta),
        derD,
        derDelta,
        v_theta,
        dv_theta,
    )


def define_f_expl_mpcc_frenet(
    x_aug: ca.SX,
    u_aug: ca.SX,
    kapparef_s: ca.Function,
    vp: dict[str, float] | None = None,
) -> ca.SX:
    """Continuous augmented Frenet dynamics for MPCC.

    State ``x_aug = [s, n, alpha, v, D, delta, theta, v_theta]``.
    Control ``u_aug = [derD, derDelta, dv_theta]``.

    Reuses ``define_f_expl_expr`` for the first six rows (the existing Frenet
    bicycle dynamics) and appends ``thetadot = v_theta`` and
    ``v_theta_dot = dv_theta``.
    """
    f_physical = define_f_expl_expr(x_aug[0:6], u_aug[0:2], kapparef_s, vp)
    v_theta = x_aug[7]
    dv_theta = u_aug[2]
    return ca.vertcat(f_physical, v_theta, dv_theta)


def define_contour_lag_exprs(
    x_aug: ca.SX,
    frame: MpccFrame,
    track_geom: MpccTrackGeometry,
) -> tuple[ca.SX, ca.SX]:
    """Symbolic contouring (perpendicular) and lag (along-path) errors.

    Cartesian (paper-style):
        ``e = p - [x^d(theta), y^d(theta)]``,
        ``t(theta) = [cos psi^d(theta), sin psi^d(theta)]``,
        ``n(theta) = [-sin psi^d(theta), cos psi^d(theta)]``,
        ``e_l = t^T e``, ``e_c = n^T e``    (both signed scalars).

    Frenet (ablation):
        Path is the centerline itself, so the contouring error is exactly the
        Frenet lateral offset ``n`` and the lag error is ``s - theta``. No path
        splines are needed; ``track_geom`` is accepted for signature symmetry.

    Returns ``(e_c, e_l)`` as 1x1 CasADi scalars.
    """
    theta = x_aug[6]
    if frame == "cartesian":
        x_pos = x_aug[0]
        y_pos = x_aug[1]
        x_d = track_geom.xref(theta)
        y_d = track_geom.yref(theta)
        psi_d = track_geom.psiref(theta)
        ex = x_pos - x_d
        ey = y_pos - y_d
        cos_p = ca.cos(psi_d)
        sin_p = ca.sin(psi_d)
        e_l = cos_p * ex + sin_p * ey
        e_c = -sin_p * ex + cos_p * ey
        return e_c, e_l
    if frame == "frenet":
        s = x_aug[0]
        n = x_aug[1]
        return n, s - theta
    raise ValueError(f"Unknown frame {frame!r}; expected 'cartesian' or 'frenet'.")


def define_a_long_a_lat_mpcc(
    x_aug: ca.SX,
    frame: MpccFrame,
    vp: dict[str, float] | None = None,
) -> tuple[ca.SX, ca.SX]:
    """Longitudinal / lateral accelerations for the path-constraint vector.

    Both frames use the same bicycle parameters and depend on ``(v, D, delta)``,
    which sit at the same indices ``(3, 4, 5)`` in the augmented state, so the
    function delegates to ``define_a_long_a_lat_exprs`` regardless of frame.
    """
    _ = frame  # both frames share v/D/delta layout
    return define_a_long_a_lat_exprs(x_aug[0:6], vp)


def frenet_obs_to_mpcc_state(
    obs_frenet: np.ndarray,
    frame: MpccFrame,
    track_geom: MpccTrackGeometry,
    vp: dict[str, float] | None = None,
) -> np.ndarray:
    """Map env (Frenet) observations to the MPCC augmented initial state.

    Args:
        obs_frenet: Shape ``(B, 6)`` or ``(6,)`` Frenet observation
            ``[s, n, alpha, v, D, delta]``.
        frame: Target frame.
        track_geom: Reference-path splines (used for the Cartesian conversion).
        vp: Vehicle parameters (defaults to the bicycle model defaults). Only
            ``C1`` is consulted, to seed ``v_theta_0`` along the path tangent.

    Returns:
        ``(B, 8)`` augmented state ``[*x_phys, theta, v_theta]``. ``theta_0`` is
        set to the observed arc length ``s`` (Cartesian and Frenet alike);
        ``v_theta_0`` is initialised to ``v cos(alpha + C1 delta)`` which is the
        on-centerline expression for ``ds/dt`` and gives a good warm-start for the
        virtual progress velocity.
    """
    vp = VEHICLE_PARAMS_DEFAULT if vp is None else vp
    C1 = float(vp["C1"])

    arr = np.atleast_2d(np.asarray(obs_frenet, dtype=np.float64))
    s = arr[:, 0]
    n = arr[:, 1]
    alpha = arr[:, 2]
    v = arr[:, 3]
    D = arr[:, 4]
    delta = arr[:, 5]

    theta_0 = s.copy()
    v_theta_0 = v * np.cos(alpha + C1 * delta)

    if frame == "frenet":
        x_phys = arr[:, :6]
    elif frame == "cartesian":
        x_cart, y_cart, psi_track = frenet_to_cartesian(
            s,
            n,
            track_geom.s_samples,
            track_geom.x_samples,
            track_geom.y_samples,
            track_geom.psi_samples,
        )
        psi = np.asarray(psi_track) + alpha
        x_phys = np.stack(
            [
                np.asarray(x_cart).ravel(),
                np.asarray(y_cart).ravel(),
                np.asarray(psi).ravel(),
                v,
                D,
                delta,
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown frame {frame!r}; expected 'cartesian' or 'frenet'.")

    return np.concatenate([x_phys, theta_0[:, None], v_theta_0[:, None]], axis=1)


def x0_default(frame: MpccFrame, track_geom: MpccTrackGeometry) -> np.ndarray:
    """Smoke-test initial augmented state corresponding to ``s=-2``, all else zero.

    Used by ``export_mpcc_ocp`` to set ``ocp.constraints.x0``; the planner overwrites
    this from the live observation at every solve.
    """
    obs0 = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return frenet_obs_to_mpcc_state(obs0, frame, track_geom).ravel()


def f_explicit_numpy_factory_mpcc(
    frame: MpccFrame,
    track_geom: MpccTrackGeometry,
    vp: dict[str, float] | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """NumPy-callable evaluator of the MPCC augmented dynamics (for tests / debug)."""
    x_sym = ca.SX.sym("x_aug", NX_MPCC)
    u_sym = ca.SX.sym("u_aug", NU_MPCC)
    if frame == "cartesian":
        expr = define_f_expl_mpcc_cartesian(x_sym, u_sym, vp)
    elif frame == "frenet":
        expr = define_f_expl_mpcc_frenet(x_sym, u_sym, track_geom.kapparef, vp)
    else:
        raise ValueError(f"Unknown frame {frame!r}.")
    fcn = ca.Function("f_mpcc", [x_sym, u_sym], [expr])

    def _f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.asarray(fcn(x, u)).flatten()

    return _f
