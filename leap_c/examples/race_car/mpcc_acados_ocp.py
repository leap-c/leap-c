"""Parametric MPCC OCP for the race-car example.

Implements the Model Predictive Contouring Control formulation described in
``mpcc/main.tex`` and adapted to the 2D bicycle in ``RaceCarEnv``. Mirrors the
structure of ``acados_ocp.py`` (param-manager based, ``EXTERNAL`` cost, RK4
discrete integrator, SQP solver) so the two planners can be compared head-to-head.

Problem statement
-----------------

Decision variables (both frames, ``nx = 8`` and ``nu = 3``):
    Cartesian: ``x = [x, y, psi, v, D, delta, theta, v_theta]``
    Frenet:    ``x = [s, n, alpha, v, D, delta, theta, v_theta]``
    Control:   ``u = [derD, derDelta, dv_theta]``

Dynamics:
    Cartesian: ``define_f_expl_mpcc_cartesian`` (Cartesian bicycle + (theta, v_theta)).
    Frenet:    ``define_f_expl_mpcc_frenet`` (Frenet bicycle + (theta, v_theta)).
    Integrated via ``integrate_erk4`` (3 substeps) to a ``disc_dyn_expr``.

Cost (``NONLINEAR_LS`` with Gauss-Newton Hessian):
    Residual vector
        ``y = [e_c, e_l, derD, derDelta, dv_theta, v_theta - v_target]``
    with ``v_target = V_TARGET_RATIO * v_theta_max`` and weight chosen so the
    residual ``v_theta - v_target`` expands to ``-mu * v_theta + (small)``
    plus a small quadratic curvature on the ``v_theta`` block that pulls
    ``v_theta`` toward its upper bound from stage 0 (without this curvature
    the optimum is indifferent to *where* in the horizon ``v_theta`` grows
    and the closed loop never escapes the cold-start equilibrium).

    An EXTERNAL cost with a literal linear ``-mu * v_theta`` term has no
    Hessian curvature on ``v_theta`` and the SQP step degenerates to MINSTEP
    on cold start. Encoding progress as a residual makes the Gauss-Newton
    Hessian ``J^T J`` positive-semidefinite by construction.

    ``e_c`` and ``e_l`` are the contouring and lag errors against the reference
    path; in Cartesian via tangent projection, in Frenet via the trivial
    identities ``e_c = n``, ``e_l = s - theta``. All cost weights are learnable
    parameters (passed in as their square roots so the realised weights are
    non-negative even under SAC sampling).

Constraints:
    Hard input bounds (``idxbu = [0, 1, 2]``):
        derD     in [-10, 10] 1/s
        derDelta in [-2,  2]  rad/s
        dv_theta in [dv_theta_min, dv_theta_max]
    Hard state box on v_theta (``idxbx = [7]``):
        v_theta in [0, v_theta_max]   (non-reversal)
    Nonlinear path constraint ``h(x, u) = [a_long, a_lat, e_c (or n), D, delta]``:
        same bounds as the existing race_car OCP; all 5 components soft, with
        ``zl = zu = 100`` and ``Zl = Zu = 1`` matching the existing slack penalties.
"""

from pathlib import Path
from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.examples.race_car.bicycle_model import (
    ALAT_MAX_DEFAULT,
    ALONG_MAX_DEFAULT,
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
)
from leap_c.examples.race_car.mpcc_model import (
    DV_THETA_MAX_DEFAULT,
    DV_THETA_MIN_DEFAULT,
    NU_MPCC,
    NX_MPCC,
    V_THETA_MAX_DEFAULT,
    MpccFrame,
    build_mpcc_path_splines,
    define_a_long_a_lat_mpcc,
    define_contour_lag_exprs,
    define_f_expl_mpcc_cartesian,
    define_f_expl_mpcc_frenet,
    x0_default,
)
from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

MpccParamInterface = Literal["global", "stagewise"]
"""``global``: one set of cost weights shared across all stages.
``stagewise``: one set of cost weights per stage."""


# Default cost-weight square-roots and progress-reward magnitudes.
#
# Square-root parametrisation matches the existing race_car ``q_diag_sqrt``
# convention and keeps the realised weight ``= sqrt^2 >= 0`` even if the
# learning loop drives a sample slightly negative.
#
# Cost-balance rationale: the progress reward ``-mu * v_theta`` must be strong
# enough to overcome the cold-start equilibrium (where ``v = v_theta = 0`` is a
# zero-cost local minimum if the dynamics linearisation gives near-zero
# costates on the physical-velocity block). Empirically ``mu = 50`` is the
# smallest value that lifts the car off the start line with this set of input
# regularisers; below that the optimiser plans "delay throttle then act" and
# the closed loop never escapes stage 0. The contouring weight is then chosen
# so that ``q_c * e_c_max^2`` (with e_c_max = 0.12 m) is comparable to the
# horizon-integrated progress reward, keeping the car on the track at the
# faster speeds the larger ``mu`` enables.
_Q_C_SQRT_DEFAULT = float(np.sqrt(10000.0))
_Q_L_SQRT_DEFAULT = float(np.sqrt(1000.0))
_MU_DEFAULT = 100.0
_R_DIAG_SQRT_DEFAULT = np.sqrt(np.array([1.0e-3, 5.0e-3, 1.0e-1]))  # derD, derDelta, dv_theta
_Q_E_C_SQRT_DEFAULT = float(np.sqrt(20000.0))
_Q_E_L_SQRT_DEFAULT = float(np.sqrt(2000.0))
_MU_E_DEFAULT = 100.0

# Offset used by the LS-residual encoding of the linear progress reward. The
# residual ``v_theta - v_target`` weighted by ``q_v = mu / v_target`` expands to
# a linear term ``-mu * v_theta`` plus a quadratic damping
# ``(mu / (2 v_target)) * v_theta^2``.
#
# v_target = 2 * v_theta_max gives a useful curvature: at v_theta = v_theta_max
# the residual gradient is ``q_v * (v_theta_max - v_target) = -mu / 2``, i.e.
# half the cold-start pull, so the Gauss-Newton step naturally drives v_theta
# toward its upper bound rather than indifferently along the linear part. With
# v_theta_max = 5 the damping at v_theta = 5 is 0.625 per stage vs ~5 from the
# progress reward, so the linear interpretation still dominates.
V_TARGET_RATIO = 2.0


def create_mpcc_params(
    param_interface: MpccParamInterface,
    N_horizon: int = 50,
) -> list[AcadosParameter]:
    """Build the learnable parameter set of the MPCC OCP.

    All cost weights are learnable, with optional per-stage variation when
    ``param_interface == "stagewise"`` (mirrors ``create_race_car_params``).

    Spaces are conservative +/- 10% boxes around the defaults; SAC trainers can
    pick samples from the space without driving the OCP into pathological
    parametrisations.
    """
    end_stages = list(range(N_horizon + 1)) if param_interface == "stagewise" else []

    def _scalar_param(name: str, default_value: float) -> AcadosParameter:
        default = np.array([default_value], dtype=np.float64)
        low = default * 0.9
        high = default * 1.1
        # Allow zero / sign-flip neighbourhood when the default is small.
        if default_value == 0.0:
            low, high = -np.ones_like(default), np.ones_like(default)
        return AcadosParameter(
            name,
            default=default,
            space=gym.spaces.Box(low=low, high=high, dtype=np.float64),
            interface="learnable",
            end_stages=end_stages,
        )

    return [
        _scalar_param("q_c_sqrt", _Q_C_SQRT_DEFAULT),
        _scalar_param("q_l_sqrt", _Q_L_SQRT_DEFAULT),
        _scalar_param("mu", _MU_DEFAULT),
        AcadosParameter(
            "r_diag_sqrt",
            default=_R_DIAG_SQRT_DEFAULT,
            space=gym.spaces.Box(
                low=_R_DIAG_SQRT_DEFAULT * 0.9,
                high=_R_DIAG_SQRT_DEFAULT * 1.1,
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=end_stages,
        ),
        _scalar_param("q_e_c_sqrt", _Q_E_C_SQRT_DEFAULT),
        _scalar_param("q_e_l_sqrt", _Q_E_L_SQRT_DEFAULT),
        _scalar_param("mu_e", _MU_E_DEFAULT),
    ]


MpccNlpSolver = Literal["SQP", "SQP_RTI"]
"""``SQP``: full SQP per call (matches the existing race_car planner). ``SQP_RTI``:
one QP per call with warm-start (paper-style MPCC, much faster but needs a good
warm-start to follow the path - degrades on the very first solve). Default is
SQP for an apples-to-apples comparison with ``RaceCarPlanner`` and for cold-start
robustness; flip to ``SQP_RTI`` for paper-style real-time behaviour."""


def export_mpcc_ocp(
    param_manager: AcadosParameterManager,
    frame: MpccFrame = "cartesian",
    track_file: Path = DEFAULT_TRACK_FILE,
    name: str | None = None,
    N_horizon: int = 50,
    T_horizon: float = 1.0,
    vehicle_params: dict[str, float] | None = None,
    v_theta_max: float = V_THETA_MAX_DEFAULT,
    dv_theta_min: float = DV_THETA_MIN_DEFAULT,
    dv_theta_max: float = DV_THETA_MAX_DEFAULT,
    nlp_solver: MpccNlpSolver = "SQP_RTI",
) -> AcadosOcp:
    """Build the parametric MPCC ``AcadosOcp`` for the given frame."""
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon
    param_manager.assign_to_ocp(ocp)

    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon
    track_geom = build_mpcc_path_splines(track_file)

    # -------- Model --------
    ocp.model.name = name if name is not None else f"race_car_mpcc_{frame}"
    ocp.dims.nx = NX_MPCC
    ocp.dims.nu = NU_MPCC
    ocp.model.x = ca.SX.sym("x", NX_MPCC)
    ocp.model.u = ca.SX.sym("u", NU_MPCC)

    if frame == "cartesian":
        f_expl = define_f_expl_mpcc_cartesian(ocp.model.x, ocp.model.u, vehicle_params)
    elif frame == "frenet":
        f_expl = define_f_expl_mpcc_frenet(
            ocp.model.x, ocp.model.u, track_geom.kapparef, vehicle_params
        )
    else:
        raise ValueError(f"Unknown frame {frame!r}; expected 'cartesian' or 'frenet'.")

    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=f_expl,
        x=ocp.model.x,
        u=ocp.model.u,
        p=param_manager.p_full,
        dt=dt,
        num_substeps=3,
    )

    # -------- Cost (NONLINEAR_LS, Gauss-Newton) --------
    #
    # We encode the MPCC objective
    #   l_k(x, u) = q_c * e_c^2 + q_l * e_l^2 + 0.5 * u^T diag(r_diag) u - mu * v_theta
    # as the squared norm of a residual ``y`` with identity weight ``W = I``:
    #   * y[0] = sqrt(2 q_c) * e_c   (gives q_c e_c^2)
    #   * y[1] = sqrt(2 q_l) * e_l   (gives q_l e_l^2)
    #   * y[2:5] = sqrt(r_diag) * u  (gives 0.5 r_diag u^2)
    #   * y[5] = sqrt(q_v) * (v_theta - v_target)
    #         -> 0.5 q_v (v_theta - v_target)^2
    #         = (q_v / 2) v_theta^2 - q_v v_target * v_theta + const.
    # with q_v = mu / v_target so the linear term recovers the MPCC progress
    # reward ``-mu * v_theta``. The damping coefficient q_v / 2 also adds a small
    # quadratic pull toward v_target. Picking v_target = V_TARGET_RATIO *
    # v_theta_max gives a Hessian curvature on the v_theta block that drives
    # v_theta toward its upper bound right at stage 0 - without this curvature
    # the optimum is indifferent to *where* in the horizon v_theta grows, and
    # the closed loop never escapes the cold-start equilibrium.
    e_c, e_l = define_contour_lag_exprs(ocp.model.x, frame, track_geom)
    v_theta = ocp.model.x[7]
    v_target = V_TARGET_RATIO * v_theta_max

    q_c_sqrt = param_manager.get("q_c_sqrt")
    q_l_sqrt = param_manager.get("q_l_sqrt")
    mu = param_manager.get("mu")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    q_e_c_sqrt = param_manager.get("q_e_c_sqrt")
    q_e_l_sqrt = param_manager.get("q_e_l_sqrt")
    mu_e = param_manager.get("mu_e")

    sqrt2 = float(np.sqrt(2.0))
    q_v_sqrt = ca.sqrt(mu / v_target)
    q_v_e_sqrt = ca.sqrt(mu_e / v_target)

    y = ca.vertcat(
        sqrt2 * q_c_sqrt * e_c,
        sqrt2 * q_l_sqrt * e_l,
        r_diag_sqrt[0] * ocp.model.u[0],
        r_diag_sqrt[1] * ocp.model.u[1],
        r_diag_sqrt[2] * ocp.model.u[2],
        q_v_sqrt * (v_theta - v_target),
    )
    y_e = ca.vertcat(
        sqrt2 * q_e_c_sqrt * e_c,
        sqrt2 * q_e_l_sqrt * e_l,
        q_v_e_sqrt * (v_theta - v_target),
    )

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_e
    ocp.cost.yref = np.zeros(y.shape[0])
    ocp.cost.yref_e = np.zeros(y_e.shape[0])
    ocp.cost.W = np.eye(y.shape[0])
    ocp.cost.W_e = np.eye(y_e.shape[0])
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    # -------- Constraints --------
    ocp.constraints.x0 = x0_default(frame, track_geom)
    ocp.constraints.idxbx_0 = np.arange(NX_MPCC)

    # Hard input bounds on [derD, derDelta, dv_theta]
    ocp.constraints.lbu = np.array(
        [DTHROTTLE_MIN_DEFAULT, DDELTA_MIN_DEFAULT, dv_theta_min],
        dtype=np.float64,
    )
    ocp.constraints.ubu = np.array(
        [DTHROTTLE_MAX_DEFAULT, DDELTA_MAX_DEFAULT, dv_theta_max],
        dtype=np.float64,
    )
    ocp.constraints.idxbu = np.arange(NU_MPCC)

    # Hard state-box on the physical velocity v (idx 3; no reverse driving) and
    # on the virtual progress velocity v_theta (idx 7; non-reversal + speed
    # cap). The `v >= 0` bound is critical for the MPCC formulation - without
    # it the cost gradient at v = 0 is ambiguous on direction (the lag cost
    # `(s - theta)^2` grows symmetrically for forward and backward motion, the
    # progress reward `-mu * v_theta` is decoupled from the sign of v) and the
    # SQP can lock into a local minimum where the car reverses while v_theta
    # climbs. The bound matches the realistic top speed envelope from the
    # existing race_car closed-loop trace (max v ~ 1.5 m/s on the LMS track).
    ocp.constraints.lbx = np.array([0.0, 0.0], dtype=np.float64)
    ocp.constraints.ubx = np.array([v_theta_max, v_theta_max], dtype=np.float64)
    ocp.constraints.idxbx = np.array([3, 7])

    # Path constraint h(x, u) = [a_long, a_lat, e_c (or n), D, delta]
    a_long, a_lat = define_a_long_a_lat_mpcc(ocp.model.x, frame, vehicle_params)
    D = ocp.model.x[4]
    delta = ocp.model.x[5]
    ocp.model.con_h_expr = ca.vertcat(a_long, a_lat, e_c, D, delta)
    ocp.constraints.lh = np.array(
        [
            -ALONG_MAX_DEFAULT,
            -ALAT_MAX_DEFAULT,
            -N_MAX_DEFAULT,
            THROTTLE_MIN_DEFAULT,
            DELTA_MIN_DEFAULT,
        ]
    )
    ocp.constraints.uh = np.array(
        [
            ALONG_MAX_DEFAULT,
            ALAT_MAX_DEFAULT,
            N_MAX_DEFAULT,
            THROTTLE_MAX_DEFAULT,
            DELTA_MAX_DEFAULT,
        ]
    )
    nsh = 5
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.arange(nsh)
    ocp.cost.zl = 100.0 * np.ones(nsh)
    ocp.cost.zu = 100.0 * np.ones(nsh)
    ocp.cost.Zl = np.ones(nsh)
    ocp.cost.Zu = np.ones(nsh)

    # -------- Solver options --------
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = nlp_solver
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    return ocp


if __name__ == "__main__":
    for _frame in ("cartesian", "frenet"):
        N = 50
        pm = AcadosParameterManager(
            parameters=create_mpcc_params(param_interface="global", N_horizon=N),
            N_horizon=N,
        )
        ocp_built = export_mpcc_ocp(pm, frame=_frame, N_horizon=N, T_horizon=1.0)
        print(f"--- {_frame} ---")
        print("nx =", ocp_built.dims.nx, "  nu =", ocp_built.dims.nu)
        print("cost_type / cost_type_e:", ocp_built.cost.cost_type, "/", ocp_built.cost.cost_type_e)
        print("disc_dyn_expr shape:", ocp_built.model.disc_dyn_expr.shape)
        print("con_h_expr shape:", ocp_built.model.con_h_expr.shape)
