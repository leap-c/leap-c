"""Parametric OCP for the Frenet-frame race-car example.

Mirrors the cartpole pattern (cartpole/acados_ocp.py): a discrete RK4 integrator built via
``integrate_erk4``, parameter-dependent symbolic ``ocp.cost.W`` (Gauss-Newton via NONLINEAR_LS or
exact via EXTERNAL), and an ``AcadosParameterManager`` exposing learnable cost weights and a
non-learnable per-stage ``s_ref`` reference.

Problem statement
-----------------

Decision variables:
    x = [s, n, alpha, v, D, delta]    (nx = 6)
    u = [derD, derDelta]              (nu = 2)

Dynamics:
    Discrete RK4 integration over the continuous Frenet-frame bicycle model from
    ``bicycle_model.define_f_expl_expr``; the same continuous expression is used by the
    env for plant-model alignment.

Cost (NONLINEAR_LS form):
    l_k(x,u) = || W_sqrt @ (y - yref) ||^2       with y = [x, u]
    l_N(x)   = || W_e_sqrt @ (y_e - yref_e) ||^2 with y_e = x
    yref = [s_ref, 0, ..., 0], yref_e = [s_ref, 0, ..., 0]. W and W_e are diagonal; the
    diagonal square roots (``q_diag_sqrt``, ``r_diag_sqrt``, ``q_e_diag_sqrt``) are
    learnable parameters. Defaults match ``diag(Q), diag(R), diag(Qe)`` in the acados
    example. ``s_ref`` is a non-learnable per-stage arc-length reference, written by
    the planner at solve time.

Constraints:
    Hard box on u:
        derD     in [-10, 10] 1/s
        derDelta in [-2,   2] rad/s
    Nonlinear path h(x, u) = [a_long, a_lat, n, D, delta]:
        a_long   in [-4, 4] m/s^2      (soft, l1 slack penalty)
        a_lat    in [-4, 4] m/s^2      (hard)
        n        in [-0.12, 0.12] m    (soft, l1 slack penalty)
        D        in [-1, 1]            (hard)
        delta    in [-0.40, 0.40] rad  (hard)
    Initial state:
        x0 = [-2, 0, 0, 0, 0, 0]       (2 m before the start line, matches acados example)

Source for the OCP structure (cost weights, constraint set):
    external/acados/examples/acados_python/race_cars/acados_settings.py

References:
----------
- Reiter, R., Nurkanović, A., Frey, J., Diehl, M. (2023).
  "Frenet-Cartesian model representations for automotive obstacle avoidance
  within nonlinear MPC."
  European Journal of Control, Vol. 74, 100847.
  Preprint: https://arxiv.org/abs/2212.13115
  Published: https://www.sciencedirect.com/science/article/pii/S0947358023000766
"""

from pathlib import Path
from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

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
    build_curvature_spline,
    define_a_long_a_lat_exprs,
    define_f_expl_expr,
)
from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

RaceCarParamInterface = Literal["global", "stagewise"]
"""``global``: one set of cost weights shared across all stages.
``stagewise``: one set of cost weights per stage (per-stage learnable parameters)."""

RaceCarCostType = Literal["EXTERNAL", "NONLINEAR_LS"]
"""``NONLINEAR_LS`` uses a Gauss-Newton Hessian; ``EXTERNAL`` uses the exact Hessian."""


# Logical cost-weight values (Q, R, Qe) from the original acados example. Scaled by unscale = N/Tf
# at parameter-build time so the realized W = unscale * blkdiag(Q, R) and W_e = Qe / unscale
# match original example's `acados_settings_dev.py`
_Q_DIAG_DEFAULT = np.array([1e-1, 1e-8, 1e-8, 1e-8, 1e-3, 5e-3])
_R_DIAG_DEFAULT = np.array([1e-3, 5e-3])
_QE_DIAG_DEFAULT = np.array([5e0, 1e1, 1e-8, 1e-8, 5e-3, 2e-3])


def create_race_car_params(
    param_interface: RaceCarParamInterface,
    N_horizon: int = 50,
    T_horizon: float = 1.0,
) -> list[AcadosParameter]:
    """Parameters of the race-car OCP.

    The original acados example time-scales the cost matrices: ``W = (N/Tf) * blkdiag(Q, R)`` and
    ``W_e = Qe / (N/Tf)``. We bake that scaling into the parameter *defaults* so the realized
    ``W`` and ``W_e`` match upstream without a symbolic rescaling layer.
    """
    unscale = N_horizon / T_horizon
    q_sqrt_default = np.sqrt(_Q_DIAG_DEFAULT * unscale)
    r_sqrt_default = np.sqrt(_R_DIAG_DEFAULT * unscale)
    q_e_sqrt_default = np.sqrt(_QE_DIAG_DEFAULT / unscale)

    end_stages_q = list(range(N_horizon + 1)) if param_interface == "stagewise" else []
    end_stages_r = list(range(N_horizon + 1)) if param_interface == "stagewise" else []
    return [
        AcadosParameter(
            "q_diag_sqrt",
            default=q_sqrt_default,
            space=gym.spaces.Box(
                low=q_sqrt_default * 0.9,
                high=q_sqrt_default * 1.1,
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=end_stages_q,
        ),
        AcadosParameter(
            "r_diag_sqrt",
            default=r_sqrt_default,
            space=gym.spaces.Box(
                low=r_sqrt_default * 0.9,
                high=r_sqrt_default * 1.1,
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=end_stages_r,
        ),
        AcadosParameter(
            "q_e_diag_sqrt",
            default=q_e_sqrt_default,
            space=gym.spaces.Box(
                low=q_e_sqrt_default * 0.9,
                high=q_e_sqrt_default * 1.1,
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=[],
        ),
        AcadosParameter(
            "s_ref",
            default=np.array([0.0]),
            interface="non-learnable",
        ),
    ]


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    track_file: Path = DEFAULT_TRACK_FILE,
    cost_type: RaceCarCostType = "NONLINEAR_LS",
    name: str = "race_car",
    N_horizon: int = 50,
    T_horizon: float = 1.0,
    vehicle_params: dict[str, float] | None = None,
) -> AcadosOcp:
    """Build the parametric AcadosOcp for the Frenet-frame race-car problem."""
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    ######## Model ########
    ocp.model.name = name
    ocp.dims.nx = 6
    ocp.dims.nu = 2
    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    kapparef_s, _pathlength = build_curvature_spline(track_file)
    f_expl = define_f_expl_expr(ocp.model.x, ocp.model.u, kapparef_s, vehicle_params)

    p = ca.vertcat(
        param_manager.non_learnable_parameters.cat,
        param_manager.learnable_parameters.cat,
    )
    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=f_expl,
        x=ocp.model.x,
        u=ocp.model.u,
        p=p,
        dt=dt,
        num_substeps=3,
    )

    ######## Cost ########
    s_ref = param_manager.get("s_ref")
    yref = ca.vertcat(
        s_ref, ca.SX.zeros(7)
    )  # track s; (n, alpha, v, D, delta, derD, derDelta) -> 0
    yref_e = ca.vertcat(s_ref, ca.SX.zeros(5))
    y = ca.vertcat(ocp.model.x, ocp.model.u)
    y_e = ocp.model.x

    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    q_e_diag_sqrt = param_manager.get("q_e_diag_sqrt")
    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    W_e_sqrt = ca.diag(q_e_diag_sqrt)
    W = W_sqrt @ W_sqrt.T
    W_e = W_e_sqrt @ W_e_sqrt.T

    if cost_type == "NONLINEAR_LS":
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = W
        ocp.cost.yref = yref
        ocp.model.cost_y_expr = y
        ocp.cost.W_e = W_e
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = y_e
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    elif cost_type == "EXTERNAL":
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)
        ocp.model.cost_expr_ext_cost_e = 0.5 * (y_e - yref_e).T @ W_e @ (y_e - yref_e)
        ocp.solver_options.hessian_approx = "EXACT"
    else:
        raise ValueError(f"Unknown cost_type {cost_type!r}; use 'NONLINEAR_LS' or 'EXTERNAL'.")

    ######## Constraints ########
    ocp.constraints.x0 = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ocp.constraints.idxbx_0 = np.arange(6)

    # Hard control rate bounds.
    ocp.constraints.lbu = np.array([DTHROTTLE_MIN_DEFAULT, DDELTA_MIN_DEFAULT])
    ocp.constraints.ubu = np.array([DTHROTTLE_MAX_DEFAULT, DDELTA_MAX_DEFAULT])
    ocp.constraints.idxbu = np.array([0, 1])

    # Nonlinear path constraint h(x, u) = [a_long, a_lat, n, D, delta]
    a_long, a_lat = define_a_long_a_lat_exprs(ocp.model.x, vehicle_params)
    ocp.model.con_h_expr = ca.vertcat(
        a_long,
        a_lat,
        ocp.model.x[1],  # n
        ocp.model.x[4],  # D
        ocp.model.x[5],  # delta
    )
    ocp.constraints.lh = np.array(
        [
            -4.0,
            -4.0,
            -N_MAX_DEFAULT,
            THROTTLE_MIN_DEFAULT,
            DELTA_MIN_DEFAULT,
        ]
    )
    ocp.constraints.uh = np.array(
        [
            4.0,
            4.0,
            N_MAX_DEFAULT,
            THROTTLE_MAX_DEFAULT,
            DELTA_MAX_DEFAULT,
        ]
    )
    nsh = 5
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.arange(nsh)

    # Soft state-box on n (lateral offset): upstream has a loose bound |n| <= 12 that only
    # exists to give the slack variable a home (the tight bound |n| <= 0.12 is enforced via
    # the h-constraint above). Matches upstream `idxsbx=[0]`, `lbx=[-12], ubx=[12]`.
    ocp.constraints.idxbx = np.array([1])
    ocp.constraints.lbx = np.array([-12.0])
    ocp.constraints.ubx = np.array([12.0])
    nsbx = 1
    ocp.constraints.idxsbx = np.arange(nsbx)
    ocp.constraints.lsbx = np.zeros(nsbx)
    ocp.constraints.usbx = np.zeros(nsbx)

    ns = nsh + nsbx
    ocp.cost.zl = 100.0 * np.ones(ns)
    ocp.cost.zu = 100.0 * np.ones(ns)
    ocp.cost.Zl = np.ones(ns)
    ocp.cost.Zu = np.ones(ns)

    ######## Solver options ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1

    # ocp.solver_options.regularize_method = "GERSHGORIN_LEVENBERG_MARQUARDT"
    # ocp.solver_options.levenberg_marquardt = 1e-4

    return ocp


if __name__ == "__main__":
    # Smoke test: build the OCP and confirm symbolic structure.
    N = 50
    pm = AcadosParameterManager(
        parameters=create_race_car_params(param_interface="global", N_horizon=N),
        N_horizon=N,
    )
    ocp = export_parametric_ocp(pm, N_horizon=N, T_horizon=1.0)
    print("OCP build OK. nx=", ocp.dims.nx, "nu=", ocp.dims.nu)
    print("integrator_type:", ocp.solver_options.integrator_type)
    print("cost_type / cost_type_e:", ocp.cost.cost_type, "/", ocp.cost.cost_type_e)
    print("disc_dyn_expr shape:", ocp.model.disc_dyn_expr.shape)
    print("con_h_expr shape:", ocp.model.con_h_expr.shape)
    print("idxsh:", ocp.constraints.idxsh)
