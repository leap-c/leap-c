import types
from typing import Literal

import gymnasium as gym
import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp

from leap_c.examples.race_car.acados_ocp import RaceCarAcadosCostType
from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

from leap_c.examples.race_car.track import Track

RaceAcadosParamInterface = Literal["global", "stagewise"]
RaceAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS"]


def create_race_params(
    param_interface: RaceAcadosParamInterface,
    N_horizon: int = 50,
) -> list[AcadosParameter]:

    q_diag_sqrt = np.sqrt(np.array([1e-1, 1e-8, 1e-8, 1e-8, 1e-3, 5e-3]))
    r_diag_sqrt = np.sqrt(np.array([1e-3, 5e-3]))
    qe_diag_sqrt = np.sqrt(np.array([5e0, 1e1, 1e-8, 1e-8, 5e-3, 2e-3]))

    """Returns a list of parameters used in racecar."""
    return [
        # Dynamics parameters
        AcadosParameter("m", default=np.array([0.043])),    # mass [kg]
        AcadosParameter("C1", default=np.array([0.5])),     # front tire parameter
        AcadosParameter("C2", default=np.array([15.5])),    # rear tire parameter
        AcadosParameter("Cm1", default=np.array([0.28])),   # motor parameter
        AcadosParameter("Cm2", default=np.array([0.05])),   # motor parameter
        AcadosParameter("Cr0", default=np.array([0.011])),  # rolling resistance
        AcadosParameter("Cr2", default=np.array([0.006])),  # air resistance
        # Cost matrix factorization parameters
        AcadosParameter(
            "q_diag_sqrt",
            default=q_diag_sqrt,
            interface="learnable",
            space=gym.spaces.Box(
                low=qe_diag_sqrt * 0.01,
                high=qe_diag_sqrt * 100.0,
                dtype=np.float32,
            ),
        ),  # cost on state residuals
        AcadosParameter(
            "r_diag_sqrt",
            default=r_diag_sqrt,
        ),
        AcadosParameter(
            "qe_diag_sqrt",
            default=qe_diag_sqrt,
            interface="learnable",
            space=gym.spaces.Box(
                low=qe_diag_sqrt * 0.01,
                high=qe_diag_sqrt * 100.0,
                dtype=np.float32,
            ),
        ),
        AcadosParameter(
            "thetaref",
            default=np.array([0.0]),
            interface="non-learnable",
            vary_stages=list(range(N_horizon + 1)),
        )
    ]

def define_f_expl_expr(
    model: AcadosModel, param_manager: AcadosParameterManager, track: Track
) -> ca.SX:
    m = param_manager.get("m")
    C1, C2 = param_manager.get("C1"), param_manager.get("C2")
    Cm1, Cm2 = param_manager.get("Cm1"), param_manager.get("Cm2")
    Cr0, Cr2 = param_manager.get("Cr0"), param_manager.get("Cr2")

    kapparef = ca.interpolant("kapparef_s", "linear", [track.thetaref], track.kapparef)
    L = float(track.thetaref[-1])

    theta = model.x[0]
    ec = model.x[1]
    epsi = model.x[2]
    v = model.x[3]
    delta = model.x[4]
    D = model.x[5]
    x = ca.vertcat(theta, ec, epsi, v, delta, D)

    ddelta = model.u[0]
    dD = model.u[1]
    u = ca.vertcat(ddelta, dD)

    s_wrapped = theta - L * ca.floor(theta / L)
    kappa = kapparef(s_wrapped)

    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * ca.tanh(5 * v)
    beta = C1 * delta
    thetadot = v * ca.cos(epsi + beta) / (1 - kappa * ec + 1e-3)

    f_expl = ca.vertcat(
        thetadot,
        v * ca.sin(epsi + beta),
        v * C2 * delta - kappa * thetadot,
        Fxd / m * ca.cos(C1 * delta),
        ddelta,
        dD
    )

    constraint = types.SimpleNamespace()
    constraint.n_min = -0.12  # width of the track [m]
    constraint.n_max = 0.12  # width of the track [m]

    # state bounds
    constraint.throttle_min = -1.0
    constraint.throttle_max = 1.0

    constraint.delta_min = -0.40  # minimum steering angle [rad]
    constraint.delta_max = 0.40  # maximum steering angle [rad]

    # input bounds
    constraint.ddelta_min = -2.0  # minimum change rate of stering angle [rad/s]
    constraint.ddelta_max = 2.0  # maximum change rate of steering angle [rad/s]
    constraint.dthrottle_min = -10  # -10.0  # minimum throttle change rate
    constraint.dthrottle_max = 10  # 10.0  # maximum throttle change rate

    # constraint on forces
    a_lat = C2 * v * v * delta + Fxd * ca.sin(C1 * delta) / m
    a_long = Fxd / m

    # Model bounds
    # nonlinear constraint
    constraint.alat_min = -4  # maximum lateral force [m/s^2]
    constraint.alat_max = 4  # maximum lateral force [m/s^1]

    constraint.along_min = -4  # maximum longitudinal force [m/s^2]
    constraint.along_max = 4  # maximum longitudinal force [m/s^2]

    constraint.alat = ca.Function("a_lat", [x, u], [a_lat])
    constraint.expr = ca.vertcat(a_long, a_lat, ec, delta, D)
    return f_expl, constraint


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    cost_type: RaceCarAcadosCostType = "NONLINEAR_LS",
    name: str = "race_car",
    track_file: str = "LMS_Track.txt",
    N_horizon: int = 50,
    T_horizon: float = 1.0,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon
    ######## Model ########

    ocp.model.name = name

    ocp.dims.nx = 6
    ocp.dims.nu = 2

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)  # [theta, ec, epsi, v, delta, D]
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)  # [ddelta, dD]

    p = ca.vertcat(
        param_manager.non_learnable_parameters.cat,
        param_manager.learnable_parameters.cat,
    )

    track = Track(track_file)
    f_expl, constraint = define_f_expl_expr(ocp.model, param_manager, track)
    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=f_expl,
        x=ocp.model.x,
        u=ocp.model.u,
        p=p,
        dt=dt,
    )

    ######## Cost ########
    thetaref = param_manager.get("thetaref")
    yref = ca.vertcat(thetaref, 0, 0, 0, 0, 0, 0, 0)
    yref_e = ca.vertcat(thetaref, 0, 0, 0, 0, 0)
    y = ca.vertcat(ocp.model.x, ocp.model.u)
    y_e = ocp.model.x

    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    qe_diag_sqrt = param_manager.get("qe_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    We_sqrt = ca.diag(qe_diag_sqrt)

    # W, We = W_sqrt, We_sqrt
    W = W_sqrt @ W_sqrt.T
    We = We_sqrt  @ We_sqrt.T

    # Scale cost matrices
    unscale = N_horizon / T_horizon
    W = unscale * W
    We = We / unscale

    if cost_type == "EXTERNAL":
        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = 0.5 * (yref - y).T @ W @ (yref - y)

        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = 0.5 * (yref_e - y_e).T @ We @ (yref_e - y_e)
    elif cost_type == "NONLINEAR_LS":
        pass
    else:
        raise ValueError(
            f"Cost type {cost_type} not supported. Use 'EXTERNAL' or 'NONLINEAR_LS'."
        )

    ######## Constraints ########
    ocp.model.con_h_expr = constraint.expr

    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5])
    ocp.constraints.x0 = np.array([-2, 0, 0, 0, 0, 0])  # Match env init: v=0.5 m/s

    ocp.constraints.lbu = np.array([constraint.ddelta_min, constraint.dthrottle_min])
    ocp.constraints.ubu = np.array([constraint.ddelta_max, constraint.dthrottle_max])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([constraint.n_min])
    ocp.constraints.ubx = np.array([constraint.n_max])
    ocp.constraints.idxbx = np.array([1])


    ocp.constraints.lbx_e = np.array([constraint.n_min])
    ocp.constraints.ubx_e = np.array([constraint.n_max])
    ocp.constraints.idxbx_e = np.array([1])

    ocp.constraints.lh = np.array(
        [
            constraint.along_min,
            constraint.alat_min,
            constraint.n_min,
            constraint.delta_min,
            constraint.throttle_min,
        ]
    )
    ocp.constraints.uh = np.array(
        [
            constraint.along_max,
            constraint.alat_max,
            constraint.n_max,
            constraint.delta_max,
            constraint.throttle_max,
        ]
    )

    nsh = 2
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array([0, 2])

    ns = 2
    ocp.cost.zl = 100 * np.ones((ns,))
    ocp.cost.Zl = 0 * np.ones((ns,))
    ocp.cost.zu = 100 * np.ones((ns,))
    ocp.cost.Zu = 0 * np.ones((ns,))

    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.hessian_approx = "EXACT"

    return ocp