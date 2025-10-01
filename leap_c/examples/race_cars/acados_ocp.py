import types
from typing import Literal

import scipy
import numpy as np
import casadi as ca
import gymnasium as gym
from acados_template import AcadosModel, AcadosOcp

from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

from leap_c.examples.race_cars.tracks.readDataFcn import getTrack


RaceCarAcadosParamInterface = Literal["global", "stagewise"]
RaceCarAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS", "LINEAR_LS"]


def create_race_car_params(
    param_interface: RaceCarAcadosParamInterface,
    N_horizon: int = 50,
    track_file: str = "LMS_Track.txt",
) -> list[AcadosParameter]:

    q_diag_sqrt = np.sqrt(np.array([1e-1, 1e-8, 1e-8, 1e-8, 1e-3, 5e-3]))
    r_diag_sqrt = np.sqrt(np.array([1e-3, 5e-3]))
    qe_diag_sqrt = np.sqrt(np.array([5e0, 1e1, 1e-8, 1e-8, 5e-3, 2e-3]))

    Sref, _, _, _, _ = getTrack(track_file)
    track_length = Sref[-1]

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
        AcadosParameter( #[s, n, alpha, v, D, delta]
            "q_diag_sqrt", default=q_diag_sqrt,
            interface="learnable",
            space=gym.spaces.Box(
                low=q_diag_sqrt * 0.01,
                high=q_diag_sqrt * 100.0,
                dtype=np.float32,
            ),
        ),  # cost on state residuals
        AcadosParameter(
            "r_diag_sqrt", default=r_diag_sqrt
        ),  # [derD, derDelta]
        # Terminal cost weights (for final state)
        AcadosParameter(
            "qe_diag_sqrt",
            default = qe_diag_sqrt,
            interface="learnable",
            space=gym.spaces.Box(
                low=qe_diag_sqrt * 0.01,
                high=qe_diag_sqrt * 100.0,
                dtype=np.float32,
            ),
        ),
        AcadosParameter(
            "sref", 
            default=np.array([track_length]),  # reference progress
            interface="non-learnable",
            vary_stages=list(range(N_horizon + 1))
            if param_interface == "stagewise"
            else [],
        ),
        AcadosParameter(
            "yref",
            default=np.array([-2, 0, 0, 0, 0, 0, 0, 0]),  # Match initial state s=-2
            interface="non-learnable",
            vary_stages=list(range(N_horizon))
            if param_interface == "stagewise"
            else [],
        ),
        AcadosParameter(
            "yref_e",
            default=np.array([-2, 0, 0, 0, 0, 0]),  # Match initial state s=-2
            interface="non-learnable",
        ),
    ]


def define_f_expl_expr(
    model: AcadosModel, param_manager: AcadosParameterManager, track_file: str = "LMS_Track.txt"
) -> ca.SX:
    m = param_manager.get("m")
    C1, C2 = param_manager.get("C1"), param_manager.get("C2")
    Cm1, Cm2 = param_manager.get("Cm1"), param_manager.get("Cm2")
    Cr0, Cr2 = param_manager.get("Cr0"), param_manager.get("Cr2")

    [s0, _, _, _, kapparef] = getTrack(track_file)
    length = len(s0)
    pathlength = s0[-1]

    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    kapparef = np.append(kapparef, kapparef[1:length])
    s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)
    kapparef_s = ca.interpolant("kapparef_s", "bspline", [s0], kapparef)

    # [s, n, alpha, v, D, delta]
    s = model.x[0]
    n = model.x[1]
    alpha = model.x[2]
    v = model.x[3]
    D = model.x[4]
    delta = model.x[5]
    x = ca.vertcat(s, n, alpha, v, D, delta)

    derD = model.u[0]
    derDelta = model.u[1]
    u = ca.vertcat(derD, derDelta)

    # Race car dynamics (from bicycle_model.py)
    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * ca.tanh(5 * v)
    
    # dynamics
    sdota = (v * ca.cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)

    f_expl = ca.vertcat(
        sdota,
        v * ca.sin(alpha + C1 * delta),
        v * C2 * delta - kapparef_s(s) * sdota,
        Fxd / m * ca.cos(C1 * delta),
        derD,
        derDelta,
    )
    constraint = types.SimpleNamespace()
    a_lat = C2 * v * v * delta + Fxd * ca.sin(C1 * delta) / m
    a_long = Fxd / m

    # Model bounds (like acados bicycle_model.py)
    constraint.n_min = -0.12  # width of the track [m]
    constraint.n_max = 0.12   # width of the track [m]
    constraint.throttle_min = -1.0
    constraint.throttle_max = 1.0
    constraint.delta_min = -0.40  # minimum steering angle [rad]
    constraint.delta_max = 0.40   # maximum steering angle [rad]
    constraint.dthrottle_min = -10  # minimum throttle change rate
    constraint.dthrottle_max = 10   # maximum throttle change rate
    constraint.ddelta_min = -2.0    # minimum change rate of steering angle [rad/s]
    constraint.ddelta_max = 2.0     # maximum change rate of steering angle [rad/s]

    # nonlinear constraint forces
    constraint.alat_min = -4  # maximum lateral force [m/s^2]
    constraint.alat_max = 4   # maximum lateral force [m/s^1]
    constraint.along_min = -4  # maximum longitudinal force [m/s^2]
    constraint.along_max = 4   # maximum longitudinal force [m/s^2]

    # define constraints struct
    constraint.alat = ca.Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    constraint.expr = ca.vertcat(a_long, a_lat, n, D, delta)
    return f_expl, constraint


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    cost_type: RaceCarAcadosCostType = "NONLINEAR_LS",
    name: str = "racecar",
    track_file: str = "LMS_Track.txt",
    N_horizon: int = 50,
    T_horizon: float = 1.0,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    ocp.model.name = name

    ocp.dims.nx = 6  # [s, n, alpha, v, D, delta]
    ocp.dims.nu = 2  # [derD, derDelta]

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    p = ca.vertcat(
        param_manager.non_learnable_parameters.cat,
        param_manager.learnable_parameters.cat,
    )
    f_expl, constraint = define_f_expl_expr(ocp.model, param_manager, track_file)

    ##########################
    # Check if we can use f_expl_expr directly or not
    # If not, we use an integrator to define the discrete dynamics
    ##########################
    # ocp.model.f_expl_expr = f_expl
    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=f_expl,
        x=ocp.model.x,
        u=ocp.model.u,
        p=p,
        dt=dt,
    )

    # Constraint
    ocp.model.con_h_expr = constraint.expr

    ######## Cost ########
    yref = param_manager.get("yref")  # [s,n,alpha,v,D,delta,derD,derDelta]
    yref_e = param_manager.get("yref_e")  # [s,n,alpha,v,D,delta]
    
    y = ca.vertcat(ocp.model.x, ocp.model.u)  # [s,n,alpha,v,D,delta,derD,derDelta]
    y_e = ocp.model.x  # [s,n,alpha,v,D,delta]
    
    # Create diagonal Q and R matrices from learnable parameters
    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    qe_diag_sqrt = param_manager.get("qe_diag_sqrt")
    
    # Construct cost matrices
    W = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))  # Combined state + control cost
    W_e = ca.diag(qe_diag_sqrt)  # Terminal cost only on states

    # Scale cost matrices
    unscale = N_horizon / T_horizon
    W = unscale * W
    W_e = W_e / unscale

    if cost_type == "LINEAR_LS":
        ocp.cost.cost_type = cost_type
        ocp.cost.cost_type_e = cost_type

        # Create Q and R matrices from sqrt parameters (like acados example)
        Q = np.diag(q_diag_sqrt ** 2)  # Convert back to full matrices
        R = np.diag(r_diag_sqrt ** 2)
        Qe = np.diag(qe_diag_sqrt ** 2)

        # Set cost matrices (exactly like acados example)
        ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Qe / unscale

        # Set transformation matrices (exactly like acados example)
        ny = ocp.dims.nx + ocp.dims.nu  # 8 = 6 + 2
        ny_e = ocp.dims.nx              # 6

        Vx = np.zeros((ny, ocp.dims.nx))
        Vx[:ocp.dims.nx, :ocp.dims.nx] = np.eye(ocp.dims.nx)
        ocp.cost.Vx = Vx

        Vu = np.zeros((ny, ocp.dims.nu))
        Vu[6, 0] = 1.0  # derD at position 6
        Vu[7, 1] = 1.0  # derDelta at position 7
        ocp.cost.Vu = Vu

        Vx_e = np.zeros((ny_e, ocp.dims.nx))
        Vx_e[:ocp.dims.nx, :ocp.dims.nx] = np.eye(ocp.dims.nx)
        ocp.cost.Vx_e = Vx_e

        # Set references (will be updated at runtime)
        ocp.cost.yref = yref
        ocp.cost.yref_e = yref_e

        # Solver options for LINEAR_LS (like acados example)
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    elif cost_type == "NONLINEAR_LS":
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
        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)
        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = 0.5 * (y_e - yref_e).T @ W_e @ (y_e - yref_e)
        ocp.solver_options.hessian_approx = "EXACT"

    ######## Constraints ########
    # Initial state constraint (will be set at runtime)
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5])
    ocp.constraints.x0 = np.array([-2, 0, 0, 0, 0, 0])

    # Control input bounds (using constraint object like acados)
    ocp.constraints.lbu = np.array([constraint.dthrottle_min, constraint.ddelta_min])
    ocp.constraints.ubu = np.array([constraint.dthrottle_max, constraint.ddelta_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # State bounds: lateral deviation
    ocp.constraints.lbx = np.array([constraint.n_min])  # n_min (track width)
    ocp.constraints.ubx = np.array([constraint.n_max])   # n_max
    ocp.constraints.idxbx = np.array([1])  # lateral deviation index

    # Terminal bounds
    ocp.constraints.lbx_e = np.array([constraint.n_min])
    ocp.constraints.ubx_e = np.array([constraint.n_max])
    ocp.constraints.idxbx_e = np.array([1])

    # Nonlinear constraints: accelerations, track bounds, input bounds (using constraint object like acados)
    ocp.constraints.lh = np.array([
        constraint.along_min,
        constraint.alat_min,
        constraint.n_min,
        constraint.throttle_min,
        constraint.delta_min,
    ])
    ocp.constraints.uh = np.array([
        constraint.along_max,
        constraint.alat_max,
        constraint.n_max,
        constraint.throttle_max,
        constraint.delta_max,
    ])

    # Soft constraints for acceleration and track bounds
    ns = 2
    ocp.constraints.lsh = np.zeros(ns)
    ocp.constraints.ush = np.zeros(ns)
    ocp.constraints.idxsh = np.array([0, 2])

    # Soft constraint costs
    ocp.cost.zl = 100 * np.ones(ns)
    ocp.cost.Zl = 0   * np.ones(ns)
    ocp.cost.zu = 100 * np.ones(ns)
    ocp.cost.Zu = 0   * np.ones(ns)

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1

    # Additional solver options from acados example (for f_expl_expr compatibility)
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    
    return ocp