from typing import Literal
import sys
import os

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosModel, AcadosOcp

from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

# Add the race car example path to sys.path
race_car_path = os.path.join(os.path.dirname(__file__), '../../../external/acados/examples/acados_python/race_cars')
if race_car_path not in sys.path:
    sys.path.insert(0, race_car_path)

try:
    from bicycle_model import bicycle_model
    from tracks.readDataFcn import getTrack
except ImportError as e:
    print(f"Warning: Could not import race car modules: {e}")

RaceCarAcadosParamInterface = Literal["global", "stagewise"]
RaceCarAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS"]


def create_racecar_params(
    param_interface: RaceCarAcadosParamInterface,
    N_horizon: int = 50,
    track_file: str = "LMS_Track.txt",
) -> list[AcadosParameter]:
    """Returns a list of parameters used in race car model."""
    
    # Load track to determine appropriate bounds
    try:
        Sref, _, _, _, _ = getTrack(track_file)
        track_length = Sref[-1]
    except:
        track_length = 10.0  # fallback
    
    return [
        # Race car dynamics parameters (from bicycle_model.py)
        AcadosParameter("m", default=np.array([0.043])),    # mass [kg]
        AcadosParameter("C1", default=np.array([0.5])),     # front tire parameter
        AcadosParameter("C2", default=np.array([15.5])),    # rear tire parameter
        AcadosParameter("Cm1", default=np.array([0.28])),   # motor parameter
        AcadosParameter("Cm2", default=np.array([0.05])),   # motor parameter
        AcadosParameter("Cr0", default=np.array([0.011])),  # rolling resistance
        AcadosParameter("Cr2", default=np.array([0.006])),  # air resistance
        
        # Cost matrix parameters - these will be learned by SAC
        # State cost weights: [s, n, alpha, v, D, delta]
        AcadosParameter(
            "Q_s", 
            default=np.array([1e-1]),  # progress weight
            interface="learnable",
            space=gym.spaces.Box(
                low=np.array([1e-3]),
                high=np.array([1.0]),
                dtype=np.float64,
            ),
        ),
        AcadosParameter(
            "Q_n", 
            default=np.array([1e1]),  # lateral deviation weight (important!)
            interface="learnable", 
            space=gym.spaces.Box(
                low=np.array([1.0]),
                high=np.array([100.0]),
                dtype=np.float64,
            ),
        ),
        AcadosParameter(
            "Q_alpha", 
            default=np.array([1e-8]),  # heading angle weight
            interface="learnable",
            space=gym.spaces.Box(
                low=np.array([1e-9]),
                high=np.array([1e-2]),
                dtype=np.float64,
            ),
        ),
        AcadosParameter(
            "Q_v", 
            default=np.array([1e-8]),  # velocity weight
            interface="learnable",
            space=gym.spaces.Box(
                low=np.array([1e-9]),
                high=np.array([1e-2]),
                dtype=np.float64,
            ),
        ),
        AcadosParameter(
            "Q_D", 
            default=np.array([1e-3]),  # throttle weight
            interface="learnable",
            space=gym.spaces.Box(
                low=np.array([1e-4]),
                high=np.array([1e-1]),
                dtype=np.float64,
            ),
        ),
        AcadosParameter(
            "Q_delta", 
            default=np.array([5e-3]),  # steering weight
            interface="learnable",
            space=gym.spaces.Box(
                low=np.array([1e-4]),
                high=np.array([1e-1]),
                dtype=np.float64,
            ),
        ),
        
        # Control cost weights: [derD, derDelta]
        AcadosParameter(
            "R_derD", 
            default=np.array([1e-3]),  # throttle rate weight
            interface="learnable",
            space=gym.spaces.Box(
                low=np.array([1e-4]),
                high=np.array([1e-1]),
                dtype=np.float64,
            ),
        ),
        AcadosParameter(
            "R_derDelta", 
            default=np.array([5e-3]),  # steering rate weight
            interface="learnable",
            space=gym.spaces.Box(
                low=np.array([1e-4]),
                high=np.array([1e-1]),
                dtype=np.float64,
            ),
        ),
        
        # Terminal cost weights (for final state)
        AcadosParameter(
            "Qe_s", 
            default=np.array([5e0]),  # terminal progress weight
            interface="learnable",
            space=gym.spaces.Box(
                low=np.array([1e-1]),
                high=np.array([50.0]),
                dtype=np.float64,
            ),
        ),
        AcadosParameter(
            "Qe_n", 
            default=np.array([1e1]),  # terminal lateral deviation weight
            interface="learnable",
            space=gym.spaces.Box(
                low=np.array([1.0]),
                high=np.array([100.0]),
                dtype=np.float64,
            ),
        ),
        
        # Reference parameters - these are set dynamically during racing
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
            default=np.zeros(8),  # reference state + control [s,n,alpha,v,D,delta,derD,derDelta]
            interface="non-learnable",
            vary_stages=list(range(N_horizon))
            if param_interface == "stagewise"
            else [],
        ),
        AcadosParameter(
            "yref_e", 
            default=np.zeros(6),  # terminal reference state [s,n,alpha,v,D,delta]
            interface="non-learnable",
        ),
    ]


def define_f_expl_expr(
    model: AcadosModel, param_manager: AcadosParameterManager, track_file: str = "LMS_Track.txt"
) -> ca.SX:
    """Define explicit dynamics for race car bicycle model."""
    
    # Get model parameters
    m = param_manager.get("m")
    C1 = param_manager.get("C1")
    C2 = param_manager.get("C2")
    Cm1 = param_manager.get("Cm1")
    Cm2 = param_manager.get("Cm2")
    Cr0 = param_manager.get("Cr0")
    Cr2 = param_manager.get("Cr2")
    
    # Load track curvature data
    try:
        s0, _, _, _, kapparef = getTrack(track_file)
        length = len(s0)
        pathlength = s0[-1]
        
        # Extend track data (as in original bicycle_model.py)
        s0_ext = ca.DM(np.append(s0, [s0[length - 1] + s0[1:length]]))
        kapparef_ext = ca.DM(np.append(kapparef, kapparef[1:length]))
        s0_ext = ca.DM(np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0_ext))
        kapparef_ext = ca.DM(np.append(kapparef[length - 80 : length - 1], kapparef_ext))
        
        # Create curvature interpolant
        kapparef_s = ca.interpolant("kapparef_s", "bspline", [s0_ext], kapparef_ext)
    except:
        # Fallback: assume straight track
        kapparef_s = lambda s: 0.0
    
    # State variables: [s, n, alpha, v, D, delta]
    s = model.x[0]      # progress along track [m]
    n = model.x[1]      # lateral deviation [m]
    alpha = model.x[2]  # heading angle relative to track [rad]
    v = model.x[3]      # velocity [m/s]
    D = model.x[4]      # throttle/brake input [-1, 1]
    delta = model.x[5]  # steering angle [rad]
    
    # Control variables: [derD, derDelta]
    derD = model.u[0]      # throttle rate
    derDelta = model.u[1]  # steering rate
    
    # Race car dynamics (from bicycle_model.py)
    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * ca.tanh(5 * v)
    
    # State derivatives
    kappa = kapparef_s(s)
    sdot = (v * ca.cos(alpha + C1 * delta)) / (1 - kappa * n)
    ndot = v * ca.sin(alpha + C1 * delta)
    alphadot = v * C2 * delta - kappa * sdot
    vdot = Fxd / m * ca.cos(C1 * delta)
    Ddot = derD
    deltadot = derDelta
    
    f_expl = ca.vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot)
    
    return f_expl


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
    
    ######## Model ########
    ocp.model.name = name
    
    ocp.dims.nx = 6  # [s, n, alpha, v, D, delta]
    ocp.dims.nu = 2  # [derD, derDelta]
    
    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)
    
    p = ca.vertcat(
        param_manager.non_learnable_parameters.cat,
        param_manager.learnable_parameters.cat,
    )
    
    f_expl = define_f_expl_expr(ocp.model, param_manager, track_file)
    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=f_expl,
        x=ocp.model.x,
        u=ocp.model.u,
        p=p,
        dt=dt,
    )
    
    # Add constraints for lateral and longitudinal accelerations
    # (as in the original race car example)
    m = param_manager.get("m")
    C1 = param_manager.get("C1")
    C2 = param_manager.get("C2")
    Cm1 = param_manager.get("Cm1")
    Cm2 = param_manager.get("Cm2")
    Cr0 = param_manager.get("Cr0")
    Cr2 = param_manager.get("Cr2")
    
    s, n, alpha, v, D, delta = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2], ocp.model.x[3], ocp.model.x[4], ocp.model.x[5]
    derD, derDelta = ocp.model.u[0], ocp.model.u[1]
    
    # Force constraints
    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * ca.tanh(5 * v)
    a_lat = C2 * v * v * delta + Fxd * ca.sin(C1 * delta) / m
    a_long = Fxd / m
    
    ocp.model.con_h_expr = ca.vertcat(a_long, a_lat, n, D, delta)
    
    ######## Cost ########
    yref = param_manager.get("yref")  # [s,n,alpha,v,D,delta,derD,derDelta]
    yref_e = param_manager.get("yref_e")  # [s,n,alpha,v,D,delta]
    
    y = ca.vertcat(ocp.model.x, ocp.model.u)  # [s,n,alpha,v,D,delta,derD,derDelta]
    y_e = ocp.model.x  # [s,n,alpha,v,D,delta]
    
    # Create diagonal Q and R matrices from learnable parameters
    Q_diag = ca.vertcat(
        param_manager.get("Q_s"),
        param_manager.get("Q_n"),
        param_manager.get("Q_alpha"),
        param_manager.get("Q_v"),
        param_manager.get("Q_D"),
        param_manager.get("Q_delta"),
    )
    
    R_diag = ca.vertcat(
        param_manager.get("R_derD"),
        param_manager.get("R_derDelta"),
    )
    
    # Terminal cost weights
    Qe_diag = ca.vertcat(
        param_manager.get("Qe_s"),
        param_manager.get("Qe_n"),
        param_manager.get("Q_alpha"),  # reuse intermediate cost
        param_manager.get("Q_v"),      # reuse intermediate cost
        param_manager.get("Q_D"),      # reuse intermediate cost
        param_manager.get("Q_delta"),  # reuse intermediate cost
    )
    
    # Construct cost matrices
    W = ca.diag(ca.vertcat(Q_diag, R_diag))  # Combined state + control cost
    W_e = ca.diag(Qe_diag)  # Terminal cost only on states
    
    # Scale cost matrices (as in original example)
    unscale = N_horizon / T_horizon
    W = unscale * W
    W_e = W_e / unscale
    
    if cost_type == "EXTERNAL":
        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)
        
        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = 0.5 * (y_e - yref_e).T @ W_e @ (y_e - yref_e)
        
        ocp.solver_options.hessian_approx = "EXACT"
        
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
    else:
        raise ValueError(
            f"Cost type {cost_type} not supported. Use 'EXTERNAL' or 'NONLINEAR_LS'."
        )
    
    ######## Constraints ########
    # Initial state constraint (will be set at runtime)
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5])
    ocp.constraints.x0 = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Default start
    
    # Control input bounds (from bicycle_model.py)
    ocp.constraints.lbu = np.array([-10.0, -2.0])  # [dthrottle_min, ddelta_min]
    ocp.constraints.ubu = np.array([10.0, 2.0])    # [dthrottle_max, ddelta_max]
    ocp.constraints.idxbu = np.array([0, 1])
    
    # State bounds: lateral deviation
    ocp.constraints.lbx = np.array([-0.12])  # n_min (track width)
    ocp.constraints.ubx = np.array([0.12])   # n_max
    ocp.constraints.idxbx = np.array([1])    # lateral deviation index
    
    # Terminal bounds
    ocp.constraints.lbx_e = np.array([-0.12])
    ocp.constraints.ubx_e = np.array([0.12])
    ocp.constraints.idxbx_e = np.array([1])
    
    # Nonlinear constraints: accelerations, track bounds, input bounds
    ocp.constraints.lh = np.array([-4.0, -4.0, -0.12, -1.0, -0.40])  # [a_long_min, a_lat_min, n_min, throttle_min, delta_min]
    ocp.constraints.uh = np.array([4.0, 4.0, 0.12, 1.0, 0.40])       # [a_long_max, a_lat_max, n_max, throttle_max, delta_max]
    
    # Soft constraints for acceleration and track bounds
    nsh = 2  # number of soft constraints
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array([0, 2])  # soften acceleration and track bounds
    
    # Soft constraint costs
    ocp.cost.zl = 100 * np.ones(nsh)
    ocp.cost.Zl = 0 * np.ones(nsh)
    ocp.cost.zu = 100 * np.ones(nsh)
    ocp.cost.Zu = 0 * np.ones(nsh)
    
    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"  # Required for sensitivity computation
    ocp.solver_options.nlp_solver_type = "SQP"  # Use SQP instead of SQP_RTI for learning
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    
    return ocp