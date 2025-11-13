from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

PointMassAcadosParamInterface = Literal["global", "stagewise"]
"""Determines the exposed parameter interface of the controller.
"global" means that learnable parameters are the same for all stages of the horizon,
while "stagewise" means that learnable parameters can vary between stages.
"""


def create_pointmass_params(
    param_interface: PointMassAcadosParamInterface,
    x_ref_value: np.ndarray | None = None,
    N_horizon: int = 20,
) -> list[AcadosParameter]:
    """Returns a list of parameters used in the pointmass controller.

    Args:
        param_interface: Determines the exposed parameter interface of the controller.
        x_ref_value: The value for the reference state.
        N_horizon: The number of steps in the MPC horizon.
    """
    q_diag_sqrt = np.array([1.0, 1.0, 1.0, 1.0])
    r_diag_sqrt = np.array([0.1, 0.1])

    if x_ref_value is None:
        x_ref_value = np.array([0.0, 0.0, 0.0, 0.0])

    return [
        # mass and friction parameters
        AcadosParameter("m", default=np.array([1.0])),  # mass [kg]
        AcadosParameter("cx", default=np.array([0.1])),  # x friction coefficient
        AcadosParameter("cy", default=np.array([0.1])),  # y friction coefficient
        # cost function parameters
        AcadosParameter(
            "q_diag_sqrt",  # weight for state residuals
            default=q_diag_sqrt,
            space=gym.spaces.Box(low=0.5 * q_diag_sqrt, high=1.5 * q_diag_sqrt, dtype=np.float64),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),
        AcadosParameter(
            "r_diag_sqrt",  # weight for control residuals
            default=r_diag_sqrt,
            space=gym.spaces.Box(low=0.5 * r_diag_sqrt, high=1.5 * r_diag_sqrt, dtype=np.float64),
            interface="learnable",
            end_stages=list(range(N_horizon)) if param_interface == "stagewise" else [],
        ),
        AcadosParameter(
            "x_ref",
            default=x_ref_value,
            space=gym.spaces.Box(
                low=np.array([0.0, 0.0, -20, -20]),
                high=np.array([4.0, 1.0, 20, 20]),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),  # state reference
        AcadosParameter(
            "u_ref",
            default=np.array([0.0, 0.0]),
            space=gym.spaces.Box(
                low=np.array([-10.0, -10.0]),
                high=np.array([10.0, 10.0]),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=list(range(N_horizon)) if param_interface == "stagewise" else [],
        ),  # action reference
    ]


def define_disc_dyn_expr(
    ocp: AcadosOcp,
    param_manager: AcadosParameterManager,
) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    m = param_manager.get("m").item()
    cx = param_manager.get("cx").item()
    cy = param_manager.get("cy").item()
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon  # type: ignore

    A = ca.vertcat(
        ca.horzcat(1, 0, dt, 0),
        ca.horzcat(0, 1, 0, dt),
        ca.horzcat(0, 0, (m / cx) * (1 - ca.exp(-cx * dt / m)), 0),
        ca.horzcat(0, 0, 0, (m / cy) * (1 - ca.exp(-cy * dt / m))),
    )
    B = ca.vertcat(
        ca.horzcat(0, 0),
        ca.horzcat(0, 0),
        ca.horzcat((m / cx) * (1 - ca.exp(-cx * dt / m)), 0),
        ca.horzcat(0, (m / cy) * (1 - ca.exp(-cy * dt / m))),
    )

    return A @ x + B @ u


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    name: str = "pointmass",
    Fmax: float = 10.0,
    N_horizon: int = 20,
    T_horizon: float = 2.0,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    ######## Model ########
    ocp.model.name = name

    ocp.dims.nu = 2
    ocp.dims.nx = 4

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)  # type: ignore
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)  # type: ignore

    ocp.model.disc_dyn_expr = define_disc_dyn_expr(ocp=ocp, param_manager=param_manager)

    ######## Cost ########
    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    x_ref = param_manager.get("x_ref")
    u_ref = param_manager.get("u_ref")
    Q = ca.diag(q_diag_sqrt**2)
    R = ca.diag(r_diag_sqrt**2)
    x_res = ocp.model.x - x_ref
    u_res = ocp.model.u - u_ref

    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = 0.5 * (x_res.T @ Q @ x_res + u_res.T @ R @ u_res)
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = 0.5 * (x_res.T @ Q @ x_res)

    ######## Constraints ########
    ocp.constraints.x0 = np.array([1.0, 1.0, 0.0, 0.0])

    # Box constraints on u
    ocp.constraints.lbu = np.array([-Fmax, -Fmax])
    ocp.constraints.ubu = np.array([Fmax, Fmax])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([0.05, 0.05, -20.0, -20.0])
    ocp.constraints.ubx = np.array([3.95, 0.95, 20.0, 20.0])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3])

    ocp.constraints.idxsbx = np.array([0, 1, 2, 3])

    ns = ocp.constraints.idxsbx.size
    ocp.cost.zl = 10000 * np.ones((ns,))
    ocp.cost.Zl = 10 * np.ones((ns,))
    ocp.cost.zu = 10000 * np.ones((ns,))
    ocp.cost.Zu = 10 * np.ones((ns,))

    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1

    return ocp
