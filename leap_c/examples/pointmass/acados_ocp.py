from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosParameterManagerTorch

PointMassAcadosParamInterface = Literal["global", "stagewise"]


def export_parametric_ocp(
    param_interface: PointMassAcadosParamInterface,
    name: str = "pointmass",
    Fmax: float = 10.0,
    N_horizon: int = 20,
    T_horizon: float = 2.0,
    x_ref_value: np.ndarray | None = None,
) -> tuple[AcadosOcp, AcadosParameterManager]:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    manager = AcadosParameterManagerTorch(N_horizon=N_horizon)

    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    if x_ref_value is None:
        x_ref_value = np.array([0.0, 0.0, 0.0, 0.0])

    q_diag_sqrt_val = np.array([1.0, 1.0, 1.0, 1.0])
    r_diag_sqrt_val = np.array([0.1, 0.1])

    # Register learnable parameters
    q_diag_sqrt = manager.register_parameter(
        "q_diag_sqrt",
        default=q_diag_sqrt_val,
        space=gym.spaces.Box(
            low=0.5 * q_diag_sqrt_val, high=1.5 * q_diag_sqrt_val, dtype=np.float64
        ),
        differentiable=True,
        splits="stagewise" if param_interface == "stagewise" else "global",
    )
    r_diag_sqrt = manager.register_parameter(
        "r_diag_sqrt",
        default=r_diag_sqrt_val,
        space=gym.spaces.Box(
            low=0.5 * r_diag_sqrt_val, high=1.5 * r_diag_sqrt_val, dtype=np.float64
        ),
        differentiable=True,
        splits="stagewise" if param_interface == "stagewise" else "global",
    )
    x_ref = manager.register_parameter(
        "x_ref",
        default=x_ref_value,
        space=gym.spaces.Box(
            low=np.array([0.0, 0.0, -20.0, -20.0]),
            high=np.array([4.0, 1.0, 20.0, 20.0]),
            dtype=np.float64,
        ),
        differentiable=True,
        splits="stagewise" if param_interface == "stagewise" else "global",
    )
    u_ref = manager.register_parameter(
        "u_ref",
        default=np.array([0.0, 0.0]),
        space=gym.spaces.Box(
            low=np.array([-10.0, -10.0]),
            high=np.array([10.0, 10.0]),
            dtype=np.float64,
        ),
        differentiable=True,
        splits="stagewise" if param_interface == "stagewise" else "global",
    )

    # Dynamics physical constants (sunsetted "fix" interface)
    m = 1.0  # mass [kg]
    cx = 0.1  # x friction coefficient
    cy = 0.1  # y friction coefficient

    ######## Model ########
    ocp.model.name = name

    ocp.dims.nu = 2
    ocp.dims.nx = 4

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    # Parametric dynamics matrices for pointmass
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

    ocp.model.disc_dyn_expr = A @ ocp.model.x + B @ ocp.model.u

    ######## Cost ########
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

    return ocp, manager
