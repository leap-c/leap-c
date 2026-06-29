from collections import OrderedDict

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.utils.parameters import stagewise_broadcast


def export_parametric_ocp(
    name: str = "pointmass",
    Fmax: float = 10.0,
    N_horizon: int = 20,
    T_horizon: float = 2.0,
    x_ref_value: np.ndarray | None = None,
) -> tuple[AcadosOcp, AcadosParameterManager, gym.spaces.Dict, dict[str, np.ndarray]]:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    manager = AcadosParameterManager(N_horizon=N_horizon)

    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    if x_ref_value is None:
        x_ref_value = np.array([0.0, 0.0, 0.0, 0.0])

    q_diag_sqrt_val = np.array([1.0, 1.0, 1.0, 1.0])
    r_diag_sqrt_val = np.array([0.1, 0.1])

    splits = "global"
    spaces: OrderedDict[str, gym.spaces.Box] = OrderedDict()

    def register_learnable(
        name: str, default: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> ca.SX | ca.MX:
        symbol = manager.register_parameter(
            name,
            default=default,
            differentiable=True,
            splits=splits,
        )
        spaces[name] = gym.spaces.Box(
            low=stagewise_broadcast(low, splits, N_horizon),
            high=stagewise_broadcast(high, splits, N_horizon),
            dtype=np.float64,
        )
        return symbol

    q_diag_sqrt = register_learnable(
        "q_diag_sqrt",
        default=q_diag_sqrt_val,
        low=0.5 * q_diag_sqrt_val,
        high=1.5 * q_diag_sqrt_val,
    )
    r_diag_sqrt = register_learnable(
        "r_diag_sqrt",
        default=r_diag_sqrt_val,
        low=0.5 * r_diag_sqrt_val,
        high=1.5 * r_diag_sqrt_val,
    )
    x_ref = register_learnable(
        "x_ref",
        default=x_ref_value,
        low=np.array([0.0, 0.0, -20.0, -20.0]),
        high=np.array([4.0, 1.0, 20.0, 20.0]),
    )
    u_ref = register_learnable(
        "u_ref",
        default=np.array([0.0, 0.0]),
        low=np.array([-10.0, -10.0]),
        high=np.array([10.0, 10.0]),
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

    param_space = gym.spaces.Dict(spaces)
    default_param = manager.default_param_dict(param_space.keys())

    return ocp, manager, param_space, default_param
