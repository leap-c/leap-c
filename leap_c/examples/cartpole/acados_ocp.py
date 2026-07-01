from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.utils.parameters import ParamSplits, n_segments

CartPoleAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS"]
"""The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS". Both model the same cost function, 
but the former uses an exact Hessian in the optimization, while the latter uses a 
Gauss-Newton Hessian approximation.
"""


def export_parametric_ocp(
    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS",
    name: str = "cartpole",
    Fmax: float = 80.0,
    x_threshold: float = 2.4,
    N_horizon: int = 50,
    T_horizon: float = 2.0,
    param_splits: ParamSplits = "global",
) -> tuple[AcadosOcp, AcadosParameterManager, gym.spaces.Box, np.ndarray]:
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    manager = AcadosParameterManager(N_horizon=N_horizon)

    # Learnable reference parameter xref1.
    # `param_splits` controls how xref1 varies across the MPC horizon:
    #   - "global" (default): one value shared across all stages -> shape (1,).
    #   - "stagewise":        one independent value per stage     -> shape (N+1, 1).
    #   - list[int], e.g. [2, N]: explicit stage boundaries         -> shape (len(splits), 1).
    #
    # For "global" the default / bounds are passed directly. For other splits,
    # broadcast the per-segment default to the per-stage shape, e.g.
    # n_segments("stagewise", 5) -> 6 -> np.broadcast_to(default, (6, 1)).
    xref0 = manager.register_parameter("xref0", default=np.array([0.0]))
    default_xref1 = np.array([0.0])
    xref1 = manager.register_parameter(
        "xref1",
        default=default_xref1,
        differentiable=True,
        splits=param_splits,
    )
    n = n_segments(param_splits, N_horizon)
    if n == 1:
        default_param = default_xref1
    else:
        default_param = np.broadcast_to(default_xref1, (n, *default_xref1.shape))
    param_space = gym.spaces.Box(
        low=-2.0 * np.pi,
        high=2.0 * np.pi,
        shape=default_param.shape,
        dtype=np.float64,
    )

    xref2 = manager.register_parameter("xref2", default=np.array([0.0]))
    xref3 = manager.register_parameter("xref3", default=np.array([0.0]))
    uref = manager.register_parameter("uref", default=np.array([0.0]))

    # Dynamics physical constants (sunsetted "fix" interface)
    M = 1.0  # mass of the cart [kg]
    m = 0.1  # mass of the ball [kg]
    g = 9.81  # gravity constant [m/s^2]
    l = 0.8  # length of the rod [m]

    q_diag_sqrt = np.sqrt(np.array([2e3, 2e3, 1e-2, 1e-2]))
    r_diag_sqrt = np.sqrt(np.array([2e-1]))

    ######## Model ########
    ocp.model.name = name

    ocp.dims.nx = 4
    ocp.dims.nu = 1

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    theta = ocp.model.x[1]
    v = ocp.model.x[2]
    dtheta = ocp.model.x[3]

    F = ocp.model.u[0]

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m * cos_theta * cos_theta
    f_expl = ca.vertcat(
        v,
        dtheta,
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F) / denominator,
        (-m * l * cos_theta * sin_theta * dtheta * dtheta + F * cos_theta + (M + m) * g * sin_theta)
        / (l * denominator),
    )

    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=f_expl,
        x=ocp.model.x,
        u=ocp.model.u,
        p=ca.vertcat(manager.learnable_symbols, manager.non_learnable_symbols),
        dt=dt,
    )

    ######## Cost ########
    xref = ca.vertcat(xref0, xref1, xref2, xref3)
    yref = ca.vertcat(xref, uref)  # type:ignore
    yref_e = yref[: ocp.dims.nx]
    y = ca.vertcat(ocp.model.x, ocp.model.u)
    y_e = ocp.model.x

    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    W = W_sqrt @ W_sqrt.T
    W_e = W[: ocp.dims.nx, : ocp.dims.nx]

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
        raise ValueError(f"Cost type {cost_type} not supported. Use 'EXTERNAL' or 'NONLINEAR_LS'.")

    ######## Constraints ########
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([-x_threshold])
    ocp.constraints.ubx = -ocp.constraints.lbx
    ocp.constraints.idxbx = np.array([0])
    ocp.constraints.lbx_e = np.array([-x_threshold])
    ocp.constraints.ubx_e = -ocp.constraints.lbx_e
    ocp.constraints.idxbx_e = np.array([0])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.Zu = ocp.cost.Zl = np.array([1e3])
    ocp.cost.zu = ocp.cost.zl = np.array([0.0])

    ocp.constraints.idxsbx_e = np.array([0])
    ocp.cost.Zu_e = ocp.cost.Zl_e = np.array([1e3])
    ocp.cost.zu_e = ocp.cost.zl_e = np.array([0.0])

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1

    return ocp, manager, param_space, default_param
