from typing import Literal

import casadi as ca
import numpy as np

from acados_template import AcadosModel, AcadosOcp
from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.parameters import Parameter, AcadosParamManager


CartPoleAcadosParamInterface = Literal["global", "stagewise"]
CartPoleAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS"]


def create_cartpole_params(
    param_interface: CartPoleAcadosParamInterface,
    N_horizon: int = 50,
) -> list[Parameter]:
    """Returns a list of parameters used in cartpole."""
    return [
        # Dynamics parameters
        Parameter("M", np.array([1.0])),  # mass of the cart [kg]
        Parameter("m", np.array([0.1])),  # mass of the ball [kg]
        Parameter("g", np.array([9.81])),  # gravity constant [m/s^2]
        Parameter("l", np.array([0.8])),  # length of the rod [m]
        # Cost matrix factorization parameters
        Parameter(
            "q_diag_sqrt", np.sqrt(np.array([2e3, 2e3, 1e-2, 1e-2]))
        ),  # cost on state residuals
        Parameter(
            "r_diag_sqrt", np.sqrt(np.array([2e-1]))
        ),  # cost on control input residuals
        # Reference parameters
        Parameter(
            "xref0",
            np.array([0.0]),
            interface="non-learnable",
        ),  # reference position
        Parameter(
            "xref1",
            np.array([0.0]),
            lower_bound=np.array([-2.0 * np.pi]),
            upper_bound=np.array([2.0 * np.pi]),
            interface="learnable",
            vary_stages=[i for i in range(N_horizon + 1)]
            if param_interface == "stagewise"
            else [],
        ),  # reference theta
        Parameter(
            "xref2",
            np.array([0.0]),
            interface="non-learnable",
        ),  # reference v
        Parameter(
            "xref3",
            np.array([0.0]),
            interface="non-learnable",
        ),  # reference thetadot
        Parameter(
            "uref",
            np.array([0.0]),
            interface="non-learnable",
        ),  # reference u
    ]


def define_f_expl_expr(model: AcadosModel, param_manager: AcadosParamManager) -> ca.SX:
    M = param_manager.get("M")
    m = param_manager.get("m")
    g = param_manager.get("g")
    l = param_manager.get("l")

    theta = model.x[1]
    v = model.x[2]
    dtheta = model.x[3]

    F = model.u[0]

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m * cos_theta * cos_theta
    f_expl = ca.vertcat(
        v,
        dtheta,
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F)
        / denominator,
        (
            -m * l * cos_theta * sin_theta * dtheta * dtheta
            + F * cos_theta
            + (M + m) * g * sin_theta
        )
        / (l * denominator),
    )

    return f_expl  # type:ignore


def export_parametric_ocp(
    param_manager: AcadosParamManager,
    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS",
    name: str = "cartpole",
    Fmax: float = 80.0,
    x_threshold: float = 2.4,
    N_horizon: int = 50,
    T_horizon: float = 2.0,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    ######## Model ########
    ocp.model.name = name

    ocp.dims.nx = 4
    ocp.dims.nu = 1

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    p = ca.vertcat(
        param_manager.non_learnable_parameters.cat,
        param_manager.learnable_parameters.cat,
    )  # type:ignore
    f_expl = define_f_expl_expr(ocp.model, param_manager)
    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=f_expl,
        x=ocp.model.x,
        u=ocp.model.u,
        p=p,
        dt=dt,
    )

    ######## Cost ########
    xref = ca.vertcat(*[param_manager.get(f"xref{i}") for i in range(4)])
    uref = param_manager.get("uref")
    yref = ca.vertcat(xref, uref)  # type:ignore
    if isinstance(yref, ca.DM):
        yref = yref.full()
    yref_e = yref[: ocp.dims.nx]
    y = ca.vertcat(ocp.model.x, ocp.model.u)
    y_e = ocp.model.x

    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
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
        raise ValueError(
            f"Cost type {cost_type} not supported. Use 'EXTERNAL' or 'NONLINEAR_LS'."
        )

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

    return ocp
