from typing import Literal

import casadi as ca
import numpy as np

from acados_template import AcadosModel, AcadosOcp
from leap_c.ocp.acados.parameters import Parameter, AcadosParamManager


CartPoleAcadosParamInterface = Literal["global", "stagewise"]
CartPoleAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS"]


def create_cartpole_params(param_interface: CartPoleAcadosParamInterface) -> list[Parameter]:
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
        Parameter("xref0", np.array([0.0]), fix=False),  # reference position
        Parameter(
            "xref1",
            np.array([0.0]),
            lower_bound=np.array([-2.0 * np.pi]),
            upper_bound=np.array([2.0 * np.pi]),
            differentiable=True,
            stagewise=True if param_interface == "stagewise" else False,
            fix=False,
        ),  # reference theta
        Parameter("xref2", np.array([0.0]), fix=False),  # reference v
        Parameter("xref3", np.array([0.0]), fix=False),  # reference thetadot
        Parameter("uref", np.array([0.0]), fix=False),  # reference u
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


def define_disc_dyn_expr(
    model: AcadosModel, param_manager: AcadosParamManager, dt: float
) -> ca.SX:
    f_expl = define_f_expl_expr(model, param_manager)

    x = model.x
    u = model.u

    # discrete dynamics via RK4
    p = ca.vertcat(param_manager.p.cat, param_manager.p_global.cat)

    ode = ca.Function("ode", [x, u, p], [f_expl])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)  # type:ignore
    k3 = ode(x + dt / 2 * k2, u, p)  # type:ignore
    k4 = ode(x + dt * k3, u, p)  # type:ignore

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # type:ignore


def define_cost_matrix(
    model: AcadosModel, param_manager: AcadosParamManager
) -> ca.SX | np.ndarray:
    q_diag = param_manager.get("q_diag_sqrt")
    r_diag = param_manager.get("r_diag_sqrt")

    W = ca.diag(ca.vertcat(q_diag, r_diag))
    W = W @ W.T

    return W


def define_yref(param_manager: AcadosParamManager):
    xref0 = param_manager.get("xref0")
    xref1 = param_manager.get("xref1")
    xref2 = param_manager.get("xref2")
    xref3 = param_manager.get("xref3")
    xref = ca.vertcat(xref0, xref1, xref2, xref3)  # type:ignore
    uref = param_manager.get("uref")

    yref = ca.vertcat(xref, uref)  # type:ignore

    if isinstance(yref, ca.DM):
        yref = yref.full()

    return yref


def define_cost_expr_ext_cost(
    ocp: AcadosOcp, param_manager: AcadosParamManager
) -> ca.SX:
    yref = define_yref(param_manager)
    W = define_cost_matrix(ocp.model, param_manager)
    y = ca.vertcat(ocp.model.x, ocp.model.u)  # type:ignore
    return 0.5 * (y - yref).T @ W @ (y - yref)


def define_cost_expr_ext_cost_e(
    ocp: AcadosOcp, param_manager: AcadosParamManager
) -> ca.SX:
    yref = define_yref(param_manager)
    yref_e = yref[: ocp.dims.nx]  # type:ignore

    W = define_cost_matrix(ocp.model, param_manager)
    W_e = W[: ocp.dims.nx, : ocp.dims.nx]  # type:ignore
    y_e = ocp.model.x
    return 0.5 * (y_e - yref_e).T @ W_e @ (y_e - yref_e)


def export_parametric_ocp(
    param_manager: AcadosParamManager,
    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS",
    exact_hess_dyn: bool = True,
    name: str = "cartpole",
    Fmax: float = 80.0,
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

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)  # type:ignore
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)  # type:ignore

    ocp.model.disc_dyn_expr = define_disc_dyn_expr(
        model=ocp.model, param_manager=param_manager, dt=dt
    )  # type:ignore

    ######## Cost ########
    if cost_type == "EXTERNAL":
        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = define_cost_expr_ext_cost(
            ocp, param_manager
        )  # type:ignore

        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = define_cost_expr_ext_cost_e(
            ocp, param_manager
        )  # type:ignore

        ocp.solver_options.hessian_approx = "EXACT"
    elif cost_type == "NONLINEAR_LS":
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ocp.cost.W = define_cost_matrix(ocp.model, param_manager=param_manager)
        ocp.cost.yref = define_yref(param_manager=param_manager)
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)

        ocp.cost.W_e = ocp.cost.W[: ocp.dims.nx, : ocp.dims.nx]  # type:ignore
        ocp.cost.yref_e = ocp.cost.yref[: ocp.dims.nx]  # type:ignore
        ocp.model.cost_y_expr_e = ocp.model.cost_y_expr[: ocp.dims.nx]  # type:ignore

        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    else:
        raise ValueError(f"Cost type {cost_type} not supported.")

    ######## Constraints ########
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([-2.4])
    ocp.constraints.ubx = -ocp.constraints.lbx
    ocp.constraints.idxbx = np.array([0])
    ocp.constraints.lbx_e = np.array([-2.4])
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

    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1

    return ocp
