import numpy as np
from acados_template import AcadosOcp
from casadi.tools import struct_symSX
from leap_c.linear_mpc import LinearMPC
import casadi as ca
from leap_c.examples.util import (
    translate_learnable_param_to_p_global,
    find_param_in_p_or_p_global,
)


class PointmassMPC(LinearMPC):
    """docstring for PointmassMPC."""

    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 20,
        discount_factor: float = 0.99,
        n_batch: int = 1,
    ):
        params = (
            {
                "m": 1.0,
                "c": 0.1,
                "q_diag": np.array([1.0, 1.0, 1.0, 1.0]),
                "r_diag": np.array([1.0, 1.0]),
                "q_diag_e": np.array([1.0, 1.0, 1.0, 1.0]),
            }
            if params is None
            else params
        )

        learnable_params = learnable_params if learnable_params is not None else []

        print("learnable_params: ", learnable_params)

        ocp = export_parametric_ocp(
            nominal_param=params, learnable_params=learnable_params, N_horizon=N_horizon
        )
        configure_ocp_solver(ocp=ocp, exact_hess_dyn=True)

        self.given_default_param_dict = params
        super().__init__(ocp=ocp, n_batch=n_batch)


def _A_disc(
    m: float | ca.SX, c: float | ca.SX, dt: float | ca.SX
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [m, c, dt]):
        a = ca.exp(-c * dt / m)
        return ca.vertcat(
            ca.horzcat(1, 0, dt, 0),
            ca.horzcat(0, 1, 0, dt),
            ca.horzcat(0, 0, a, 0),
            ca.horzcat(0, 0, 0, a),
        )
    else:
        a = np.exp(-c * dt / m)
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, a, 0],
                [0, 0, 0, a],
            ]
        )


def _B_disc(
    m: float | ca.SX, c: float | ca.SX, dt: float | ca.SX
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [m, c, dt]):
        b = (m / c) * (1 - ca.exp(-c * dt / m))
        return ca.vertcat(
            ca.horzcat(0, 0),
            ca.horzcat(0, 0),
            ca.horzcat(b, 0),
            ca.horzcat(0, b),
        )
    else:
        b = (m / c) * (1 - np.exp(-c * dt / m))
        return np.array(
            [
                [0, 0],
                [0, 0],
                [b, 0],
                [0, b],
            ]
        )


def _A_cont(
    m: float | ca.SX, c: float | ca.SX, dt: float | ca.SX
) -> np.ndarray | ca.SX:
    if isinstance(m, float):
        return np.array(
            [
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0],
                [0, 0, -(c / m), 0],
                [0, 0, 0, -(c / m)],
            ]
        )
    else:
        return ca.vertcat(
            ca.horzcat(0, 0, 1.0, 0),
            ca.horzcat(0, 0, 0, 1.0),
            ca.horzcat(0, 0, -(c / m), 0),
            ca.horzcat(0, 0, 0, -(c / m)),
        )


def _B_cont(
    m: float | ca.SX, c: float | ca.SX, dt: float | ca.SX
) -> np.ndarray | ca.SX:
    if isinstance(m, float):
        return np.array([[0, 0], [0, 0], [1.0 / m, 0], [0, 1.0 / m]])
    else:
        return ca.vertcat(
            ca.horzcat(0, 0),
            ca.horzcat(0, 0),
            ca.horzcat(1.0 / m, 0),
            ca.horzcat(0, 1.0 / m),
        )


def _Q_sqrt(
    _q_sqrt: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [_q_sqrt]):
        return ca.diag(_q_sqrt)
    else:
        return np.diag(_q_sqrt)


def _R_sqrt(
    _r_diag: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [_r_diag]):
        return ca.diag(_r_diag)
    else:
        return np.diag(_r_diag)


def _disc_dyn_expr(
    ocp: AcadosOcp,
) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    m = find_param_in_p_or_p_global(["m"], ocp.model)["m"]
    c = find_param_in_p_or_p_global(["c"], ocp.model)["c"]
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    A = _A_disc(m=m, c=c, dt=dt)
    B = _B_disc(m=m, c=c, dt=dt)

    return A @ x + B @ u


def _cost_expr_ext_cost(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    Q_sqrt = _Q_sqrt(find_param_in_p_or_p_global(["q_diag"], ocp.model)["q_diag"])
    R_sqrt = _R_sqrt(find_param_in_p_or_p_global(["r_diag"], ocp.model)["r_diag"])

    return 0.5 * (
        ca.mtimes([ca.transpose(x), Q_sqrt.T, Q_sqrt, x])
        + ca.mtimes([ca.transpose(u), R_sqrt.T, R_sqrt, u])
    )


def _cost_expr_ext_cost_e(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x

    Q_sqrt_e = _Q_sqrt(find_param_in_p_or_p_global(["q_diag_e"], ocp.model)["q_diag_e"])

    return 0.5 * ca.mtimes([ca.transpose(x), Q_sqrt_e.T, Q_sqrt_e, x])


def export_parametric_ocp(
    nominal_param: dict[str, np.ndarray],
    cost_type: str = "EXTERNAL",
    name: str = "pointmass",
    learnable_params: list[str] | None = None,
    N_horizon: int = 50,
    tf: float = 2.0,
    x0: np.ndarray = np.array([1.0, 1.0, 0.0, 0.0]),
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = N_horizon

    ocp.model.name = "pointmass"

    # ocp.dims.nx = 4
    # ocp.dims.nu = 2

    ocp.model.x = ca.SX.sym("x", 4)
    ocp.model.u = ca.SX.sym("u", 2)

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_param, learnable_param=learnable_params, ocp=ocp
    )

    ocp.model.disc_dyn_expr = _disc_dyn_expr(ocp=ocp)
    ocp.model.cost_expr_ext_cost = _cost_expr_ext_cost(ocp=ocp)
    ocp.cost.cost_type = cost_type
    ocp.model.cost_expr_ext_cost_e = _cost_expr_ext_cost_e(ocp=ocp)
    ocp.cost.cost_type_e = cost_type

    ocp.constraints.x0 = x0

    # #############################
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp


def configure_ocp_solver(ocp: AcadosOcp, exact_hess_dyn: bool):
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = True
