import numpy as np
from acados_template import AcadosOcp
from casadi.tools import struct_symSX, entry
from leap_c.linear_mpc import LinearMPC
from leap_c.mpc import MPC
import casadi as ca
from leap_c.examples.util import (
    translate_learnable_param_to_p_global,
    find_param_in_p_or_p_global,
)
from leap_c.examples.pointmass.env import _A_disc, _B_disc, WindParam, get_wind_velocity
from pathlib import Path


# class PointMassMPC(LinearMPC):
class PointMassMPC(MPC):
    """docstring for PointMassMPC."""

    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 20,
        T_horizon: float = 2.0,
        discount_factor: float = 0.99,
        n_batch: int = 64,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
        wind_param: WindParam | None = None,
    ):
        params = (
            {
                "m": 1.0,
                "cx": 0.1,
                "cy": 0.1,
                "q_diag": np.array([1.0, 1.0, 1.0, 1.0]),
                "r_diag": np.array([0.1, 0.1]),
                "q_diag_e": np.array([1.0, 1.0, 1.0, 1.0]),
                "xref": np.array([0.0, 0.0, 0.0, 0.0]),
                "uref": np.array([0.0, 0.0]),
                "xref_e": np.array([0.0, 0.0, 0.0, 0.0]),
                "u_wind": np.array([0.0, 0.0]),
            }
            if params is None
            else params
        )

        learnable_params = learnable_params if learnable_params is not None else []

        print("learnable_params: ", learnable_params)

        ocp = export_parametric_ocp(
            nominal_param=params,
            learnable_params=learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
            wind_param=wind_param,
        )
        configure_ocp_solver(ocp=ocp, exact_hess_dyn=True)

        self.given_default_param_dict = params
        super().__init__(
            ocp=ocp,
            n_batch=n_batch,
            export_directory=export_directory,
            export_directory_sensitivity=export_directory_sensitivity,
            throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        )


def _create_diag_matrix(
    _q_sqrt: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [_q_sqrt]):
        return ca.diag(_q_sqrt)
    else:
        return np.diag(_q_sqrt)


def _disc_dyn_expr(
    ocp: AcadosOcp,
    wind_param: WindParam | None = None,
) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    m = find_param_in_p_or_p_global(["m"], ocp.model)["m"]
    cx = find_param_in_p_or_p_global(["cx"], ocp.model)["cx"]
    cy = find_param_in_p_or_p_global(["cy"], ocp.model)["cy"]
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    A = _A_disc(m=m, cx=cx, cy=cy, dt=dt)
    B = _B_disc(m=m, cx=cx, cy=cy, dt=dt)

    if wind_param is None:
        u_wind = find_param_in_p_or_p_global(["u_wind"], ocp.model)["u_wind"]
    else:
        u_wind_x, u_wind_y = get_wind_velocity(
            x=ocp.model.x[0],
            y=ocp.model.x[1],
            scale=wind_param.scale,
            vortex_center=wind_param.vortex_center,
            vortex_strength=wind_param.vortex_strength,
        )

        u_wind = ca.vertcat(u_wind_x, u_wind_y)

    return A @ x + B @ (u + u_wind)


def _cost_expr_ext_cost(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    Q_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["q_diag"], ocp.model)["q_diag"]
    )
    R_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["r_diag"], ocp.model)["r_diag"]
    )

    xref = find_param_in_p_or_p_global(["xref"], ocp.model)["xref"]
    uref = find_param_in_p_or_p_global(["uref"], ocp.model)["uref"]

    return 0.5 * (
        ca.mtimes([ca.transpose(x - xref), Q_sqrt.T, Q_sqrt, x - xref])
        + ca.mtimes([ca.transpose(u - uref), R_sqrt.T, R_sqrt, u - uref])
    )


def _cost_expr_ext_cost_e(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x

    Q_sqrt_e = _create_diag_matrix(
        find_param_in_p_or_p_global(["q_diag_e"], ocp.model)["q_diag_e"]
    )

    xref_e = find_param_in_p_or_p_global(["xref_e"], ocp.model)["xref_e"]

    return 0.5 * ca.mtimes([ca.transpose(x - xref_e), Q_sqrt_e.T, Q_sqrt_e, x - xref_e])


def export_parametric_ocp(
    nominal_param: dict[str, np.ndarray],
    name: str = "pointmass",
    learnable_params: list[str] | None = None,
    N_horizon: int = 50,
    tf: float = 2.0,
    x0: np.ndarray = np.array([1.0, 1.0, 0.0, 0.0]),
    wind_param: WindParam | None = None,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = N_horizon

    ocp.model.name = name

    ocp.dims.nu = 2
    ocp.dims.nx = 4

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    # ocp.model.p = struct_symSX([entry("u_wind", shape=(2, 1))])

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_param, learnable_param=learnable_params, ocp=ocp
    )

    ocp.model.disc_dyn_expr = _disc_dyn_expr(ocp=ocp, wind_param=wind_param)
    ocp.model.cost_expr_ext_cost_0 = _cost_expr_ext_cost(ocp=ocp)
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = _cost_expr_ext_cost(ocp=ocp)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = _cost_expr_ext_cost_e(ocp=ocp)
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.constraints.x0 = x0

    Fmax = 10.0
    # Box constraints on u
    ocp.constraints.lbu = np.array([-Fmax, -Fmax])
    ocp.constraints.ubu = np.array([Fmax, Fmax])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-5.0, -5.0, -50.0, -50.0])
    ocp.constraints.ubx = np.array([5.0, 5.0, 50.0, 50.0])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3])

    ocp.constraints.idxsbx = np.array([0, 1])

    ns = 2
    ocp.cost.zl = 100 * np.ones((ns,))
    ocp.cost.Zl = 0 * np.ones((ns,))
    ocp.cost.zu = 100 * np.ones((ns,))
    ocp.cost.Zu = 0 * np.ones((ns,))

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
    ocp.solver_options.with_solution_sens_wrt_params = True
