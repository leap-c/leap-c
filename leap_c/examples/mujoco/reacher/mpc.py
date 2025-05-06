from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi.tools import struct_symSX
from pinocchio import casadi as cpin

from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.mpc import Mpc


class ReacherMpc(Mpc):
    """docstring for NLinkRobotMpc."""

    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 500,
        T_horizon: float = 5.0,
        discount_factor: float = 0.99,
        n_batch: int = 64,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
        urdf_path: Path | None = None,
        mjcf_path: Path | None = None,
        state_representation: str = "sin_cos",
    ):
        model = None

        if urdf_path is not None and mjcf_path is not None:
            raise Exception("Please provide either a URDF or MJCF file, not both.")

        if urdf_path is not None and mjcf_path is None:
            model = pin.buildModelFromUrdf(urdf_path)
        elif mjcf_path is not None and urdf_path is None:
            model = pin.buildModelFromMJCF(mjcf_path)
        else:
            path = Path(__file__).parent / "reacher.xml"
            print(f"No urdf or mjcf provided. Using default model : {path}")
            model = pin.buildModelFromMJCF(path)

        params = (
            {
                "xy_ee_ref": np.array([0.1, 0.0]),
                "q_sqrt_diag": np.array([10.0, 10.0]),
                "r_sqrt_diag": np.array([0.05] * model.nq),
            }
            if params is None
            else params
        )

        learnable_params = learnable_params if learnable_params is not None else []

        print("learnable_params: ", learnable_params)

        ocp = export_parametric_ocp(
            pinocchio_model=model,
            nominal_param=params,
            learnable_params=learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
            state_representation=state_representation,
        )

        configure_ocp_solver(ocp=ocp, exact_hess_dyn=True)

        self.given_default_param_dict = params
        super().__init__(
            ocp=ocp,
            n_batch_max=n_batch,
            export_directory=export_directory,
            export_directory_sensitivity=export_directory_sensitivity,
            throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        )


def get_disc_dyn_expr(ocp: AcadosOcp, dt: float) -> ca.SX:
    # discrete dynamics via RK4
    ode = ca.Function("ode", [ocp.model.x, ocp.model.u], [ocp.model.f_expl_expr])
    k1 = ode(ocp.model.x, ocp.model.u)
    k2 = ode(ocp.model.x + dt / 2 * k1, ocp.model.u)
    k3 = ode(ocp.model.x + dt / 2 * k2, ocp.model.u)
    k4 = ode(ocp.model.x + dt * k3, ocp.model.u)

    return ocp.model.x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def create_diag_matrix(
    v_diag: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    return ca.diag(v_diag) if isinstance(v_diag, ca.SX) else np.diag(v_diag)


def export_parametric_ocp(
    pinocchio_model: pin.Model,
    nominal_param: dict[str, np.ndarray],
    name: str = "n_link_robot",
    learnable_params: list[str] | None = None,
    N_horizon: int = 100,
    tf: float = 1.0,
    state_representation: str = "sin_cos",
) -> AcadosOcp:
    ocp = AcadosOcp()

    model = cpin.Model(pinocchio_model)
    data = model.createData()

    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = N_horizon
    ocp.model.name = name

    # States
    dq = ca.SX.sym("dq", model.nq, 1)

    if state_representation == "sin_cos":
        cosq = ca.SX.sym("cos(q)", model.nq, 1)
        sinq = ca.SX.sym("sin(q)", model.nq, 1)
        q = ca.atan2(sinq, cosq)
        ocp.model.x = ca.vertcat(cosq, sinq, dq)
        q_fun = ca.Function("q_fun", [cosq, sinq], [q])
        print("q_fun(1, 0): ", q_fun(1, 0))
    else:
        q = ca.SX.sym("q", model.nq, 1)
        ocp.model.x = ca.vertcat(q, dq)

    ocp.dims.nx = ocp.model.x.shape[0]

    # Controls
    tau = ca.SX.sym("tau", model.nq, 1)
    ocp.model.u = tau
    ocp.dims.nu = ocp.model.u.shape[0]

    # Parameters
    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_param,
        learnable_param=learnable_params,
        ocp=ocp,
    )

    # Dynamics
    # ocp.model.f_expl_expr = ca.vertcat(dq, cpin.aba(model, data, q, dq, 200 * tau))
    ocp.model.f_expl_expr = ca.vertcat(
        -sinq * dq, cosq * dq, cpin.aba(model, data, q, dq, 200 * tau)
    )
    ocp.model.disc_dyn_expr = get_disc_dyn_expr(
        ocp=ocp,
        dt=ocp.solver_options.tf / ocp.solver_options.N_horizon,
    )

    # Cost

    # Get the position of the fingertip
    cpin.forwardKinematics(model, data, q, dq)
    cpin.updateFramePlacements(model, data)
    xy_ee = data.oMf[model.getFrameId("fingertip")].translation[:2]

    xy_ee_ref = find_param_in_p_or_p_global(["xy_ee_ref"], ocp.model)["xy_ee_ref"]
    q_diag = find_param_in_p_or_p_global(["q_sqrt_diag"], ocp.model)["q_sqrt_diag"] ** 2
    r_diag = find_param_in_p_or_p_global(["r_sqrt_diag"], ocp.model)["r_sqrt_diag"] ** 2

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = ca.diag(ca.vertcat(q_diag, r_diag))
    ocp.model.cost_y_expr = ca.vertcat(xy_ee, ocp.model.u)
    ocp.cost.yref = ca.vertcat(xy_ee_ref, ca.SX.zeros(ocp.dims.nu))

    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = ocp.cost.W[:2, :2]
    ocp.model.cost_y_expr_e = xy_ee
    ocp.cost.yref_e = ocp.cost.yref[:2]

    # Constraints
    if state_representation == "sin_cos":
        ocp.constraints.x0 = np.concatenate(
            [
                np.ones(model.nq),
                np.zeros(model.nq),
                np.zeros(model.nv),
            ]
        )
    else:
        ocp.constraints.x0 = np.concatenate(
            [
                np.zeros(model.nq),
                np.zeros(model.nv),
            ]
        )

    # ocp.constraints.lbx = np.array([pinocchio_model.lowerPositionLimit[-1]])
    # ocp.constraints.ubx = np.array([pinocchio_model.upperPositionLimit[-1]])
    # ocp.constraints.idxbx = np.array([1])

    # # Add slack variables for lbx, ubx
    # ocp.constraints.idxsbx = np.array([0])
    # ns = ocp.constraints.idxsbx.size
    # ocp.cost.zl = 10000 * np.ones((ns,))
    # ocp.cost.Zl = 10 * np.ones((ns,))
    # ocp.cost.zu = 10000 * np.ones((ns,))
    # ocp.cost.Zu = 10 * np.ones((ns,))

    ocp.constraints.lbu = np.array([-1.0] * pinocchio_model.nv)
    ocp.constraints.ubu = np.array([+1.0] * pinocchio_model.nv)
    ocp.constraints.idxbu = np.arange(
        pinocchio_model.nv,
        dtype=int,
    )

    # Cast parameters to the correct type required by acados
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp


def configure_ocp_solver(ocp: AcadosOcp, exact_hess_dyn: bool):
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    # ocp.solver_options.with_value_sens_wrt_params = True
    # ocp.solver_options.with_solution_sens_wrt_params = True
    # ocp.solver_options.with_batch_functionality = True
