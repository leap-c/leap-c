from dataclasses import fields
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
from acados_template import AcadosOcp
from acados_template.acados_ocp_batch_solver import AcadosOcpFlattenedBatchIterate
from casadi.tools import struct_symSX
from pinocchio import casadi as cpin

from leap_c.examples.mujoco.reacher.util import InverseKinematicsSolver, get_mjcf_path
from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.ocp.acados.mpc import Mpc, MpcBatchedState, MpcInput


class ReacherMpc(Mpc):
    """docstring for NLinkRobotMpc."""

    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 50,
        T_horizon: float = 0.5,
        discount_factor: float = 0.99,
        n_batch: int = 64,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
        urdf_path: Path | None = None,
        mjcf_path: Path | None = None,
        state_representation: str = "q",
    ):
        pinocchio_model = None

        if urdf_path is not None and mjcf_path is not None:
            raise Exception("Please provide either a URDF or MJCF file, not both.")

        if urdf_path is not None and mjcf_path is None:
            pinocchio_model = pin.buildModelFromUrdf(urdf_path)
        elif mjcf_path is not None and urdf_path is None:
            pinocchio_model = pin.buildModelFromMJCF(mjcf_path)
        else:
            path = get_mjcf_path("reacher")
            print(f"No urdf or mjcf provided. Using default model : {path}")
            pinocchio_model = pin.buildModelFromMJCF(path)

        params = (
            {
                "xy_ee_ref": np.array([0.21, 0.0]),
                "q_sqrt_diag": np.array([10.0, 10.0]),
                "r_sqrt_diag": np.array([0.05] * pinocchio_model.nq),
            }
            if params is None
            else params
        )

        learnable_params = learnable_params if learnable_params is not None else []

        print("learnable_params: ", learnable_params)

        ocp = export_parametric_ocp(
            pinocchio_model=pinocchio_model,
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
            discount_factor=discount_factor,
            export_directory=export_directory,
            export_directory_sensitivity=export_directory_sensitivity,
            throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        )

        # Use the inverse kinematics solver to initialize the state
        self.ik_solver = InverseKinematicsSolver(
            pinocchio_model=pinocchio_model,
            step_size=0.1,
            max_iter=1000,
            tol=1e-6,
            print_level=0,
            plot_level=0,
        )

        def init_state_fn(mpc_input: MpcInput) -> MpcBatchedState:
            iterate = self.ocp_batch_solver.store_iterate_to_flat_obj()

            batch_size = len(mpc_input.x0) if mpc_input.is_batched() else 1

            print(f"Running init_state_fn; batch_size = {batch_size}")

            # TODO: Make this work for batched input with batch_size > 1
            if batch_size == 1:
                xy_ee_ref = mpc_input.parameters.p_global.flatten()[:2]
                target_angle = np.arctan2(xy_ee_ref[1], xy_ee_ref[0])

                q_ref, dq_ref, _, pos, _ = self.ik_solver(
                    q=np.array([target_angle] * pinocchio_model.nq),
                    dq=np.zeros(pinocchio_model.nv),
                    target_position=np.concatenate([xy_ee_ref, np.array([0.01])]),
                )

                x_ref = np.concatenate([q_ref, dq_ref])

                for stage in range(
                    self.ocp_solver.acados_ocp.solver_options.N_horizon + 1
                ):
                    self.ocp_solver.set(stage, "x", x_ref)

                self.ocp_solver.set_p_global_and_precompute_dependencies(
                    mpc_input.parameters.p_global.flatten()
                )

                # Set the initial velocity to zero
                x0 = mpc_input.x0.flatten()
                x0[:2] = np.zeros(2)

                _ = self.ocp_solver.solve_for_x0(x0, fail_on_nonzero_status=False)

                if self.ocp_solver.status != 0:
                    print("Failed to solve for x0")
                    print("status", self.ocp_solver.status)
                    print("q", x0[:2])
                    print("q_ref", q_ref)
                    print("dq", x0[2:])

                iterate = self.ocp_solver.store_iterate_to_flat_obj()

            ####

            kw = {}
            for f in fields(iterate):
                n = f.name
                kw[n] = np.tile(getattr(iterate, n), (batch_size, 1))

            return AcadosOcpFlattenedBatchIterate(**kw, N_batch=batch_size)

        self.init_state_fn = init_state_fn


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
    N_horizon: int,
    tf: float,
    state_representation: str,
    name: str = "reacher",
    learnable_params: list[str] | None = None,
) -> AcadosOcp:
    ocp = AcadosOcp()

    model = cpin.Model(pinocchio_model)
    data = model.createData()

    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = N_horizon
    ocp.model.name = name

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

    # States
    dq = ca.SX.sym("dq", model.nq, 1)

    if state_representation == "sin_cos":
        cosq = ca.SX.sym("cos(q)", model.nq, 1)
        sinq = ca.SX.sym("sin(q)", model.nq, 1)
        q = ca.atan2(sinq, cosq)
        ocp.model.x = ca.vertcat(cosq, sinq, dq)
        q_fun = ca.Function("q_fun", [cosq, sinq], [q])
        print("q_fun(1, 0): ", q_fun(1, 0))
        ocp.model.f_expl_expr = ca.vertcat(
            -sinq * dq, cosq * dq, cpin.aba(model, data, q, dq, 200 * tau)
        )
    else:
        q = ca.SX.sym("q", model.nq, 1)
        ocp.model.x = ca.vertcat(q, dq)
        ocp.model.f_expl_expr = ca.vertcat(dq, cpin.aba(model, data, q, dq, 200 * tau))

    ocp.dims.nx = ocp.model.x.shape[0]

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
    # ns = fun.constraints.idxsbx.size
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
    ocp.solver_options.integrator_type = "DISCRETE"
    # ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.nlp_solver_tol_stat = 1e-4
    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_batch_functionality = True
