from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from acados_template import AcadosOcp, AcadosOcpFlattenedIterate
import casadi as ca
from casadi.tools import struct_symSX
import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.mujoco.reacher.config import make_default_reacher_params
from leap_c.examples.mujoco.reacher.util import (
    InverseKinematicsSolver,
    get_mjcf_path,
    require_pinocchio,
)
from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.ocp.acados.data import AcadosOcpSolverInput
from leap_c.ocp.acados.initializer import (
    AcadosDiffMpcInitializer,
    create_zero_iterate_from_ocp,
)
from leap_c.ocp.acados.torch import AcadosDiffMpc

# Optional pinocchio import
try:
    import pinocchio as pin
    from pinocchio import casadi as cpin

    HAS_PINOCCHIO = True
except ImportError:
    pin = None
    HAS_PINOCCHIO = False


class ReacherController(ParameterizedController):
    """docstring for ReacherController."""

    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 50,
        T_horizon: float = 0.5,
        discount_factor: float = 0.99,
        urdf_path: Path | None = None,
        mjcf_path: Path | None = None,
        state_representation: str = "q",
    ):
        super().__init__()
        require_pinocchio()
        self.pinocchio_model = None

        if urdf_path is not None and mjcf_path is not None:
            raise Exception("Please provide either a URDF or MJCF file, not both.")

        if urdf_path is not None and mjcf_path is None:
            self.pinocchio_model = pin.buildModelFromUrdf(urdf_path)
        elif mjcf_path is not None and urdf_path is None:
            self.pinocchio_model = pin.buildModelFromMJCF(mjcf_path)
        else:
            path = get_mjcf_path("reacher")
            print(f"No urdf or mjcf provided. Using default model : {path}")
            self.pinocchio_model = pin.buildModelFromMJCF(path)

        self.params = (
            make_default_reacher_params(self.pinocchio_model)
            if params is None
            else params
        )
        self.learnable_params = learnable_params if learnable_params is not None else []

        print("learnable_params: ", self.learnable_params)

        self.ocp = export_parametric_ocp(
            pinocchio_model=self.pinocchio_model,
            nominal_param=asdict(self.params),
            learnable_params=self.learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
            state_representation=state_representation,
        )

        configure_ocp_solver(ocp=self.ocp, exact_hess_dyn=True)

        self.ik_solver = InverseKinematicsSolver(
            pinocchio_model=self.pinocchio_model,
            step_size=0.1,
            max_iter=1000,
            tol=1e-6,
            print_level=0,
            plot_level=0,
        )
        self.initializer = ReacherInitializer(
            ocp=self.ocp,
            ik_solver=self.ik_solver,
            pinocchio_model=self.pinocchio_model,
        )

        self.acados_layer = AcadosDiffMpc(
            self.ocp, initializer=self.initializer, discount_factor=discount_factor
        )

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        x0 = torch.as_tensor(obs, dtype=torch.float64)
        p_global = torch.as_tensor(param, dtype=torch.float64)
        ctx, u0, x, u, value = self.acados_layer(
            x0.unsqueeze(0), p_global=p_global.unsqueeze(0), ctx=ctx
        )
        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.acados_layer.sensitivity(ctx, field_name="du0_dp_global")

    def param_space(self) -> gym.Space:
        # TODO: can't determine the param space because it depends on the learnable parameters
        # we need to define boundaries for every parameter and based on that create a gym.Space
        raise NotImplementedError

    def default_param(self) -> np.ndarray:
        return np.concatenate(
            [asdict(self.params)[p].flatten() for p in self.learnable_params]
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
    pinocchio_model,
    nominal_param: dict[str, np.ndarray],
    N_horizon: int,
    tf: float,
    state_representation: str,
    name: str = "reacher",
    learnable_params: list[str] | None = None,
) -> AcadosOcp:
    require_pinocchio()

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


class ReacherInitializer(AcadosDiffMpcInitializer):
    def __init__(self, ocp: AcadosOcp, ik_solver, pinocchio_model):
        self.ocp = ocp
        self.ik_solver = ik_solver
        self.pinocchio_model = pinocchio_model
        self.zero_iterate = create_zero_iterate_from_ocp(ocp)

    def single_iterate(
        self, solver_input: AcadosOcpSolverInput
    ) -> AcadosOcpFlattenedIterate:
        # Use the same logic as in mpc.py for single initialization
        xy_ee_ref = mpc_input.p_global.flatten()[:2]  # type: ignore
        target_angle = np.arctan2(xy_ee_ref[1], xy_ee_ref[0])

        q_ref, dq_ref, _, pos, _ = self.ik_solver(
            q=np.array([target_angle] * self.pinocchio_model.nq),
            dq=np.zeros(self.pinocchio_model.nv),
            target_position=np.concatenate([xy_ee_ref, np.array([0.01])]),
        )

        x_ref = np.concatenate([q_ref, dq_ref])
        iterate = deepcopy(self.zero_iterate)
        iterate.x = x_ref

        return iterate
