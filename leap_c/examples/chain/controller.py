from copy import deepcopy
from dataclasses import asdict
from typing import Any

from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpFlattenedIterate,
)
import casadi as ca
from casadi import SX, norm_2, vertcat
from casadi.tools import entry, struct_symSX
from casadi.tools.structure3 import ssymStruct
import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.chain.config import ChainParams, make_default_chain_params
from leap_c.examples.chain.utils import (
    RestingChainSolver,
    nominal_params_to_structured_nominal_params,
)
from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.ocp.acados.data import AcadosOcpSolverInput
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer, create_zero_iterate_from_ocp
from leap_c.ocp.acados.torch import AcadosDiffMpc


class ChainController(ParameterizedController):
    def __init__(
        self,
        params: ChainParams | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 20,
        T_horizon: float = 1.0,
        discount_factor: float = 1.0,
        n_mass: int = 5,
        fix_point: np.ndarray | None = None,
        pos_last_mass_ref: np.ndarray | None = None,
    ):
        super().__init__()
        self.params = make_default_chain_params(n_mass) if params is None else params
        self.learnable_params = learnable_params if learnable_params is not None else []

        if fix_point is None:
            fix_point = np.zeros(3)

        if pos_last_mass_ref is None:
            pos_last_mass_ref = fix_point + np.array([0.033 * (n_mass - 1), 0, 0])

        self.ocp = export_parametric_ocp(
            nominal_params=asdict(self.params),
            learnable_params=self.learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
            n_mass=n_mass,
            fix_point=fix_point,
            pos_last_mass_ref=pos_last_mass_ref,
        )

        initializer = ChainInitializer(
            self.ocp,
            nominal_params=asdict(self.params),
            n_mass=n_mass,
            fix_point=fix_point,
            pos_last_mass_ref=pos_last_mass_ref,
        )
        self.diff_mpc = AcadosDiffMpc(
            self.ocp,
            initializer=initializer,
            discount_factor=discount_factor,
        )

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        x0 = torch.as_tensor(obs)
        p_global = torch.as_tensor(param)
        ctx, u0, *_ = self.diff_mpc(
            x0.unsqueeze(0), p_global=p_global.unsqueeze(0), ctx=ctx
        )
        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.diff_mpc.sensitivity(ctx, field_name="du0_dp_global")

    def param_space(self) -> gym.Space:
        # TODO: can't determine the param space because it depends on the learnable parameters
        # we need to define boundaries for every parameter and based on that create a gym.Space
        raise NotImplementedError

    def default_param(self) -> np.ndarray:
        return np.concatenate(
            [asdict(self.params)[p].flatten() for p in self.learnable_params]
        )


def export_parametric_ocp(
    nominal_params: dict,
    learnable_params: list[str],
    name: str = "chain",
    N_horizon: int = 30,  # noqa: N803
    tf: float = 6.0,
    n_mass: int = 5,
    fix_point: np.ndarray = np.array([0.0, 0.0, 0.0]),
    pos_last_mass_ref: np.ndarray = np.array([1.0, 0.0, 0.0]),
) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.Tsim = tf
    ocp.solver_options.tf = tf

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_params,
        learnable_param=learnable_params,
        ocp=ocp,
    )

    ocp.model.x = struct_symSX(
        [
            entry("pos", shape=(3, 1), repeat=n_mass - 1),
            entry("vel", shape=(3, 1), repeat=n_mass - 2),
        ]
    )

    ocp.model.xdot = ca.SX.sym("xdot", ocp.model.x.cat.shape[0], 1)  # type:ignore

    ocp.model.u = ca.SX.sym("u", 3, 1)  # type: ignore

    p = find_param_in_p_or_p_global(["D", "L", "C", "m", "w"], ocp.model)

    ocp.model.f_expl_expr = get_f_expl_expr(
        x=ocp.model.x, u=ocp.model.u, p=p, x0=fix_point  # type:ignore
    )
    ocp.model.f_impl_expr = ocp.model.xdot - ocp.model.f_expl_expr
    ocp.model.disc_dyn_expr = get_disc_dyn_expr(ocp.model, tf / N_horizon)
    ocp.model.name = name

    resting_chain_solver = RestingChainSolver(
        n_mass=n_mass, fix_point=fix_point, f_expl=get_f_expl_expr
    )

    structured_nominal_params = nominal_params_to_structured_nominal_params(
        nominal_params=nominal_params
    )
    for i in range(n_mass - 1):
        resting_chain_solver.set_mass_param(i, "D", structured_nominal_params["D"][i])
        resting_chain_solver.set_mass_param(i, "L", structured_nominal_params["L"][i])
        resting_chain_solver.set_mass_param(i, "C", structured_nominal_params["C"][i])
        resting_chain_solver.set_mass_param(i, "m", structured_nominal_params["m"][i])

    resting_chain_solver.set("fix_point", fix_point)
    resting_chain_solver.set("p_last", pos_last_mass_ref)

    x_ss, u_ss = resting_chain_solver(p_last=pos_last_mass_ref)

    q_sqrt_diag = find_param_in_p_or_p_global(["q_sqrt_diag"], ocp.model)["q_sqrt_diag"]
    r_sqrt_diag = find_param_in_p_or_p_global(["r_sqrt_diag"], ocp.model)["r_sqrt_diag"]

    Q = ca.diag(q_sqrt_diag) @ ca.diag(q_sqrt_diag).T
    R = ca.diag(r_sqrt_diag) @ ca.diag(r_sqrt_diag).T

    nx = ocp.model.x.cat.shape[0]
    nu = ocp.model.u.shape[0]

    x_e = ocp.model.x.cat - x_ss
    u_e = ocp.model.u - np.zeros((nu, 1))

    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = 0.5 * (x_e.T @ Q @ x_e + u_e.T @ R @ u_e)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = 0.5 * (x_e.T @ Q @ x_e + u_e.T @ R @ u_e)
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = 0.5 * (x_e.T @ Q @ x_e)

    # set constraints
    umax = 1 * np.ones((nu,))

    ocp.constraints.lbu = -umax
    ocp.constraints.ubu = umax
    ocp.constraints.idxbu = np.array(range(nu))
    ocp.constraints.x0 = x_ss.reshape((nx,))

    # #############################
    if isinstance(ocp.model.x, struct_symSX):
        ocp.model.x = ocp.model.x.cat

    if isinstance(ocp.model.u, struct_symSX):
        ocp.model.u = ocp.model.u.cat

    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.exact_hess_dyn = True
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.qp_tol = 1e-7
    # TODO (Jasper): This should be/ is set automatically!?
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_batch_functionality = True

    return ocp


def get_f_expl_expr(
    x: ssymStruct,
    u: ca.SX,
    p: dict[str, ca.SX],
    x0: ca.SX = ca.SX.zeros(3),
) -> ca.SX:
    n_masses = p["m"].shape[0] + 1

    xpos = vertcat(*x["pos"])
    xvel = vertcat(*x["vel"])

    # Force on intermediate masses
    f = SX.zeros(3 * (n_masses - 2), 1)  # type: ignore

    # Gravity force on intermediate masses
    for i in range(int(f.shape[0] / 3)):
        f[3 * i + 2] = -9.81

    n_link = n_masses - 1

    # Spring force
    for i in range(n_link):
        if i == 0:
            dist = xpos[i * 3 : (i + 1) * 3] - x0
        else:
            dist = xpos[i * 3 : (i + 1) * 3] - xpos[(i - 1) * 3 : i * 3]

        F = ca.SX.zeros(3, 1)  # type: ignore
        for j in range(F.shape[0]):
            F[j] = (
                p["D"][i + j] / p["m"][i] * (1 - p["L"][i + j] / norm_2(dist)) * dist[j]
            )

        # mass on the right
        if i < n_link - 1:
            f[i * 3 : (i + 1) * 3] -= F

        # mass on the left
        if i > 0:
            f[(i - 1) * 3 : i * 3] += F

    # Damping force
    for i in range(n_link):
        if i == 0:
            vel = xvel[i * 3 : (i + 1) * 3]
        elif i == n_link - 1:
            vel = u - xvel[(i - 1) * 3 : i * 3]
        else:
            vel = xvel[i * 3 : (i + 1) * 3] - xvel[(i - 1) * 3 : i * 3]

        F = ca.SX.zeros(3, 1)  # type: ignore
        for j in range(3):
            F[j] = p["C"][i + j] * ca.norm_1(vel[j]) * vel[j]

        # mass on the right
        if i < n_masses - 2:
            f[i * 3 : (i + 1) * 3] -= F

        # mass on the left
        if i > 0:
            f[(i - 1) * 3 : i * 3] += F

    # Disturbance on intermediate masses
    for i in range(n_masses - 2):
        f[i * 3 : (i + 1) * 3] += p["w"][i]

    return vertcat(xvel, u, f)  # type: ignore


def get_disc_dyn_expr(model: AcadosModel, dt: float) -> ca.SX:
    f_expl_expr = model.f_expl_expr

    x = model.x
    u = model.u
    p = ca.vertcat(
        *find_param_in_p_or_p_global(["L", "C", "D", "m", "w"], model).values()
    )

    ode = ca.Function("ode", [x, u, p], [f_expl_expr])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)  # type:ignore
    k3 = ode(x + dt / 2 * k2, u, p)  # type: ignore
    k4 = ode(x + dt * k3, u, p)  # type: ignore

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # type: ignore


class ChainInitializer(AcadosDiffMpcInitializer):
    def __init__(
        self,
        ocp: AcadosOcp,
        nominal_params: dict,
        n_mass: int = 5,
        fix_point: np.ndarray = np.zeros(3),
        pos_last_mass_ref: np.ndarray = np.array([1.0, 0.0, 0.0]),
    ):

        resting_chain_solver = RestingChainSolver(
            n_mass=n_mass, fix_point=fix_point, f_expl=get_f_expl_expr
        )

        structured_nominal_params = nominal_params_to_structured_nominal_params(
            nominal_params=nominal_params
        )
        for i in range(n_mass - 1):
            resting_chain_solver.set_mass_param(
                i, "D", structured_nominal_params["D"][i]
            )
            resting_chain_solver.set_mass_param(
                i, "L", structured_nominal_params["L"][i]
            )
            resting_chain_solver.set_mass_param(
                i, "C", structured_nominal_params["C"][i]
            )
            resting_chain_solver.set_mass_param(
                i, "m", structured_nominal_params["m"][i]
            )

        resting_chain_solver.set("fix_point", fix_point)
        resting_chain_solver.set("p_last", pos_last_mass_ref)

        x_ss, u_ss = resting_chain_solver(p_last=pos_last_mass_ref)
        x_ref = np.tile(x_ss, (1, ocp.solver_options.N_horizon + 1))[0]  # type:ignore

        self.default_iterate = create_zero_iterate_from_ocp(ocp)
        self.default_iterate.x = x_ref  # type:ignore

    def single_iterate(
        self, solver_input: AcadosOcpSolverInput
    ) -> AcadosOcpFlattenedIterate:
        return deepcopy(self.default_iterate)
