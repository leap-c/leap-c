from dataclasses import fields
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from acados_template.acados_ocp_batch_solver import AcadosOcpFlattenedBatchIterate
from casadi import SX, norm_2, vertcat
from casadi.tools import entry, struct_symSX
from casadi.tools.structure3 import DMStruct, ssymStruct
from leap_c.examples.chain.utils import (
    RestingChainSolver,
    nominal_params_to_structured_nominal_params,
    plot_steady_state,
)
from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.ocp.acados.mpc import Mpc, MpcBatchedState, MpcInput


class ChainMpc(Mpc):
    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 20,
        T_horizon: float = 1.0,
        discount_factor: float = 1.0,
        n_batch: int = 64,
        exact_hess_dyn: bool = True,
        n_mass: int = 5,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
        fix_point: np.ndarray | None = None,
        pos_last_mass_ref: np.ndarray | None = None,
    ):
        if params is None:
            params = {}

            # rest length of spring
            params["L"] = np.repeat([0.033, 0.033, 0.033], n_mass - 1)

            # spring constant
            params["D"] = np.repeat([1.0, 1.0, 1.0], n_mass - 1)

            # damping constant
            params["C"] = np.repeat([0.1, 0.1, 0.1], n_mass - 1)

            # mass of the balls
            params["m"] = np.repeat([0.033], n_mass - 1)

            # disturbance on intermediate balls
            params["w"] = np.repeat([0.0, 0.0, 0.0], n_mass - 2)

            # Weight on state
            params["q_sqrt_diag"] = np.ones(3 * (n_mass - 1) + 3 * (n_mass - 2))

            # Weight on control inputs
            params["r_sqrt_diag"] = 1e-1 * np.ones(3)

        if fix_point is None:
            fix_point = np.zeros(3)

        if pos_last_mass_ref is None:
            pos_last_mass_ref = fix_point + np.array([0.033 * (n_mass - 1), 0, 0])

        learnable_params = learnable_params if learnable_params is not None else []

        ocp = export_parametric_ocp(
            nominal_params=params,
            learnable_params=learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
            n_mass=n_mass,
            fix_point=fix_point,
            pos_last_mass_ref=pos_last_mass_ref,
        )

        set_ocp_solver_options(ocp, exact_hess_dyn)

        self.given_default_param_dict = params

        super().__init__(
            ocp=ocp,
            n_batch_max=n_batch,
            export_directory=export_directory,
            export_directory_sensitivity=export_directory_sensitivity,
            throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        )

        # Build the initial state
        for stage in range(self.ocp_solver.N + 1):
            self.ocp_solver.set(stage, "x", self.ocp_solver.acados_ocp.constraints.x0)
        self.ocp_solver.solve()

        iterate = self.ocp_solver.store_iterate_to_flat_obj()

        def init_state_fn(mpc_input: MpcInput) -> MpcBatchedState:
            if not mpc_input.is_batched():
                batch_size = 1
            else:
                batch_size = len(mpc_input.x0)
            kw = {}

            for f in fields(iterate):
                n = f.name
                kw[n] = np.tile(getattr(iterate, n), (batch_size, 1))

            return AcadosOcpFlattenedBatchIterate(**kw, N_batch=batch_size)

        self.init_state_fn = init_state_fn


def export_parametric_ocp(
    nominal_params: dict,
    name: str = "chain",
    learnable_params: list[str] | None = None,
    N_horizon: int = 30,  # noqa: N803
    tf: float = 6.0,
    n_mass: int = 5,
    fix_point: np.ndarray = np.array([0.0, 0.0, 0.0]),
    pos_last_mass_ref: np.ndarray = np.array([1.0, 0.0, 0.0]),
) -> tuple[AcadosOcp, DMStruct]:
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

    ocp.model.xdot = ca.SX.sym("xdot", ocp.model.x.cat.shape[0], 1)

    ocp.model.u = ca.SX.sym("u", 3, 1)

    p = find_param_in_p_or_p_global(["D", "L", "C", "m", "w"], ocp.model)

    ocp.model.f_expl_expr = get_f_expl_expr(
        x=ocp.model.x, u=ocp.model.u, p=p, x0=fix_point
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

    # pos_last_mass = pos_first_mass + np.array([1.0, 0.0, 0.0])

    resting_chain_solver.set("p_last", pos_last_mass_ref)

    x_ss, u_ss = resting_chain_solver(p_last=pos_last_mass_ref)

    if False:
        plot_steady_state(x_ss=x_ss, u_ss=u_ss, n_mass=n_mass, pos_first_mass=fix_point)

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

    return ocp


def set_ocp_solver_options(ocp: AcadosOcp, exact_hess_dyn: bool):
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.qp_tol = 1e-7
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_batch_functionality = True


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
    f = SX.zeros(3 * (n_masses - 2), 1)

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

        F = ca.SX.zeros(3, 1)
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

        F = ca.SX.zeros(3, 1)
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

    return vertcat(xvel, u, f)


def get_disc_dyn_expr(model: AcadosModel, dt: float) -> ca.SX:
    f_expl_expr = model.f_expl_expr

    x = model.x
    u = model.u
    p = ca.vertcat(
        *find_param_in_p_or_p_global(["L", "C", "D", "m", "w"], model).values()
    )

    ode = ca.Function("ode", [x, u, p], [f_expl_expr])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)
    k3 = ode(x + dt / 2 * k2, u, p)
    k4 = ode(x + dt * k3, u, p)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
