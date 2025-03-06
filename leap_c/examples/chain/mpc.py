import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi.tools import struct_symSX
from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.mpc import Mpc, MpcInput, MpcSingleState, MpcBatchedState
from copy import deepcopy
from dataclasses import fields


from casadi import SX, norm_2, vertcat
from casadi.tools import entry
from casadi.tools.structure3 import DMStruct
import matplotlib.pyplot as plt

from acados_template.acados_ocp_batch_solver import AcadosOcpFlattenedBatchIterate

from typing import Tuple

from pathlib import Path


class ChainMpc(Mpc):
    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 20,
        T_horizon: float = 1.0,
        discount_factor: float = 1.0,
        n_batch: int = 64,
        least_squares_cost: bool = True,
        exact_hess_dyn: bool = True,
        n_mass: int = 5,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
    ):
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

        learnable_params = learnable_params if learnable_params is not None else []

        ocp = export_parametric_ocp(
            nominal_params=params,
            learnable_params=learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
            n_mass=n_mass,
            u_init=np.array([-1, 1, 1]),
            with_wall=True,
            yPosWall=-0.05,
            pos_first_mass=np.zeros(3),
            nlp_iter=50,
            nlp_tol=1e-5,
        )

        configure_ocp_solver(ocp, exact_hess_dyn)

        self.given_default_param_dict = params

        super().__init__(
            ocp=ocp,
            n_batch=n_batch,
            export_directory=export_directory,
            export_directory_sensitivity=export_directory_sensitivity,
            throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        )

        # Build the initial state
        for stage in range(self.ocp_solver.N + 1):
            self.ocp_solver.set(stage, "x", self.ocp_solver.acados_ocp.constraints.x0)
        self.ocp_solver.solve()

        iterate = self.ocp_solver.store_iterate_to_obj()

        def init_state_fn(mpc_input: MpcInput) -> MpcSingleState | MpcBatchedState:
            # TODO (batch_rules): This should be updated if we switch to only batch solvers.

            if not mpc_input.is_batched():
                return deepcopy(iterate)

            batch_size = len(mpc_input.x0)
            kw = {}

            for f in fields(iterate):
                n = f.name
                kw[n] = np.tile(getattr(iterate, n), (batch_size, 1))

            return AcadosOcpFlattenedBatchIterate(**kw, N_batch=batch_size)

        self.init_state_fn = init_state_fn


def plot_steady_state(
    x_ss: np.ndarray, u_ss: np.ndarray, n_mass, pos_first_mass
) -> Tuple[plt.Figure, plt.Figure]:
    pos_ss = x_ss[: 3 * (n_mass - 1)]

    # Concatenate xPosFirstMass and pos_ss
    pos_ss = np.concatenate((pos_first_mass, pos_ss))

    vel_ss = x_ss[3 * (n_mass - 1) :]

    # Concatenate vel_ss and u_ss
    vel_first_mass = np.zeros(3)
    vel_last_mass = u_ss
    vel_ss = np.concatenate((vel_first_mass, vel_ss, vel_last_mass))

    pos_fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(pos_ss[0::3], "o-")
    plt.subplot(3, 1, 2)
    plt.plot(pos_ss[1::3], "o-")
    plt.subplot(3, 1, 3)
    plt.plot(pos_ss[2::3], "o-")

    vel_fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(vel_ss[0::3], "o-")
    plt.subplot(3, 1, 2)
    plt.plot(vel_ss[1::3], "o-")
    plt.subplot(3, 1, 3)
    plt.plot(vel_ss[2::3], "o-")

    return pos_fig, vel_fig


def export_parametric_ocp(
    nominal_params: dict,
    name: str = "chain",
    learnable_params: list[str] | None = None,
    N_horizon: int = 30,
    tf: float = 6.0,
    n_mass=5,
    u_init=np.array([-1, 1, 1]),
    with_wall=True,
    yPosWall=-0.05,
    pos_first_mass=np.zeros(3),
    nlp_iter=50,
    nlp_tol=1e-5,
) -> Tuple[AcadosOcp, DMStruct]:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.Tsim = tf
    ocp.solver_options.tf = tf
    Ts = tf / N_horizon

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

    ocp.model.f_expl_expr = f_expl_expr(ocp.model)
    ocp.model.f_impl_expr = ocp.model.xdot - ocp.model.f_expl_expr
    ocp.model.disc_dyn_expr = disc_dyn_expr(ocp.model, Ts)
    ocp.model.name = name

    ######## Find steady state ########

    resting_link_length = np.linalg.norm(nominal_params["L"][:3])

    x_end_ref = np.array([resting_link_length * (n_mass - 1), 0.0, 0.0])

    model = ocp.model

    p = find_param_in_p_or_p_global(["D", "L", "C", "m"], model)

    nx = model.x.shape[0]

    # Free masses
    n_masses = p["m"].shape[0] + 1
    M = n_masses - 2

    # initial guess for state
    pos0_x = np.linspace(pos_first_mass[0], x_end_ref[0], n_masses)
    x0 = np.zeros((nx, 1))
    x0[: 3 * (M + 1) : 3] = pos0_x[1:].reshape((n_masses - 1, 1))

    # decision variables
    w = [model.x, model.xdot, model.u]

    # initial guess
    w0 = ca.vertcat(*[x0, np.zeros(model.xdot.shape), np.zeros(model.u.shape)])

    # constraints
    g = []
    g += [model.f_impl_expr]  # steady state
    g += [model.x["pos", -1] - x_end_ref]  # fix position of last mass
    g += [model.u]  # don't actuate controlled mass

    # misuse IPOPT as nonlinear equation solver
    nlp = {"x": ca.vertcat(*w), "f": 0, "g": ca.vertcat(*g), "p": model.p_global.cat}

    solver = ca.nlpsol("solver", "ipopt", nlp)
    sol = solver(x0=w0, lbg=0, ubg=0, p=ocp.p_global_values)

    x_ss = sol["x"].full()[:nx].flatten()
    u_ss = sol["x"].full()[-model.u.shape[0] :].flatten()

    if False:
        plot_steady_state(
            x_ss=x_ss, u_ss=u_ss, n_mass=n_masses, pos_first_mass=pos_first_mass
        )

    ########## End of steady state computation ##########

    q_sqrt_diag = find_param_in_p_or_p_global(["q_sqrt_diag"], model)["q_sqrt_diag"]
    r_sqrt_diag = find_param_in_p_or_p_global(["r_sqrt_diag"], model)["r_sqrt_diag"]

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

    # ocp.model.cost_y_expr = vertcat(x_e, u_e)

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

    # ocp.solver_options.tf = ocp.solver_options.N_horizon * Ts

    # TODO (dirk): Print nominal values for each. Useful for debugging, but should be handled differently.
    # print("ocp.p_global_values:")
    # for i, value in enumerate(ocp.p_global_values):
    #     print(f"  {ocp.model.p_global[i]} = {value}")

    # print("ocp.parameter_values:")
    # for i, value in enumerate(ocp.parameter_values):
    #     print(f"  {ocp.model.p[i]} = {value}")

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


def f_expl_expr(
    model: AcadosModel,
    x0: ca.SX = ca.SX.zeros(3),
) -> ca.SX:
    xpos = vertcat(*model.x["pos"])
    xvel = vertcat(*model.x["vel"])

    p = find_param_in_p_or_p_global(["D", "L", "C", "m", "w"], model)

    n_masses = p["m"].shape[0] + 1

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
            vel = model.u - xvel[(i - 1) * 3 : i * 3]
        else:
            vel = xvel[i * 3 : (i + 1) * 3] - xvel[(i - 1) * 3 : i * 3]

        F = ca.SX.zeros(3, 1)
        for j in range(3):
            F[j] = p["C"][i + j] * vel[j]

        # mass on the right
        if i < n_masses - 2:
            f[i * 3 : (i + 1) * 3] -= F

        # mass on the left
        if i > 0:
            f[(i - 1) * 3 : i * 3] += F

    # Disturbance on intermediate masses
    for i in range(n_masses - 2):
        f[i * 3 : (i + 1) * 3] += p["w"][i]

    return vertcat(xvel, model.u, f)


def disc_dyn_expr(model: AcadosModel, dt: float) -> ca.SX:
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
