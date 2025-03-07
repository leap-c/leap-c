import json
from collections import OrderedDict
from copy import copy
from typing import Dict, List

import casadi as ca
import numpy as np
import scipy
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpIterate
from casadi.tools import struct_symSX

from leap_c.examples.quadrotor.casadi_models import get_rhs_quadrotor
from leap_c.examples.quadrotor.utils import read_from_yaml
from leap_c.examples.util import translate_learnable_param_to_p_global
from leap_c.mpc import Mpc, MpcInput
from leap_c.utils import set_standard_sensitivity_options
from os.path import dirname, abspath

PARAMS = OrderedDict(
    [
        ("m", np.array([0.6])),
        ("g", np.array([9.81])),
    ]
)


class QuadrotorMpc(Mpc):

    def __init__(
            self,
            params: dict[str, np.ndarray] | None = None,
            discount_factor: float = 0.99,
            n_batch: int = 64,
            N_horizon: int = 3,
            params_learnable: list[str] | None = None,
    ):
        """
        Args:
            params: A dict with the parameters of the ocp, together with their default values.
                For a description of the parameters, see the docstring of the class.
            learnable_params: A list of the parameters that should be learnable
                (necessary for calculating their gradients).
            N_horizon: The number of steps in the MPC horizon.
                The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal node N).
            T_horizon: The length (meaning time) of the MPC horizon.
                One step in the horizon will equal T_horizon/N_horizon simulation time.
            discount_factor: The discount factor for the cost.
            n_batch: The batch size the MPC should be able to process
                (currently this is static).
            least_squares_cost: If True, the cost will be the LLS cost, if False it will
                be the general quadratic cost(see above).
            exact_hess_dyn: If False, the contributions of the dynamics will be left out of the Hessian.
        """
        params = params if params is not None else PARAMS

        ocp = export_parametric_ocp(
            name="quadrotor_lls",
            N_horizon=N_horizon,
            sensitivity_ocp=False,
            params_learnable=params_learnable,
        )

        ocp_sens = export_parametric_ocp(
            name="quadrotor_lls_exact",
            N_horizon=N_horizon,
            sensitivity_ocp=True,
            params_learnable=params_learnable,
        )

        self.given_default_param_dict = params

        with open(dirname(abspath(__file__)) + "/init_iterate.json", "r") as file:
            init_iterate = json.load(file)  # Parse JSON into a Python dictionary
            init_iterate = parse_ocp_iterate(init_iterate, N=N_horizon)

        def initialize_default(mpc_input: MpcInput):
            init_iterate.x_traj = [mpc_input.x0] * (ocp.solver_options.N_horizon + 1)
            return init_iterate

        init_state_fn = initialize_default
        # Convert dictionary to a namedtuple

        super().__init__(
            ocp=ocp,
            ocp_sensitivity=ocp_sens,
            discount_factor=discount_factor,
            n_batch=n_batch,
            init_state_fn=init_state_fn,
        )


def export_parametric_ocp(
        name: str = "quadrotor",
        N_horizon: int = 5,
        sensitivity_ocp=False,
        params_learnable: list[str] | None = None,
) -> AcadosOcp:
    ocp = AcadosOcp()
    param_external_cost = (params_learnable is not None) and ("terminal_cost" in params_learnable)

    ######## Dimensions ########
    dt = 0.04  # 0.005

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon * dt

    ######## Model ########
    # Quadrotor parameters
    model_params = read_from_yaml(dirname(abspath(__file__)) + "/model_params.yaml")

    x, u, p, rhs, rhs_func = get_rhs_quadrotor(model_params, model_fidelity="low")
    ocp.model.disc_dyn_expr = disc_dyn_expr(rhs, x, u, p, dt)

    ocp.model.name = name
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.p_global = p[0]
    ocp.p_global_values = np.array([model_params["mass"]])

    xdot = ca.SX.sym('xdot', x.shape)
    ocp.model.xdot = xdot
    ocp.model.f_impl_expr = xdot - rhs

    ocp.dims.nx = x.size()[0]
    ocp.dims.nu = u.size()[0]
    nx, nu, ny, ny_e = ocp.dims.nx, ocp.dims.nu, ocp.dims.nx + ocp.dims.nu, ocp.dims.nx

    ######## Cost ########
    # stage cost
    Q = np.diag([1e4, 1e4, 1e4,
                 1e0, 1e4, 1e4, 1e0,
                 1e1, 1e1, 1e3,
                 1e1, 1e1, 1e1])

    R = np.diag([1, 1, 1, 1]) / 16

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx: nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    # terminal cost
    if param_external_cost:
        q_e_diag = ca.SX.sym("q_e_diag", nx)
        Q_sqrt_e = ca.diag(q_e_diag)
        xref_e = ca.SX.sym("xref_e", nx)
        ocp.model.p_global = ca.vertcat(ocp.model.p_global, q_e_diag, xref_e)
        ocp.p_global_values = np.concatenate([ocp.p_global_values, 100 * np.diag(Q), np.zeros(nx)])

        ocp.model.cost_expr_ext_cost_e = 0.5 * ca.mtimes([ca.transpose(x - xref_e), Q_sqrt_e.T, Q_sqrt_e, x - xref_e])
        ocp.cost.cost_type_e = "EXTERNAL"
    else:
        Qe = 100 * Q
        ocp.cost.W_e = Qe
        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx_e = Vx_e

        ocp.cost.yref_e = np.zeros((ny_e,))
        ocp.cost.yref_e[3] = 1

    # constraints
    ocp.constraints.idxbx = np.array([2])
    ocp.constraints.lbx = np.array([-model_params["bound_z"] * 10])
    ocp.constraints.ubx = np.array([model_params["bound_z"]])

    ocp.constraints.idxbx_e = np.array([2])
    ocp.constraints.lbx_e = np.array([-model_params["bound_z"] * 10])
    ocp.constraints.ubx_e = np.array([model_params["bound_z"]])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zu = ocp.cost.zl = np.array([0])
    ocp.cost.Zu = ocp.cost.Zl = np.array([1e10])

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref[3] = 1
    ocp.cost.yref[nx:nx + nu] = 970.437

    ######## Constraints ########
    ocp.constraints.x0 = np.array([0] * 13)
    ocp.constraints.lbu = np.array([0] * 4)
    ocp.constraints.ubu = np.array([model_params["motor_omega_max"]] * 4)
    ocp.constraints.idxbu = np.array(range(4))

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 2

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    if sensitivity_ocp:
        set_standard_sensitivity_options(ocp)

    return ocp


def parse_ocp_iterate(data: Dict[str, List[float]], N=None) -> AcadosOcpIterate:
    """
    Parses the given JSON-like dictionary into an instance of AcadosOcpIterate.

    Args:
        data (dict): The input dictionary containing state, control, and dual variables.

    Returns:
        AcadosOcpIterate: The parsed iterate structure.
    """
    # Extract state trajectory
    x_traj = [np.array(data[key]) for key in sorted(data.keys()) if key.startswith("x_")]
    # Extact control trajectory
    u_traj = [np.array(data[key]) for key in sorted(data.keys()) if key.startswith("u_")]
    # Extract dual variables
    pi_traj = [np.array(data[key]) for key in sorted(data.keys()) if key.startswith("pi_")]
    lam_traj = [np.array(data[key]) for key in sorted(data.keys()) if key.startswith("lam_")]
    sl_traj = [np.array(data[key]) for key in sorted(data.keys()) if key.startswith("sl_")]
    su_traj = [np.array(data[key]) for key in sorted(data.keys()) if key.startswith("su_")]

    if N is not None and len(x_traj) < N + 1:
        for i in range((N + 1) + len(x_traj)):
            x_traj.append(copy(x_traj[-1]))
            pi_traj.append(copy(pi_traj[-1]))
            lam_traj.append(copy(lam_traj[-1]))
            u_traj.append(copy(u_traj[-1]))

    # Assuming `z_traj` is empty since there's no "z_" in the given data
    z_traj = []

    return AcadosOcpIterate(
        x_traj=x_traj,
        u_traj=u_traj,
        z_traj=z_traj,
        sl_traj=sl_traj,
        su_traj=su_traj,
        pi_traj=pi_traj,
        lam_traj=lam_traj,
    )


def disc_dyn_expr(rhs, x, u, p, dt: float) -> ca.SX:
    ode = ca.Function("ode", [x, u, *p], [rhs])
    k1 = ode(x, u, *p)
    k2 = ode(x + dt / 2 * k1, u, *p)  # type:ignore
    k3 = ode(x + dt / 2 * k2, u, *p)  # type:ignore
    k4 = ode(x + dt * k3, u, *p)  # type:ignore

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
