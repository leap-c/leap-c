from collections import OrderedDict

import casadi as ca
import numpy as np
import scipy
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi.tools import struct_symSX

from leap_c.examples.quadrotor.casadi_models import get_rhs_quadrotor
from leap_c.examples.quadrotor.utils import read_from_yaml
from leap_c.examples.util import translate_learnable_param_to_p_global
from leap_c.mpc import MPC
from leap_c.utils import set_standard_sensitivity_options

PARAMS = OrderedDict(
    [
        ("m", np.array([0.6])),
        ("g", np.array([9.81])),
    ]
)


class QuadrotorMPC(MPC):

    def __init__(
            self,
            params: dict[str, np.ndarray] | None = None,
            learnable_params: list[str] | None = None,
            discount_factor: float = 0.99,
            n_batch: int = 64
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
            nominal_param=params,
            cost_type="LINEAR_LS",
            exact_hess_dyn=False,
            name="quadrotor_lls",
            learnable_param=learnable_params,
            sensitivity_ocp=False,
        )

        ocp_sens = export_parametric_ocp(
            nominal_param=params,
            cost_type="LINEAR_LS",
            exact_hess_dyn=True,
            name="quadrotor_lls",
            learnable_param=learnable_params,
            sensitivity_ocp=True,
        )

        self.given_default_param_dict = params

        super().__init__(
            ocp=ocp,
            ocp_sensitivity=ocp_sens,
            discount_factor=discount_factor,
            n_batch=n_batch,
        )


def export_parametric_ocp(
        nominal_param: dict[str, np.ndarray],
        cost_type: str = "LINEAR_LS",
        exact_hess_dyn: bool = True,
        name: str = "quadrotor",
        learnable_param: list[str] | None = None,
        sensitivity_ocp=False,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ######## Dimensions ########
    N_horizon = 50
    dt = 0.005
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon*dt



    ######## Model ########
    # Quadrotor parameters
    model_params = read_from_yaml("./examples/quadrotor/model_params.yaml")

    x, u, p, rhs, rhs_func = get_rhs_quadrotor(model_params, model_fidelity="low")
    ocp.model.name = name
    ocp.model.f_expl_expr = rhs
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


    # ocp = translate_learnable_param_to_p_global(
    #     nominal_param=nominal_param,
    #     learnable_param=learnable_param if learnable_param is not None else [],
    #     ocp=ocp,
    # )

    ######## Cost ########
    if cost_type == "LINEAR_LS":
        Q = np.diag([0, 0, 0,
                     1e1, 1e1, 1e1, 1e1,
                     1e6, 1e6, 1e6,
                     1e-1, 1e-1, 1e5])
        R = np.diag([1, 1, 1, 1])
        Qe = 100 * Q

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Qe

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[nx: nx + nu, :] = np.eye(nu)
        ocp.cost.Vu = Vu

        # constraints
        ocp.constraints.idxbx = np.array([2])
        ocp.constraints.lbx = np.array([-model_params["bound_z"]*10])
        ocp.constraints.ubx = np.array([model_params["bound_z"]])
        ocp.constraints.idxbx_e = np.array([2])
        ocp.constraints.lbx_e = np.array([-model_params["bound_z"]*10])
        ocp.constraints.ubx_e = np.array([model_params["bound_z"]])

        ocp.constraints.idxsbx = np.array([0])
        ocp.cost.zu = ocp.cost.zl = np.array([0])#np.array([5e7])
        ocp.cost.Zu = ocp.cost.Zl = np.array([1e10])
        #ocp.constraints.idxsbx_e = np.array([0])
        #ocp.cost.zu = ocp.cost.zl_e = np.array([0])
        #ocp.cost.Zu = ocp.cost.Zl_e = np.array([1e1])

        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx_e = Vx_e

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref[3] = 1
        ocp.cost.yref[nx:nx+nu] = 970.437
        ocp.cost.yref_e = np.zeros((ny_e,))
        ocp.cost.yref_e[3] = 1
        ocp.cost.yref_e[nx:nx + nu] = 970.437

    else:
        raise ValueError(f"Cost type {cost_type} not supported.")
        # TODO: Implement NONLINEAR_LS with y_expr = sqrt(Q) * x and sqrt(R) * u

    ######## Constraints ########
    ocp.constraints.x0 = np.array([0] * 13)
    ocp.constraints.lbu = np.array([0] * 4)
    ocp.constraints.ubu = np.array([model_params["motor_omega_max"]] * 4)
    ocp.constraints.idxbu = np.array(range(4))

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = (
        "GAUSS_NEWTON" if cost_type == "LINEAR_LS" else "EXACT"
    )
    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 2

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    if sensitivity_ocp:
        set_standard_sensitivity_options(ocp)

    #ocp.solver_options.qp_solver_ric_alg = 1

    #####################################################

    # if sensitivity_ocp:
    #    if cost_type == "EXTERNAL":
    #        pass
    #    else:
    #        W = cost_matrix_casadi(ocp.model)
    #        W_e = W[:4, :4]
    #        yref = yref_casadi(ocp.model)
    #        yref_e = yref[:4]
    #        ocp.translate_cost_to_external_cost(W=W, W_e=W_e, yref=yref, yref_e=yref_e)
    #    set_standard_sensitivity_options(ocp)

    # if isinstance(ocp.model.p, struct_symSX):
    #     ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []
    #
    # if isinstance(ocp.model.p_global, struct_symSX):
    #     ocp.model.p_global = (
    #         ocp.model.p_global.cat if ocp.model.p_global is not None else None
    #     )
    return ocp
