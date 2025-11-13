from copy import deepcopy
from typing import Literal, OrderedDict

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp, AcadosOcpFlattenedIterate
from casadi.tools import entry, struct_symSX

from leap_c.examples.chain.dynamics import define_f_expl_expr
from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.data import AcadosOcpSolverInput
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

ChainAcadosParamInterface = Literal["global", "stagewise"]
"""Determines the exposed parameter interface of the controller.
"global" means that learnable parameters are the same for all stages of the horizon,
while "stagewise" means that learnable parameters can vary between stages.
"""


def create_chain_params(
    param_interface: ChainAcadosParamInterface = "global",
    n_mass: int = 5,
    N_horizon: int = 30,
) -> list[AcadosParameter]:
    """Returns a list of parameters used in the chain controller.

    Args:
        param_interface: Determines the exposed parameter interface of the controller.
        n_mass: Number of masses in the chain.
        N_horizon: The number of steps in the MPC horizon.
    """
    q_diag_sqrt = np.ones(3 * (n_mass - 1) + 3 * (n_mass - 2))
    r_diag_sqrt = 1e-1 * np.ones(3)

    return [
        # dynamics parameters
        AcadosParameter(
            "L", default=np.repeat([0.033, 0.033, 0.033], n_mass - 1)
        ),  # rest length of spring [m]
        AcadosParameter(
            "D", default=np.repeat([1.0, 1.0, 1.0], n_mass - 1)
        ),  # spring stiffness [N/m]
        AcadosParameter(
            "C", default=np.repeat([0.1, 0.1, 0.1], n_mass - 1)
        ),  # damping coefficient [Ns/m]
        AcadosParameter(
            "m", default=np.repeat([0.033], n_mass - 1)
        ),  # mass of each of the masses [kg]
        AcadosParameter(
            "w", default=np.repeat([0.0, 0.0, 0.0], n_mass - 2)
        ),  # disturbance on intermediate masses [N]
        # cost parameters
        AcadosParameter(
            "q_diag_sqrt",  # cost weights of state residuals
            default=q_diag_sqrt,
            space=gym.spaces.Box(low=0.5 * q_diag_sqrt, high=1.5 * q_diag_sqrt, dtype=np.float64),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),
        AcadosParameter(
            "r_diag_sqrt",  # cost weights of action residuals
            default=r_diag_sqrt,
            space=gym.spaces.Box(low=0.5 * r_diag_sqrt, high=1.5 * r_diag_sqrt, dtype=np.float64),
            interface="learnable",
            end_stages=list(range(N_horizon)) if param_interface == "stagewise" else [],
        ),
    ]


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    x_ref: np.ndarray,
    fix_point: np.ndarray,
    name: str = "chain",
    N_horizon: int = 30,  # noqa: N803
    T_horizon: float = 6.0,
    n_mass: int = 5,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    ######## Model ########
    ocp.model.name = name

    ocp.model.x = struct_symSX(
        [
            entry("pos", shape=(3, 1), repeat=n_mass - 1),
            entry("vel", shape=(3, 1), repeat=n_mass - 2),
        ]
    )
    ocp.model.u = ca.SX.sym("u", 3, 1)  # type: ignore
    nx = ocp.model.x.cat.shape[0]
    nu = ocp.model.u.shape[0]

    x = ocp.model.x
    u = ocp.model.u
    dyn_param_dict = OrderedDict(
        [
            ("D", param_manager.get("D")),
            ("L", param_manager.get("L")),
            ("C", param_manager.get("C")),
            ("m", param_manager.get("m")),
            ("w", param_manager.get("w")),
        ]
    )

    p_cat_sym = ca.vertcat(*[v for v in dyn_param_dict.values() if not isinstance(v, np.ndarray)])
    f_expl = define_f_expl_expr(
        x=x,
        u=u,
        p=dyn_param_dict,
        fix_point=fix_point,
    )
    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl,
        x.cat,
        u,
        p_cat_sym,
        T_horizon / N_horizon,  # type:ignore
    )

    ######## Cost ########
    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    Q = ca.diag(q_diag_sqrt**2)
    R = ca.diag(r_diag_sqrt**2)
    x_res = ocp.model.x.cat - x_ref
    u = ocp.model.u

    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = 0.5 * (x_res.T @ Q @ x_res + u.T @ R @ u)
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = 0.5 * (x_res.T @ Q @ x_res)

    ######## Constraints ########
    umax = np.ones((nu,))
    ocp.constraints.lbu = -umax
    ocp.constraints.ubu = umax
    ocp.constraints.idxbu = np.array(range(nu))
    # load dummy initial state, will be overwritten before solving
    ocp.constraints.x0 = x_ref.reshape((nx,))

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.qp_tol = 1e-7

    # flatten x
    if isinstance(ocp.model.x, struct_symSX):
        ocp.model.x = ocp.model.x.cat

    return ocp


class ChainInitializer(AcadosDiffMpcInitializer):
    """Initializer for the chain controller."""

    def __init__(self, ocp: AcadosOcp, x_ref: np.ndarray):
        """Initialize the state variables in the iterate of the controller.

        The state variables are initialized with the state set to the reference state.
        All other variables in the iterate are set to zero.

        Args:
            ocp: The acados OCP object.
            x_ref: The reference state used to initialize the state variables.

        """
        iterate = ocp.create_default_initial_iterate().flatten()
        iterate.x = np.tile(x_ref, ocp.solver_options.N_horizon + 1)  # type:ignore
        self.default_iterate = iterate

    def single_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedIterate:
        """Returns a single iterate for the given solver input. Using the default iterate."""
        return deepcopy(self.default_iterate)
