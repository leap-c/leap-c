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
from leap_c.ocp.acados.diff_ocp import AcadosDiffOcp
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer

ChainAcadosParamInterface = Literal["global", "stagewise"]
"""Determines the exposed parameter interface of the controller.
"global" means that learnable parameters are the same for all stages of the horizon,
while "stagewise" means that learnable parameters can vary between stages.
"""


def export_parametric_ocp(
    param_interface: ChainAcadosParamInterface,
    x_ref: np.ndarray,
    fix_point: np.ndarray,
    name: str = "chain",
    N_horizon: int = 30,  # noqa: N803
    T_horizon: float = 6.0,
    n_mass: int = 5,
) -> AcadosDiffOcp:
    ocp = AcadosDiffOcp(N_horizon=N_horizon)

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    # Register only learnable parameters
    q_diag_sqrt_val = np.ones(3 * (n_mass - 1) + 3 * (n_mass - 2))
    r_diag_sqrt_val = 1e-1 * np.ones(3)

    q_diag_sqrt = ocp.register_param(
        "q_diag_sqrt",
        default=q_diag_sqrt_val,
        space=gym.spaces.Box(
            low=0.5 * q_diag_sqrt_val, high=1.5 * q_diag_sqrt_val, dtype=np.float64
        ),
        differentiable=True,
        splits="stagewise" if param_interface == "stagewise" else "global",
    )
    r_diag_sqrt = ocp.register_param(
        "r_diag_sqrt",
        default=r_diag_sqrt_val,
        space=gym.spaces.Box(
            low=0.5 * r_diag_sqrt_val, high=1.5 * r_diag_sqrt_val, dtype=np.float64
        ),
        differentiable=True,
        splits="stagewise" if param_interface == "stagewise" else "global",
    )

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

    # Physical parameters (sunsetted "fix" interface)
    dyn_param_dict = OrderedDict(
        [
            ("D", np.repeat([1.0, 1.0, 1.0], n_mass - 1)),
            ("L", np.repeat([0.033, 0.033, 0.033], n_mass - 1)),
            ("C", np.repeat([0.1, 0.1, 0.1], n_mass - 1)),
            ("m", np.repeat([0.033], n_mass - 1)),
            ("w", np.repeat([0.0, 0.0, 0.0], n_mass - 2)),
        ]
    )

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
        ca.SX(),
        T_horizon / N_horizon,  # type:ignore
    )

    ######## Cost ########
    Q = ca.diag(q_diag_sqrt**2)
    R = ca.diag(r_diag_sqrt**2)
    x_res = ocp.model.x.cat - x_ref

    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = 0.5 * (x_res.T @ Q @ x_res + ocp.model.u.T @ R @ ocp.model.u)
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
