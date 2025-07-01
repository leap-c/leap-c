from dataclasses import asdict
from typing import Any

import casadi as ca
import gymnasium as gym
import numpy as np
import torch
from casadi.tools import struct_symSX

from acados_template import AcadosModel, AcadosOcp
from leap_c.ocp.acados.parameters import AcadosParamManager
from leap_c.controller import ParameterizedController
from leap_c.examples.cartpole.config import CartPoleParams, make_default_cartpole_params
from leap_c.examples.util import (
    assign_lower_triangular,
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.ocp.acados.torch import AcadosDiffMpc


class CartPoleController(ParameterizedController):
    """
    Describes an inverted pendulum on a cart.
    The (possibly learnable) parameters of the system are given by
        ---------Dynamics---------
        M: mass of the cart [kg]
        m: mass of the ball [kg]
        g: gravity constant [m/s^2]
        l: length of the rod [m]

        ---------Cost---------
        The parameters of the quadratic cost matrix describe a cholesky factorization of the cost matrix.
        In more detail, the cost matrix W is calculated like this:
        L_diag = np.diag([L11, L22, L33, L44, L55]) # cost matrix factorization diagonal
        L_diag[np.tril_indices_from(L_diag, -1)] = L_lower_offdiag
        W = L@L.T

        If the cost is a least squares cost (see docstring of __init__), the parameters
        c1, c2, c3, c4, c5 are not used.
        Instead, the parameters xref1, xref2, xref3, xref4, uref are used for the reference vector.
        If the cost is not the least squares cost, the parameters
        xref1, xref2, xref3, xref4, uref are not used.
        Instead, the parameters c1, c2, c3, c4, c5 are used for the linear cost vector.

        The possible costs are either a least squares cost or a general quadratic cost.
        The least squares cost takes the form of:
            z_ref = cat(xref, uref)
            cost = 0.5 * (z - z_ref).T @ W @ (z - z_ref), where W is the quadratic cost matrix from above.
        The general quadratic cost takes the form of:
            z = cat(x, u)
            cost = 0.5 * z.T @ W @ z + c.T @ z, where W is the quadratic cost matrix from above

    """

    def __init__(
        self,
        params: CartPoleParams | None = None,
        N_horizon: int = 20,
        T_horizon: float = 1.0,
        Fmax: float = 80.0,
        discount_factor: float = 0.99,
        exact_hess_dyn: bool = True,
        cost_type: str = "NONLINEAR_LS",
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
            Fmax: The maximum force that can be applied to the cart.
            discount_factor: The discount factor for the cost.
            n_batch: The batch size the MPC should be able to process
                (currently this is static).
            exact_hess_dyn: If False, the contributions of the dynamics will be left out of the Hessian.
            cost_type: The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS".
        """
        super().__init__()
        self.params = make_default_cartpole_params() if params is None else params
        tuple_params = tuple(asdict(self.params).values())

        param_manager = AcadosParamManager(params=tuple_params, N_horizon=N_horizon)

        self.ocp = export_parametric_ocp(
            param_manager=param_manager,
            cost_type=cost_type,
            exact_hess_dyn=exact_hess_dyn,
            name="cartpole",
            N_horizon=N_horizon,
            tf=T_horizon,
            Fmax=Fmax,
        )

        self.diff_mpc = AcadosDiffMpc(self.ocp, discount_factor=discount_factor)

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        x0 = torch.as_tensor(obs, dtype=torch.float64)
        p_global = torch.as_tensor(param, dtype=torch.float64)
        ctx, u0, x, u, value = self.diff_mpc(
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


def define_f_expl_expr(ocp: AcadosOcp, param_manager: AcadosParamManager) -> ca.SX:
    model = ocp.model

    M = param_manager.get("M")
    m = param_manager.get("m")
    g = param_manager.get("g")
    l = param_manager.get("l")

    theta = model.x[1]
    v1 = model.x[2]
    dtheta = model.x[3]

    F = model.u[0]

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m * cos_theta * cos_theta
    f_expl = ca.vertcat(
        v1,
        dtheta,
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F)
        / denominator,
        (
            -m * l * cos_theta * sin_theta * dtheta * dtheta
            + F * cos_theta
            + (M + m) * g * sin_theta
        )
        / (l * denominator),
    )

    return f_expl  # type:ignore


def define_disc_dyn_expr(
    model: AcadosModel, param_manager: AcadosParamManager, dt: float
) -> ca.SX:
    f_expl = define_f_expl_expr(model, param_manager)

    x = model.x
    u = model.u

    # discrete dynamics via RK4
    p = ca.vertcat(*find_param_in_p_or_p_global(["M", "m", "g", "l"], model).values())

    ode = ca.Function("ode", [x, u, p], [f_expl])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)  # type:ignore
    k3 = ode(x + dt / 2 * k2, u, p)  # type:ignore
    k4 = ode(x + dt * k3, u, p)  # type:ignore

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # type:ignore


def define_cost_matrix(
    model: AcadosModel, param_manager: AcadosParamManager
) -> tuple[ca.SX, ca.SX] | tuple[np.ndarray, np.ndarray]:
    q_diag = param_manager.get("q_diag")
    r_diag = param_manager.get("r_diag")
    q_diag_e = param_manager.get("q_diag_e")

    if isinstance(q_diag, np.ndarray) and isinstance(r_diag, np.ndarray):
        W = np.diag([q_diag, r_diag])
        W = W @ W.T
    else:
        # TODO (Jasper): Check whether we can even vertcat the diagonal elementst
        #               if numpy array
        W = ca.diag(ca.vertcat(q_diag, r_diag))
        W = W @ W.T

    if isinstance(q_diag_e, np.ndarray):
        W_e = np.diag(q_diag_e)
    else:
        W_e = ca.diag(q_diag_e)

    return W, W_e


def define_yref(param_manager: AcadosParamManager) -> np.ndarray:
    xref = param_manager.get("xref")
    uref = param_manager.get("uref")

    if isinstance(xref, np.ndarray) and isinstance(uref, np.ndarray):
        return np.concatenate([xref, uref])

    return ca.vertcat(xref, uref)  # type:ignore


def define_cost_expr_ext_cost(
    ocp: AcadosOcp, param_manager: AcadosParamManager
) -> ca.SX:
    yref = define_yref(param_manager)
    W = define_cost_matrix(ocp.model, param_manager)[0]
    y = ca.vertcat(ocp.model.x, ocp.model.u)  # type:ignore
    return 0.5 * ca.mtimes(ca.mtimes((y - yref).T, W), (y - yref))


def define_cost_expr_ext_cost_e(
    ocp: AcadosOcp, param_manager: AcadosParamManager
) -> ca.SX:
    yref_e = param_manager.get("yref_e")
    W_e = define_cost_matrix(ocp.model, param_manager)[1]
    y_e = ocp.model.x
    return 0.5 * ca.mtimes(ca.mtimes((y_e - yref_e).T, W_e), (y_e - yref_e))


def export_parametric_ocp(
    param_manager: AcadosParamManager,
    cost_type: str = "NONLINEAR_LS",
    exact_hess_dyn: bool = True,
    name: str = "cartpole",
    Fmax: float = 80.0,
    N_horizon: int = 50,
    tf: float = 2.0,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon

    param_manager.assign_to_ocp(ocp)

    ocp.solver_options.tf = tf
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    ######## Model ########
    ocp.model.name = name

    ocp.dims.nx = 4
    ocp.dims.nu = 1

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)  # type:ignore
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)  # type:ignore

    ocp.model.disc_dyn_expr = define_disc_dyn_expr(
        model=ocp.model, param_manager=param_manager, dt=dt
    )  # type:ignore

    ######## Cost ########
    if cost_type == "EXTERNAL":
        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = define_cost_expr_ext_cost(ocp, param_manager)  # type:ignore

        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = define_cost_expr_ext_cost_e(ocp, param_manager)  # type:ignore

        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.exact_hess_cost = True
        ocp.solver_options.exact_hess_constr = True
    elif cost_type == "NONLINEAR_LS":
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ocp.cost.W = define_cost_matrix(ocp.model, param_manager=param_manager)[0]
        ocp.cost.yref = define_yref(param_manager=param_manager)
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)

        ocp.cost.W_e = define_cost_matrix(ocp.model, param_manager=param_manager)[1]
        ocp.cost.yref_e = param_manager.get("xref_e")
        ocp.model.cost_y_expr_e = ocp.model.x

        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    else:
        raise ValueError(f"Cost type {cost_type} not supported.")

    ######## Constraints ########
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([-2.4])
    ocp.constraints.ubx = -ocp.constraints.lbx
    ocp.constraints.idxbx = np.array([0])
    ocp.constraints.lbx_e = np.array([-2.4])
    ocp.constraints.ubx_e = -ocp.constraints.lbx_e
    ocp.constraints.idxbx_e = np.array([0])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.Zu = ocp.cost.Zl = np.array([1e3])
    ocp.cost.zu = ocp.cost.zl = np.array([0.0])

    ocp.constraints.idxsbx_e = np.array([0])
    ocp.cost.Zu_e = ocp.cost.Zl_e = np.array([1e3])
    ocp.cost.zu_e = ocp.cost.zl_e = np.array([0.0])

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"

    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_batch_functionality = True

    return ocp
