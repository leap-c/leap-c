from typing import Any

import casadi as ca
import gymnasium as gym
import numpy as np
import torch
from casadi.tools import struct_symSX
from dataclasses import asdict

from acados_template import AcadosOcp
from leap_c.controller import ParameterizedController
from leap_c.examples.pointmass.env import _A_disc, _B_disc
from leap_c.examples.pointmass.config import PointMassParams, make_default_pointmass_params
from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.ocp.acados.torch import AcadosDiffMpc


class PointMassController(ParameterizedController):
    """docstring for PointMassController."""
    def __init__(
            self,
            params: PointMassParams | None = None,
            learnable_params: list[str] | None = None,
            N_horizon: int = 20,
            T_horizon: float = 2.0,
    ) -> None:
        super().__init__()
        self.params = make_default_pointmass_params() if params is None else params
        self.learnable_params = learnable_params if learnable_params is not None else []

        self.ocp = export_parametric_ocp(
            nominal_param=asdict(self.params),
            learnable_params=self.learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
        )
        configure_ocp_solver(ocp=self.ocp, exact_hess_dyn=True)

        self.acados_layer = AcadosDiffMpc(self.ocp)

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        x0 = torch.as_tensor(obs, dtype=torch.float64)
        p_global = torch.as_tensor(param, dtype=torch.float64)
        ctx, u0, x, u, value = self.acados_layer(x0.unsqueeze(0), p_global=p_global.unsqueeze(0), ctx=ctx)
        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.acados_layer.sensitivity(ctx, field_name="du0_dp_global")

    def param_space(self) -> gym.Space:
        # TODO: can't determine the param space because it depends on the learnable parameters
        # we need to define boundaries for every parameter and based on that create a gym.Space
        raise NotImplementedError

    def default_param(self) -> np.ndarray:
        return np.concatenate([np.array(asdict(self.params)[p]).flatten() for p in self.learnable_params])


def _create_diag_matrix(
        _q_sqrt: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [_q_sqrt]):
        return ca.diag(_q_sqrt)
    return np.diag(_q_sqrt)


def _disc_dyn_expr(
        ocp: AcadosOcp,
) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    m = find_param_in_p_or_p_global(["m"], ocp.model)["m"]
    cx = find_param_in_p_or_p_global(["cx"], ocp.model)["cx"]
    cy = find_param_in_p_or_p_global(["cy"], ocp.model)["cy"]
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    A = _A_disc(m=m, cx=cx, cy=cy, dt=dt)
    B = _B_disc(m=m, cx=cx, cy=cy, dt=dt)

    return A @ x + B @ u


def _cost_expr_ext_cost(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    Q_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["q_diag"], ocp.model)["q_diag"]
    )
    R_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["r_diag"], ocp.model)["r_diag"]
    )

    xref = find_param_in_p_or_p_global(["xref"], ocp.model)["xref"]
    uref = find_param_in_p_or_p_global(["uref"], ocp.model)["uref"]

    return 0.5 * (
            ca.mtimes([ca.transpose(x - xref), Q_sqrt.T, Q_sqrt, x - xref])
            + ca.mtimes([ca.transpose(u - uref), R_sqrt.T, R_sqrt, u - uref])
    )


def _cost_expr_ext_cost_e(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x

    Q_sqrt_e = _create_diag_matrix(
        find_param_in_p_or_p_global(["q_diag_e"], ocp.model)["q_diag_e"]
    )

    xref_e = find_param_in_p_or_p_global(["xref_e"], ocp.model)["xref_e"]

    return 0.5 * ca.mtimes([ca.transpose(x - xref_e), Q_sqrt_e.T, Q_sqrt_e, x - xref_e])


def export_parametric_ocp(
        nominal_param: dict[str, np.ndarray | float],
        name: str = "pointmass",
        learnable_params: list[str] | None = None,
        N_horizon: int = 50,
        tf: float = 2.0,
        x0: np.ndarray | None = None,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = N_horizon

    ocp.model.name = name

    ocp.dims.nu = 2
    ocp.dims.nx = 4

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_param, learnable_param=learnable_params, ocp=ocp
    )

    ocp.model.disc_dyn_expr = _disc_dyn_expr(ocp=ocp)
    ocp.model.cost_expr_ext_cost_0 = _cost_expr_ext_cost(ocp=ocp)
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = _cost_expr_ext_cost(ocp=ocp)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = _cost_expr_ext_cost_e(ocp=ocp)
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.constraints.x0 = np.array([1.0, 1.0, 0.0, 0.0]) if x0 is None else x0

    Fmax = 10.0
    # Box constraints on u
    ocp.constraints.lbu = np.array([-Fmax, -Fmax])
    ocp.constraints.ubu = np.array([Fmax, Fmax])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([0.05, 0.05, -20.0, -20.0])
    ocp.constraints.ubx = np.array([3.95, 0.95, 20.0, 20.0])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3])

    ocp.constraints.idxsbx = np.array([0, 1, 2, 3])

    ns = ocp.constraints.idxsbx.size
    ocp.cost.zl = 10000 * np.ones((ns,))
    ocp.cost.Zl = 10 * np.ones((ns,))
    ocp.cost.zu = 10000 * np.ones((ns,))
    ocp.cost.Zu = 10 * np.ones((ns,))

    # Cast parameters to appropriate types for acados
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    configure_ocp_solver(ocp=ocp, exact_hess_dyn=True)

    return ocp


def configure_ocp_solver(ocp: AcadosOcp, exact_hess_dyn: bool) -> None:
    """Configure the OCP solver options."""
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_batch_functionality = True
