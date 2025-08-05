from dataclasses import asdict
from pathlib import Path
from typing import Any

from acados_template import AcadosOcp
import casadi as ca
import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.pointmass.config import (
    PointMassParams,
    make_default_pointmass_params,
)
from leap_c.examples.pointmass.env import _A_disc, _B_disc
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.parameters import AcadosParamManager
from leap_c.ocp.acados.torch import AcadosDiffMpc


class PointMassController(ParameterizedController):
    """docstring for PointMassController."""

    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        params: PointMassParams | None = None,
        N_horizon: int = 20,
        T_horizon: float = 2.0,
        stagewise: bool = False,
        export_directory: Path | None = None,
    ):
        super().__init__()
        self.params = (
            make_default_pointmass_params(stagewise) if params is None else params
        )
        tuple_params = tuple(asdict(self.params).values())

        self.param_manager = AcadosParamManager(
            params=tuple_params,
            N_horizon=N_horizon,  # type: ignore
        )

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            N_horizon=N_horizon,
            tf=T_horizon,
        )

        self.diff_mpc = AcadosDiffMpc(self.ocp, export_directory=export_directory)

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        x = obs[:, :4]
        p_stagewise = self.param_manager.combine_parameter_values(batch_size=x.shape[0])
        ctx, u0, x, u, value = self.diff_mpc(
            x, p_global=param, p_stagewise=p_stagewise, ctx=ctx
        )
        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.diff_mpc.sensitivity(ctx, field_name="du0_dp_global")

    @property
    def param_space(self) -> gym.Space:
        low, high = self.param_manager.get_p_global_bounds()
        return gym.spaces.Box(low=low, high=high, dtype=np.float64)  # type:ignore

    def default_param(self, obs) -> np.ndarray:
        return self.param_manager.p_global_values.cat.full().flatten()  # type:ignore


def _create_diag_matrix(
    _q_sqrt: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [_q_sqrt]):
        return ca.diag(_q_sqrt)
    return np.diag(_q_sqrt)


def _disc_dyn_expr(
    ocp: AcadosOcp,
    param_manager: AcadosParamManager,
) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    m = param_manager.get("m").item()
    cx = param_manager.get("cx").item()
    cy = param_manager.get("cy").item()
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon  # type: ignore

    A = _A_disc(m=m, cx=cx, cy=cy, dt=dt)
    B = _B_disc(m=m, cx=cx, cy=cy, dt=dt)

    return A @ x + B @ u  # type: ignore


def _cost_expr_ext_cost(ocp: AcadosOcp, param_manager: AcadosParamManager) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    Q_sqrt = _create_diag_matrix(param_manager.get("q_sqrt_diag"))
    Q = Q_sqrt.T @ Q_sqrt
    R_sqrt = _create_diag_matrix(param_manager.get("r_sqrt_diag"))
    R = R_sqrt.T @ R_sqrt

    xref = param_manager.get("xref")
    uref = param_manager.get("uref")

    return 0.5 * ((x - xref).T @ Q @ (x - xref) + (u - uref).T @ R @ (u - uref))


def _cost_expr_ext_cost_e(ocp: AcadosOcp, param_manager: AcadosParamManager) -> ca.SX:
    x = ocp.model.x

    Q_sqrt_e = _create_diag_matrix(param_manager.get("q_sqrt_diag"))
    xref_e = param_manager.get("xref")

    return 0.5 * ca.mtimes([ca.transpose(x - xref_e), Q_sqrt_e.T, Q_sqrt_e, x - xref_e])


def export_parametric_ocp(
    param_manager: AcadosParamManager,
    name: str = "pointmass",
    Fmax: float = 10.0,
    N_horizon: int = 50,
    tf: float = 2.0,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = tf

    param_manager.assign_to_ocp(ocp)

    ######## Model ########
    ocp.model.name = name

    ocp.dims.nu = 2
    ocp.dims.nx = 4

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)  # type: ignore
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)  # type: ignore

    ocp.model.disc_dyn_expr = _disc_dyn_expr(ocp=ocp, param_manager=param_manager)

    ######## Cost ########
    ocp.model.cost_expr_ext_cost_0 = _cost_expr_ext_cost(
        ocp=ocp, param_manager=param_manager
    )
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = _cost_expr_ext_cost(
        ocp=ocp, param_manager=param_manager
    )
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = _cost_expr_ext_cost_e(
        ocp=ocp, param_manager=param_manager
    )
    ocp.cost.cost_type_e = "EXTERNAL"

    ######## Constraints ########
    ocp.constraints.x0 = np.array([1.0, 1.0, 0.0, 0.0])

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

    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.exact_hess_dyn = True
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1

    return ocp
