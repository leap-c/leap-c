from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, OrderedDict

from acados_template import (
    AcadosOcp,
    AcadosOcpFlattenedIterate,
)
import casadi as ca
from casadi.tools import entry, struct_symSX
import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.chain.config import ChainParams, make_default_chain_params
from leap_c.examples.chain.utils.dynamics import rk4_integrator_casadi, get_f_expl_expr
from leap_c.examples.chain.utils.resting_chain_solver import RestingChainSolver
from leap_c.ocp.acados.data import AcadosOcpSolverInput
from leap_c.ocp.acados.initializer import (
    AcadosDiffMpcInitializer,
    create_zero_iterate_from_ocp,
)
from leap_c.ocp.acados.parameters import AcadosParamManager
from leap_c.ocp.acados.diff_mpc import collate_acados_diff_mpc_ctx, AcadosDiffMpcCtx
from leap_c.ocp.acados.torch import AcadosDiffMpc


class ChainController(ParameterizedController):
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        params: ChainParams | None = None,
        N_horizon: int = 20,
        T_horizon: float = 1.0,
        discount_factor: float = 1.0,
        n_mass: int = 5,
        pos_last_mass_ref: np.ndarray | None = None,
        stagewise: bool = False,
        export_directory: Path | None = None,
    ):
        super().__init__()
        params = (
            make_default_chain_params(n_mass, stagewise) if params is None else params
        )

        # find resting reference position
        if pos_last_mass_ref is None:
            pos_last_mass_ref = params.fix_point.value + np.array(
                [0.033 * (n_mass - 1), 0, 0]
            )

        resting_chain_solver = RestingChainSolver(
            n_mass=n_mass, f_expl=get_f_expl_expr, params=params
        )
        x_ref, u_ref = resting_chain_solver(p_last=pos_last_mass_ref)

        self.param_manager = AcadosParamManager(
            params=asdict(params).values(),
            N_horizon=N_horizon,  # type:ignore
        )

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            x_ref=x_ref,
            N_horizon=N_horizon,
            tf=T_horizon,
            n_mass=n_mass,
        )

        initializer = ChainInitializer(self.ocp, x_ref=x_ref)
        self.diff_mpc = AcadosDiffMpc(
            self.ocp,
            initializer=initializer,
            discount_factor=discount_factor,
            export_directory=export_directory,
        )

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        p_stagewise = self.param_manager.combine_parameter_values(
            batch_size=obs.shape[0]
        )
        ctx, u0, x, u, value = self.diff_mpc(
            obs, p_global=param, p_stagewise=p_stagewise, ctx=ctx
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


def export_parametric_ocp(
    param_manager: AcadosParamManager,
    x_ref: np.ndarray,
    name: str = "chain",
    N_horizon: int = 30,  # noqa: N803
    tf: float = 6.0,
    n_mass: int = 5,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = tf

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

    p_cat_sym = ca.vertcat(
        *[v for v in dyn_param_dict.values() if not isinstance(v, np.ndarray)]
    )
    f_expl = get_f_expl_expr(
        x=x,
        u=u,
        p=dyn_param_dict,
        x0=param_manager.get("fix_point"),  # type:ignore
    )
    ocp.model.disc_dyn_expr = rk4_integrator_casadi(
        f_expl,
        x.cat,
        u,
        p_cat_sym,
        tf / N_horizon,  # type:ignore
    )

    ######## Cost ########
    q_sqrt_diag = param_manager.get("q_sqrt_diag")
    r_sqrt_diag = param_manager.get("r_sqrt_diag")
    Q = ca.diag(q_sqrt_diag) @ ca.diag(q_sqrt_diag).T
    R = ca.diag(r_sqrt_diag) @ ca.diag(r_sqrt_diag).T
    x_res = ocp.model.x.cat - x_ref
    u = ocp.model.u

    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = 0.5 * (x_res.T @ Q @ x_res + u.T @ R @ u)
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = 0.5 * (x_res.T @ Q @ x_res)

    ######## Constraints ########
    umax = 1 * np.ones((nu,))
    ocp.constraints.lbu = -umax
    ocp.constraints.ubu = umax
    ocp.constraints.idxbu = np.array(range(nu))
    ocp.constraints.x0 = x_ref.reshape((nx,))

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.exact_hess_dyn = True
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.qp_tol = 1e-7

    # flatten x
    if isinstance(ocp.model.x, struct_symSX):
        ocp.model.x = ocp.model.x.cat

    return ocp


class ChainInitializer(AcadosDiffMpcInitializer):
    def __init__(self, ocp: AcadosOcp, x_ref: np.ndarray):
        iterate = create_zero_iterate_from_ocp(ocp).flatten()
        iterate.x = np.tile(x_ref, ocp.solver_options.N_horizon + 1)
        self.default_iterate = iterate

    def single_iterate(
        self, solver_input: AcadosOcpSolverInput
    ) -> AcadosOcpFlattenedIterate:
        return deepcopy(self.default_iterate)
