from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.chain.acados_ocp import (
    export_parametric_ocp,
    ChainInitializer,
    create_chain_params,
)
from leap_c.examples.chain.acados_ocp import ChainAcadosParamInterface
from leap_c.examples.chain.dynamics import define_f_expl_expr
from leap_c.examples.chain.utils.resting_chain_solver import RestingChainSolver
from leap_c.ocp.acados.parameters import AcadosParameterManager, AcadosParameter
from leap_c.ocp.acados.diff_mpc import collate_acados_diff_mpc_ctx, AcadosDiffMpcCtx
from leap_c.ocp.acados.torch import AcadosDiffMpc
from acados_template.acados_ocp import AcadosOcp


@dataclass(kw_only=True)
class ChainControllerConfig:
    """Configuration for the Chain controller.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
            The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal
            node N).
        T_horizon: The duration of the MPC horizon. One step during planning
            will equal T_horizon/N_horizon simulation time.
        n_mass: The number of masses in the chain.
        param_interface: Determines the exposed paramete interface of the
            controller.
    """

    N_horizon: int = 20
    T_horizon: float = 0.25
    n_mass: int = 5
    param_interface: ChainAcadosParamInterface = "global"


class ChainController(ParameterizedController):
    """Acados-based controller for the hanging chain system.
    The state and action correspond to the observation and action of the Chain environment.
    The cost function takes the form of a weighted least-squares cost on the full state and action
    and the dynamics correspond to the simulated ODE also found
    in the Chain environment (using RK4). The inequality constraints
    are box constraints on the action.

    Attributes:
        cfg: A configuration object containing high-level settings for the MPC problem,
            such as horizon length.
        ocp: The acados ocp object representing the optimal control problem structure.
        param_manager: For managing the parameters of the ocp.
        diff_mpc: An object wrapping the acados ocp solver for differentiable MPC solving.
        collate_fn_map: A mapping for collating AcadosDiffMpcCtx objects in batches.
    """

    cfg: ChainControllerConfig
    ocp: AcadosOcp
    param_manager: AcadosParameterManager
    diff_mpc: AcadosDiffMpc

    collate_fn_map: dict[type, Callable] = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: ChainControllerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        """Initializes the ChainController.

        Args:
            cfg: cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length and maximum force. If not provided,
                a default config is used.
            params: An optional list of parameters to define the
                ocp object. If not provided, default parameters for the Chain
                system will be created based on the cfg.
            export_directory: Directory to export the acados ocp files.
        """
        super().__init__()
        self.cfg = ChainControllerConfig() if cfg is None else cfg
        params = (
            create_chain_params(
                self.cfg.param_interface,
                self.cfg.n_mass,
                N_horizon=self.cfg.N_horizon,
            )
            if params is None
            else params
        )

        self.param_manager = AcadosParameterManager(
            parameters=params,
            N_horizon=self.cfg.N_horizon,  # type:ignore
        )

        fix_point = np.zeros(3)

        # find resting reference position
        rest_length = self.param_manager.parameters["L"].default[0]
        pos_last_mass_ref = fix_point + np.array([rest_length * (self.cfg.n_mass - 1), 0, 0])

        dyn_param_dict = {k: self.param_manager.parameters[k].default for k in "LDCmw"}

        resting_chain_solver = RestingChainSolver(
            n_mass=self.cfg.n_mass,
            f_expl=define_f_expl_expr,
            **dyn_param_dict,
            fix_point=fix_point,
        )

        x_ref, u_ref = resting_chain_solver(p_last=pos_last_mass_ref)

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            x_ref=x_ref,
            fix_point=fix_point,
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            n_mass=self.cfg.n_mass,
        )

        initializer = ChainInitializer(self.ocp, x_ref=x_ref)

        self.diff_mpc = AcadosDiffMpc(
            self.ocp,
            initializer=initializer,
            export_directory=export_directory,
        )

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=obs.shape[0]
        )
        ctx, u0, x, u, value = self.diff_mpc(obs, p_global=param, p_stagewise=p_stagewise, ctx=ctx)
        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.diff_mpc.sensitivity(ctx, field_name="du0_dp_global")

    @property
    def param_space(self) -> gym.Space:
        return self.param_manager.get_param_space(dtype=np.float64)

    def default_param(self, obs) -> np.ndarray:
        return self.param_manager.learnable_parameters_default.cat.full().flatten()  # type:ignore
