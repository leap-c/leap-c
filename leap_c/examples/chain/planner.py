from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leap_c.examples.chain.acados_ocp import (
    ChainAcadosParamInterface,
    ChainInitializer,
    create_chain_params,
    export_parametric_ocp,
)
from leap_c.examples.chain.dynamics import define_f_expl_expr
from leap_c.examples.chain.utils.resting_chain_solver import RestingChainSolver
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch


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


class ChainPlanner(AcadosPlanner[AcadosDiffMpcCtx]):
    """Acados-based controller for the hanging `Chain` system.

    The state and action correspond to the observation and action of the `Chain` environment. The
    cost function takes the form of a weighted least-squares cost on the full state and action and
    the dynamics correspond to the simulated ODE also found in the `Chain` environment (using RK4).
    The inequality constraints are box constraints on the action.

    Attributes:
        cfg: A configuration object containing high-level settings for the MPC problem,
            such as horizon length.
    """

    cfg: ChainControllerConfig

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

        param_manager = AcadosParameterManager(
            parameters=params,
            N_horizon=self.cfg.N_horizon,  # type:ignore
        )

        fix_point = np.zeros(3)

        # find resting reference position
        rest_length = param_manager.parameters["L"].default[0]
        pos_last_mass_ref = fix_point + np.array([rest_length * (self.cfg.n_mass - 1), 0, 0])

        dyn_param_dict = {k: param_manager.parameters[k].default for k in "LDCmw"}

        resting_chain_solver = RestingChainSolver(
            n_mass=self.cfg.n_mass,
            f_expl=define_f_expl_expr,
            **dyn_param_dict,
            fix_point=fix_point,
        )

        x_ref, u_ref = resting_chain_solver(p_last=pos_last_mass_ref)

        ocp = export_parametric_ocp(
            param_manager=param_manager,
            x_ref=x_ref,
            fix_point=fix_point,
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            n_mass=self.cfg.n_mass,
        )

        initializer = ChainInitializer(ocp, x_ref=x_ref)

        diff_mpc = AcadosDiffMpcTorch(
            ocp,
            initializer=initializer,
            export_directory=export_directory,
        )
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)
