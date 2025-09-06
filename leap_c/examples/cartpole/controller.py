from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.cartpole.acados_ocp import (
    CartPoleAcadosCostType,
    CartPoleAcadosParamInterface,
    create_cartpole_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.parameters import AcadosParameterManager, AcadosParameter
from leap_c.ocp.acados.torch import AcadosDiffMpc


@dataclass(kw_only=True)
class CartPoleControllerConfig:
    """Configuration for the CartPole controller.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
            The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal
            node N).
        T_horizon: The simulation time between two MPC nodes will equal
            T_horizon/N_horizon [s] simulation time.
        Fmax: Bounds of the box constraints on the maximum force that can be applied to the cart [N] (hard constraint)
        x_threshold: Bounds of the box constraints of the maximum absolute position of the cart [m] (soft/slacked constraint)
        cost_type: The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS".
        param_interface: Determines the exposed parameter interface of the controller.
    """

    N_horizon: int = 5
    T_horizon: float = 0.25
    Fmax: float = 80.0
    x_threshold: float = 2.4

    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS"
    param_interface: CartPoleAcadosParamInterface = "global"


class CartPoleController(ParameterizedController):
    """Acados-based controller for CartPole, aka inverted pendulum.
    The state and action correspond to the observation and action of the CartPole environment.
    The cost function takes the form of a weighted least-squares cost on the full state and action,
    and the dynamics correspond to the simulated ODE of the standard CartPole environment (using RK4).
    The inequality constraints are box constraints on the action and on the cart position.

    Attributes:
        cfg: A configuration object containing high-level settings for the MPC problem, such as horizon length.
        param_manager: For managing the parameters of the OCP.
        ocp: The acados OCP object representing the optimal control problem.
        diff_mpc: An object wrapping the acados ocp solver for differentiable MPC solving.
        collate_fn_map: A mapping for collating AcadosDiffMpcCtx objects in batches.
    """

    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: CartPoleControllerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        """Initializes the CartPoleController.

        Args:
            cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length and maximum force. If not provided, a default config is used.
            params: An optional list of parameters to define the
                OCP. If not provided, default parameters for the CartPole
                system will be created based on the cfg.
            export_directory: An optional directory path where the generated
                `acados` solver code will be exported.
        """
        super().__init__()
        self.cfg = CartPoleControllerConfig() if cfg is None else cfg
        params = (
            create_cartpole_params(
                param_interface=self.cfg.param_interface,
                N_horizon=self.cfg.N_horizon,
            )
            if params is None
            else params
        )

        self.param_manager = AcadosParameterManager(
            parameters=params, N_horizon=self.cfg.N_horizon
        )

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            cost_type=self.cfg.cost_type,
            name="cartpole",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
            x_threshold=self.cfg.x_threshold,
        )

        self.diff_mpc = AcadosDiffMpc(self.ocp, export_directory=export_directory)

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
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
        return self.param_manager.get_param_space(dtype=np.float32)

    def default_param(self, obs) -> np.ndarray:
        return self.param_manager.learnable_parameters_default.cat.full().flatten()  # type:ignore
