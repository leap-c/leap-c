from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch

from leap_c.examples.cartpole.acados_ocp import (
    create_cartpole_params,
    export_parametric_ocp,
    ParamInterface,
    CostType,
)
from leap_c.ocp.acados.parameters import Parameter
from leap_c.controller import ParameterizedController
from leap_c.ocp.acados.torch import AcadosDiffMpc
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx


@dataclass(kw_only=True)
class CartPoleControllerCfg:
    """Configuration for the CartPole controller.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
            The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal
            node N).
        T_horizon: The duration of the MPC horizon. One step during planning
            will equal T_horizon/N_horizon simulation time.
        Fmax: The maximum force that can be applied to the cart.
        cost_type: The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS".
        param_interface: Determines the exposed paramete interface of the
            controller.
    """

    N_horizon: int = 5
    T_horizon: float = 0.25
    Fmax: float = 80
    cost_type: CostType = "NONLINEAR_LS"
    param_interface: ParamInterface = "global"


class CartPoleController(ParameterizedController):
    """Acados based CartPoleController."""

    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: CartPoleControllerCfg | None = None,
        params: list[Parameter] | None = None,
        export_directory: Path | None = None,
    ):
        """Initializes the CartPoleController.

        Args:
            cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length and maximum force.
            params: An optional list of `Parameter` objects to define the
                OCP. If not provided, default parameters for the cart-pole
                system will be created based on the `cfg.param_interface`.
            export_directory: An optional directory path where the generated
                `acados` solver code will be exported.
        """
        super().__init__()
        self.cfg = CartPoleControllerCfg() if cfg is None else cfg
        params = (
            create_cartpole_params(self.cfg.param_interface)
            if params is None
            else params
        )

        self.param_manager = AcadosParamManager(
            params=params, N_horizon=self.cfg.N_horizon
        )

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            cost_type=self.cfg.cost_type,
            name="cartpole",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
        )

        self.diff_mpc = AcadosDiffMpc(self.ocp, export_directory=export_directory)

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
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)  # type:ignore

    def default_param(self, obs) -> np.ndarray:
        return self.param_manager.p_global_values.cat.full().flatten()  # type:ignore
