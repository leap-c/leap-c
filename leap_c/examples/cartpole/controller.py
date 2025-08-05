from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch

from leap_c.examples.cartpole.acados_ocp import create_cartpole_params, export_parametric_ocp, ParamInterface
from leap_c.ocp.acados.parameters import Parameter
from leap_c.controller import ParameterizedController
from leap_c.ocp.acados.torch import AcadosDiffMpc
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx


@dataclass(kw_only=True)
class CartPoleControllerCfg:
    """Configuration for the CartPole controller.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
            The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal node N).
        T_horizon: The length (meaning time) of the MPC horizon.
            One step in the horizon will equal T_horizon/N_horizon simulation time.
        Fmax: The maximum force that can be applied to the cart.
        cost_type: The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS".
        param_interface: Specifies how parameters are handled. "global" means they are the same across all stages,
            while "stagewise" (not shown in the default) would allow them to vary.
    """

    N_horizon: int = 5
    T_horizon: float = 0.25
    Fmax: float = 80
    cost_type: Literal["EXTERNAL", "NONLINEAR_LS"] = "NONLINEAR_LS"
    param_interface: ParamInterface = "global"


class CartPoleController(ParameterizedController):
    """acados based CartPoleController."""

    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: CartPoleControllerCfg,
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

        self.cfg = cfg
        self.params = (
            create_cartpole_params(cfg.param_interface) if params is None else params
        )

        self.param_manager = AcadosParamManager(
            params=self.params, N_horizon=N_horizon  # type:ignore
        )

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            cost_type=cfg.cost_type,
            name="cartpole",
            N_horizon=cfg.N_horizon,
            T_horizon=cfg.T_horizon,
            Fmax=cfg.Fmax,
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
