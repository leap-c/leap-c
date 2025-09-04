from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.pointmass.acados_ocp import (
    PointMassAcadosParamInterface,
    create_pointmass_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.parameters import AcadosParameterManager, AcadosParameter
from leap_c.ocp.acados.torch import AcadosDiffMpc


@dataclass(kw_only=True)
class PointMassControllerConfig:
    """Configuration for the PointMass controller.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
        T_horizon: The duration of the MPC horizon.
        Fmax: The maximum force that can be applied to the point mass.
        param_interface: Determines the exposed parameter interface of the controller.
    """

    N_horizon: int = 20
    T_horizon: float = 2.0
    Fmax: float = 10.0
    param_interface: PointMassAcadosParamInterface = "global"


class PointMassController(ParameterizedController):
    """acados-based controller for PointMass"""

    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: PointMassControllerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        """Initializes the PointMassController.

        Args:
            cfg: Configuration object for the MPC problem.
            params: Optional list of `Parameter` objects for the OCP.
            export_directory: Optional directory for generated acados solver code.
        """
        super().__init__()
        self.cfg = PointMassControllerConfig() if cfg is None else cfg
        params = (
            create_pointmass_params(
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
            name="pointmass",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
        )

        self.diff_mpc = AcadosDiffMpc(self.ocp, export_directory=export_directory)

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=obs.shape[0]
        )
        # remove wind field from observation, this is only observed by
        # the network, not used in the MPC
        x0 = obs[:, :4]
        ctx, u0, x, u, value = self.diff_mpc(
            x0, p_global=param, p_stagewise=p_stagewise, ctx=ctx
        )
        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.diff_mpc.sensitivity(ctx, field_name="du0_dp_global")

    @property
    def param_space(self) -> gym.Space:
        return self.param_manager.get_param_space()

    def default_param(self, obs) -> np.ndarray:
        return self.param_manager.learnable_parameters_default.cat.full().flatten()  # type:ignore
