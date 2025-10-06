from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.race_car.acados_ocp import (
    RaceAcadosCostType,
    RaceAcadosParamInterface,
    create_race_params,
    export_parametric_ocp,
)

from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpc

@dataclass(kw_only=True)
class RaceCarControllerConfig:
    N_horizon: int = 50
    T_horizon: float = 1.0
    track_file: str = "LMS_Track.txt"
    pp_ref_horizon_length: float = 2.0

    cost_type: RaceAcadosCostType = "EXTERNAL"
    param_interface: RaceAcadosParamInterface = "stagewise"

class RaceCarController(ParameterizedController):

    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: RaceCarControllerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        super().__init__()
        self.cfg = RaceCarControllerConfig() if cfg is None else cfg
        params = (
            create_race_params(
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
            name="race_car",
            cost_type=self.cfg.cost_type,
            track_file=self.cfg.track_file,
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
        )

        self.diff_mpc = AcadosDiffMpc(self.ocp, export_directory=export_directory)
        self._ref_horizon_length = self.cfg.pp_ref_horizon_length

    def forward(self, obs, param, ctx=None) -> tuple[Any, torch.Tensor]:
        batch_size = obs.shape[0]

        theta0 = obs[:, 0:1]  # shape: (batch_size, 1)

        # horizon_steps = np.linspace(0.0, 1.0, self.cfg.N_horizon + 1, dtype=np.float64).reshape(1, -1, 1)
        # thetaref = theta0[:, None, :] + horizon_steps * self._ref_horizon_length  # (B, N+1, 1)

        horizon_steps = torch.linspace(0, 1, self.cfg.N_horizon + 1, device=obs.device)
        thetaref = theta0.unsqueeze(1) + horizon_steps.view(1, -1, 1) * self._ref_horizon_length


        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=batch_size,
            # thetaref=thetaref
            thetaref=thetaref.cpu().numpy()
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
