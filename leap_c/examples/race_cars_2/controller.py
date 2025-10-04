from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.examples.race_cars_2.acados_ocp import (
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

    cost_type: RaceAcadosCostType = "EXTERNAL"
    param_interface: RaceAcadosParamInterface = "global"

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
                cost_type=self.cfg.cost_type,
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

        def forward(self, obs, params, ctx=None) -> tuple[Any, torch.Tensor]:
            p_stagewise = self