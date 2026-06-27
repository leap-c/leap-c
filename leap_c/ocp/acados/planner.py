from typing import Generic, get_args

import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import CtxType
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcSensitivityOptions, AcadosDiffMpcTorch
from leap_c.planner import ParameterizedPlanner, SensitivityOptions

# TODO Jasper: Needs to be updated if we go to a dictionary based sensitivity derivation.
TO_ACADOS_DIFFMPC_SENSOPTS: dict[SensitivityOptions, AcadosDiffMpcSensitivityOptions] = {
    "du0_dp": "du0_dp_global",
    "dx_dp": "dx_dp_global",
    "du_dp": "du_dp_global",
    "dvalue_dp": "dvalue_dp_global",
    "dvalue_daction": "dvalue_du0",
    "du0_dx0": "du0_dx0",
    "dvalue_dx0": "dvalue_dx0",
}


class AcadosPlanner(ParameterizedPlanner[CtxType], Generic[CtxType]):
    """acados-based MPC planner.

    This class wraps `AcadosDiffMpcTorch`, delegating all parameter-handling and
    solving functionality directly to it.
    """

    param_manager: AcadosParameterManager
    diff_mpc: AcadosDiffMpcTorch

    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(self, param_manager: AcadosParameterManager, diff_mpc: AcadosDiffMpcTorch):
        super().__init__()
        self.param_manager = param_manager
        self.diff_mpc = diff_mpc

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        params: dict[str, torch.Tensor | np.ndarray] | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.diff_mpc(x0=obs, u0=action, params=params, ctx=ctx)

    def sensitivity(self, ctx: CtxType, name: SensitivityOptions) -> np.ndarray:
        if name not in TO_ACADOS_DIFFMPC_SENSOPTS:
            raise ValueError(
                f"Unknown sensitivity option `{name}`; available options: "
                + ", ".join(get_args(SensitivityOptions))
            )
        return self.diff_mpc.sensitivity(ctx, TO_ACADOS_DIFFMPC_SENSOPTS[name])

    @property
    def param_space(self) -> gym.Space:
        return self.diff_mpc.param_space

    def default_param(self, obs: np.ndarray) -> np.ndarray:
        return self.diff_mpc.default_param(obs)
