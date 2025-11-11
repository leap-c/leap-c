from typing import Generic, get_args

import gymnasium as gym
import torch
from numpy import ndarray

from leap_c.controller import CtxType
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcSensitivityOptions, AcadosDiffMpcTorch
from leap_c.planner import ParameterizedPlanner, SensitivityOptions

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

    This class provides a simple standard implementation of the functionalities needed in the
    `ParameterizedPlanner` interface. It wraps the `AcadosDiffMpcTorch` module, handling how to
    process parameters and observations.

    Attributes:
        param_manager: For managing the parameters of the ocp.
        diff_mpc: An object wrapping the acados ocp solver for differentiable MPC solving.
        collate_fn_map: A mapping for collating `AcadosDiffMpcCtx` objects in batches.
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
        param: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Passes `obs`, `action`, `param` and `ctx`, as-is to the `AcadosDiffMpcTorch` object.

        This can be subclassed if observations, actions or parameters need to be passed differently.

        Note that `param` is assumed to be the learnable parameters only, while the non-learnable
        parameters are automatically obtained from the `param_manager`.
        """
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(obs.shape[0])
        return self.diff_mpc(obs, action, param, p_stagewise, ctx=ctx)

    def sensitivity(self, ctx: CtxType, name: SensitivityOptions) -> ndarray:
        if name not in TO_ACADOS_DIFFMPC_SENSOPTS:
            raise ValueError(
                f"Unknown sensitivity option `{name}`; available options: "
                + ", ".join(get_args(SensitivityOptions))
            )
        return self.diff_mpc.sensitivity(ctx, TO_ACADOS_DIFFMPC_SENSOPTS[name])

    @property
    def param_space(self) -> gym.Space:
        return self.param_manager.get_param_space()

    def default_param(self, obs: ndarray | None) -> ndarray:
        return self.param_manager.learnable_parameters_default.cat.full().flatten()  # type:ignore
