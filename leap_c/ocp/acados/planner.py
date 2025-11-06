import gymnasium as gym
import numpy as np
import torch

from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch
from leap_c.planner import ParameterizedPlanner, SensitivityOptions


class AcadosPlanner(ParameterizedPlanner):
    """acados based MPC planner

    This providing a simple standard implementation of the functionalities needed in the
    `ParameterizedPlanner` interface. It wraps the AcadosDiffMpcTorch module, handling
    how to process parameters and observations.

    Attributes:
        param_manager: For managing the parameters of the ocp.
        diff_mpc: An object wrapping the acados ocp solver for differentiable MPC solving.
        collate_fn_map: A mapping for collating AcadosDiffMpcCtx objects in batches.
    """

    param_manager: AcadosParameterManager
    diff_mpc: AcadosDiffMpcTorch

    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(self, param_manager: AcadosParameterManager, diff_mpc: AcadosDiffMpcTorch):
        super().__init__()
        self.param_manager = param_manager
        self.diff_mpc = diff_mpc

    def forward(
        self, obs: torch.Tensor, action=None, param=None, ctx: AcadosDiffMpcCtx | None = None
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Passes obs, param and ctx, as-is to the AcadosDiffMpcTorch object.

        This can be subclassed if observations or parameters need to be passed differently.

        Note that param is assumed to be the learnable parameters only, while the
        non-learnable parameters are automatically obtained from the param_manager.
        """
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=obs.shape[0]
        )
        return self.diff_mpc(obs, action=action, p_global=param, p_stagewise=p_stagewise, ctx=ctx)

    def sensitivity(self, ctx: AcadosDiffMpcCtx, name: SensitivityOptions) -> np.ndarray:
        if name == "du0_dp":
            return self.diff_mpc.sensitivity(ctx, "du0_dp_global")
        elif name == "dx_dp":
            return self.diff_mpc.sensitivity(ctx, "dx_dp_global")
        elif name == "du_dp":
            return self.diff_mpc.sensitivity(ctx, "du_dp_global")
        elif name == "dvalue_dp":
            return self.diff_mpc.sensitivity(ctx, "dvalue_dp_global")
        elif name == "dvalue_daction":
            return self.diff_mpc.sensitivity(ctx, "dvalue_du0")
        elif name == "du0_dx0":
            return self.diff_mpc.sensitivity(ctx, "du0_dx0")
        elif name == "dvalue_dx0":
            return self.diff_mpc.sensitivity(ctx, "dvalue_dx0")
        raise ValueError(f"Unknown sensitivity option: {name}")

    @property
    def param_space(self) -> gym.Space:
        return self.param_manager.get_param_space()

    def default_param(self, obs) -> np.ndarray:
        return self.param_manager.learnable_parameters_default.cat.full().flatten()  # type:ignore
