from typing import Any, Callable

import gymnasium as gym
import numpy as np

from leap_c.controller import ParameterizedController
from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpc


class AcadosController(ParameterizedController):
    """Acados-based controller, providing a simple standard implementation of
    most of the functionalities needed in the `ParameterizedController` interface.

    Attributes:
        param_manager: For managing the parameters of the ocp.
        diff_mpc: An object wrapping the acados ocp solver for differentiable MPC solving.
        collate_fn_map: A mapping for collating AcadosDiffMpcCtx objects in batches.
    """

    param_manager: AcadosParameterManager
    diff_mpc: AcadosDiffMpc

    collate_fn_map: dict[type, Callable] = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def forward(self, obs, param, ctx=None) -> tuple[Any, np.ndarray]:
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=obs.shape[0]
        )
        ctx, u0, x, u, value = self.diff_mpc(obs, p_global=param, p_stagewise=p_stagewise, ctx=ctx)
        return ctx, u0

    def jacobian_action_param(self, ctx) -> np.ndarray:
        return self.diff_mpc.sensitivity(ctx, field_name="du0_dp_global")

    @property
    def param_space(self) -> gym.Space:
        return self.param_manager.get_param_space(dtype=np.float32)

    def default_param(self, obs) -> np.ndarray:
        return self.param_manager.learnable_parameters_default.cat.full().flatten()  # type:ignore
