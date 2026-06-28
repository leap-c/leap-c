from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from leap_c.examples.pointmass.acados_ocp import (
    PointMassAcadosParamInterface,
    export_parametric_ocp,
)
from leap_c.ocp.acados.diff_mpc import collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.planner import acados_sensitivity
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch
from leap_c.planner import ParameterizedPlanner, SensitivityOptions


@dataclass(kw_only=True)
class PointMassControllerConfig:
    """Configuration for the PointMass controller.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
        T_horizon: The duration of the MPC horizon.
        Fmax: The maximum force that can be applied to the point mass.
        param_interface: Determines the exposed parameter interface of the controller.
        discount_factor: discount factor along the MPC horizon.
            If `None`, it defaults to the behavior of `AcadosOcpOptions.cost_scaling`.
        n_batch_init: Initially supported batch size of the batch OCP solver.
            Using larger batches will trigger a delay for the creation of more solvers.
            If `None`, a default value is used.
        num_threads_batch_solver: Number of parallel threads to use for the batch OCP solver.
            If `None`, a default value is used.
        dtype: Type the planner output tensors will automatically be cast to. If `None`, PyTorch
            default dtype is used.
    """

    N_horizon: int = 20
    T_horizon: float = 2.0
    Fmax: float = 10.0
    param_interface: PointMassAcadosParamInterface = "global"
    x_ref_value: np.ndarray | None = None

    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None


class PointMassPlanner(ParameterizedPlanner[AcadosDiffMpcCtx]):
    """Acados-based controller for the `PointMass` system.

    The state corresponds to the observation of the `PointMass` environment, without the wind force.
    The cost function takes a weighted least-squares form, and the dynamics correspond to the ones
    in the environment, but without the wind force. The inequality constraints are box constraints
    on the action (hard) and on the position of the ball, the latter representing the bounds of the
    world (soft/slacked).

    Attributes:
        cfg: A configuration object containing high-level settings for the MPC problem,
            such as horizon length.
    """

    cfg: PointMassControllerConfig
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: PointMassControllerConfig | None = None,
        export_directory: Path | None = None,
    ) -> None:
        """Initializes the PointMassController.

        Args:
            cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length and maximum force.
                If not provided, a default config is used.
            export_directory: Optional directory for generated acados solver code.
        """
        self.cfg = PointMassControllerConfig() if cfg is None else cfg
        super().__init__()

        ocp, param_manager, param_space, default_param = export_parametric_ocp(
            param_interface=self.cfg.param_interface,
            name="pointmass",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
            x_ref_value=self.cfg.x_ref_value,
        )

        diff_mpc = AcadosDiffMpcTorch(
            ocp,
            param_manager,
            discount_factor=self.cfg.discount_factor,
            export_directory=export_directory,
            n_batch_init=self.cfg.n_batch_init,
            num_threads_batch_solver=self.cfg.num_threads_batch_solver,
            dtype=self.cfg.dtype,
        )
        self.param_manager = param_manager
        self.diff_mpc = diff_mpc
        self._param_space = param_space
        self._default_param = default_param

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        params: dict[str, torch.Tensor | np.ndarray] | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # remove wind field from observation, this is only observed by
        # the network, not used in the MPC
        x0 = obs[:, :4]
        return self.diff_mpc(x0=x0, u0=action, params=params, ctx=ctx)

    def sensitivity(self, ctx: AcadosDiffMpcCtx, name: SensitivityOptions) -> np.ndarray:
        return acados_sensitivity(self.diff_mpc, ctx, name)

    @property
    def param_space(self):
        return self._param_space

    def default_param(self, obs: np.ndarray | torch.Tensor | None = None) -> dict[str, np.ndarray]:
        default = {key: np.asarray(value) for key, value in self._default_param.items()}
        for key in default:
            if obs is not None and obs.ndim > 1:
                default[key] = np.broadcast_to(default[key], (*obs.shape[:-1], *default[key].shape))
        return default
