from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from leap_c.examples.mass_spring_damper.acados_ocp import (
    export_parametric_ocp,
)
from leap_c.ocp.acados.diff_mpc import collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.planner import acados_sensitivity
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch
from leap_c.planner import ParameterizedPlanner, SensitivityOptions


@dataclass(kw_only=True)
class MassSpringDamperPlannerConfig:
    """Configuration for the MassSpringDamper planner.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
            The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal
            node N).
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

    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None


class MassSpringDamperPlanner(ParameterizedPlanner[AcadosDiffMpcCtx]):
    """Acados-based planner for the mass-spring-damper system.

    The state corresponds to [position, velocity] and the action is [force].
    The cost function is a quadratic cost with Q and R matrices, plus a
    terminal cost matrix P. The dynamics are discrete-time double integrator.

    Attributes:
        cfg: A configuration object containing high-level settings for the MPC problem,
            such as horizon length.
    """

    cfg: MassSpringDamperPlannerConfig
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: MassSpringDamperPlannerConfig | None = None,
        export_directory: Path | None = None,
    ):
        """Initializes the MassSpringDamperPlanner.

        Args:
            cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length. If not provided,
                a default config is used.
            export_directory: An optional directory path where the generated
                `acados` solver code will be exported.
        """
        self.cfg = MassSpringDamperPlannerConfig() if cfg is None else cfg
        super().__init__()

        ocp, param_manager, param_space, default_param = export_parametric_ocp(
            N_horizon=self.cfg.N_horizon,
            name="mass_spring_damper",
            x0=np.array([1.0, 0.0]),
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
    ):
        return self.diff_mpc(x0=obs, u0=action, params=params, ctx=ctx)

    def sensitivity(self, ctx: AcadosDiffMpcCtx, name: SensitivityOptions) -> np.ndarray:
        return acados_sensitivity(self.diff_mpc, ctx, name)

    @property
    def param_space(self):
        return self._param_space

    def default_param(self, obs: np.ndarray | torch.Tensor | None = None) -> dict[str, np.ndarray]:
        # Broadcast each parameter's per-stage default to the batch shape implied
        # by obs, e.g. obs (B, obs_dim) -> default[key] (B, *param_shape).
        # Without obs the unbatched defaults are returned.
        default = {key: np.asarray(value) for key, value in self._default_param.items()}
        for key in default:
            if obs is not None and obs.ndim > 1:
                default[key] = np.broadcast_to(default[key], (*obs.shape[:-1], *default[key].shape))
        return default
