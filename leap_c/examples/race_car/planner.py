"""Acados-based MPC planner for the Frenet-frame race-car environment.

Mirrors the ``AcadosPlanner`` pattern (cartpole) but overrides ``forward`` to write a per-stage
``s_ref`` non-learnable parameter at each solve, derived from the observation's current arc-length
and the configured lookahead. Action is returned in the OCP's native units - no normalization -
so it matches ``RaceCarEnv.action_space`` directly.

See ``acados_ocp.py`` for the full OCP problem statement (state, cost, constraints) and
``env.py`` for the observation / action space documentation.

References:
----------
- Reiter, R., Nurkanović, A., Frey, J., Diehl, M. (2023).
  "Frenet-Cartesian model representations for automotive obstacle avoidance
  within nonlinear MPC."
  European Journal of Control, Vol. 74, 100847.
  Preprint: https://arxiv.org/abs/2212.13115
  Published: https://www.sciencedirect.com/science/article/pii/S0947358023000766
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from leap_c.examples.race_car.acados_ocp import (
    RaceCarCostType,
    RaceCarParamInterface,
    create_race_car_params,
    export_parametric_ocp,
)
from leap_c.examples.race_car.bicycle_model import DEFAULT_TRACK_FILE
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch


@dataclass(kw_only=True)
class RaceCarPlannerConfig:
    """Configuration for the race-car planner.

    Attributes:
        N_horizon: Number of MPC shooting intervals.
        T_horizon: Horizon duration [s] (so one step is ``T_horizon / N_horizon``).
        sref_lookahead: How far ahead (in arc length) the terminal ``s_ref`` is set.
            Mirrors ``sref_N`` in the acados example (default 3 m).
        track_file: Track reference file used to build the curvature spline.
        cost_type: ``"NONLINEAR_LS"`` (Gauss-Newton) or ``"EXTERNAL"`` (exact Hessian).
        param_interface: ``"global"`` (one set of cost weights) or ``"stagewise"`` (per stage).
        discount_factor: Discount along the MPC horizon. ``None`` defers to acados defaults.
        n_batch_init: Initially supported batch size for the batch OCP solver.
        num_threads_batch_solver: Threads for the batch solver.
        dtype: Output tensor dtype. Uses PyTorch default if ``None``.
    """

    N_horizon: int = 50
    T_horizon: float = 1.0
    sref_lookahead: float = 3.0
    track_file: Path = field(default_factory=lambda: DEFAULT_TRACK_FILE)
    cost_type: RaceCarCostType = "NONLINEAR_LS"
    param_interface: RaceCarParamInterface = "global"
    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None


class RaceCarPlanner(AcadosPlanner[AcadosDiffMpcCtx]):
    """MPC planner: 6-state Frenet bicycle, tracks centerline with a moving ``s_ref`` target."""

    cfg: RaceCarPlannerConfig

    def __init__(
        self,
        cfg: RaceCarPlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ) -> None:
        self.cfg = RaceCarPlannerConfig() if cfg is None else cfg
        params = (
            create_race_car_params(
                param_interface=self.cfg.param_interface,
                N_horizon=self.cfg.N_horizon,
                T_horizon=self.cfg.T_horizon,
            )
            if params is None
            else params
        )
        param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)
        ocp = export_parametric_ocp(
            param_manager=param_manager,
            track_file=self.cfg.track_file,
            cost_type=self.cfg.cost_type,
            name="race_car",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
        )
        diff_mpc = AcadosDiffMpcTorch(
            ocp,
            discount_factor=self.cfg.discount_factor,
            export_directory=export_directory,
            n_batch_init=self.cfg.n_batch_init,
            num_threads_batch_solver=self.cfg.num_threads_batch_solver,
            dtype=self.cfg.dtype,
        )
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        param: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the MPC and return the first applied control.

        A per-stage arc-length reference is constructed from the current observation
        and the configured lookahead:

            s_ref[j] = s0 + sref_lookahead * j / N_horizon,    j = 0, ..., N_horizon

        where ``s0`` is the observed arc length (``obs[:, 0]``). This reproduces the
        moving ``sref_N`` target from the acados race_cars example at each solve.

        Args:
            obs: ``(batch_size, 6)`` Frenet state ``[s, n, alpha, v, D, delta]``.
            action: Optional warm-start action.
            param: Optional learnable parameter override (defaults filled in if ``None``).
            ctx: Previous solver context for warm-starting.

        Returns:
            Tuple ``(ctx, u0, x_plan, u_plan, value)`` as produced by ``AcadosDiffMpcTorch``.
            ``u0`` has shape ``(batch_size, 2)`` in native units ``[derD, derDelta]`` and
            matches ``RaceCarEnv.action_space`` directly (no normalization).
        """
        x0 = obs
        batch_size = x0.shape[0]
        N = self.cfg.N_horizon

        s0 = x0[:, 0:1].detach().cpu().numpy()  # (B, 1)
        j = np.arange(N + 1, dtype=np.float64).reshape(1, N + 1, 1)
        s_ref_staged = s0[:, np.newaxis, :] + self.cfg.sref_lookahead * j / N  # (B, N+1, 1)

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(s_ref=s_ref_staged)

        if param is None and self.param_manager.learnable_parameters.size > 0:
            default_flat = torch.from_numpy(
                self.param_manager.learnable_parameters_default.cat.full().flatten()
            ).to(x0.device)
            param = default_flat.unsqueeze(0).expand(batch_size, -1).contiguous()

        return self.diff_mpc(x0, action, param, p_stagewise, ctx=ctx)
