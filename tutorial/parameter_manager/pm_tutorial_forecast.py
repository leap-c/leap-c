"""Parameter manager tutorial - forecast-aware planner with non-default learnable params.

Demonstrates:
- Subclassing AcadosPlanner and overriding forward to inject a per-call outdoor
  temperature forecast via combine_non_learnable_parameter_values
- Passing non-default learnable parameter values (custom comfort setpoint and
  stage-varying price) via combine_default_learnable_parameter_values
- Passing the observation as a dict with keys "state" and "outdoor_temp_forecast"

Observation format expected by TempCtrlPlanner.forward:
    obs = {
        "state":                 torch.Tensor  shape (batch_size, 1),   # room temperature
        "outdoor_temp_forecast": np.ndarray    shape (batch_size, N_stages, 1),
    }
"""

from typing import Any

import numpy as np
import torch
from numpy import ndarray
from utils import BATCH_SIZE, N_HORIZON, build_ocp, make_params

from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

# ── Custom planner ────────────────────────────────────────────────────────────


class TempCtrlPlanner(AcadosPlanner):
    """Temperature-control planner that reads outdoor_temp from the observation dict.

    Overrides :meth:`forward` to:
    1. Extract the per-call outdoor temperature forecast from ``obs``.
    2. Build the per-stage non-learnable parameter array via
       :meth:`combine_non_learnable_parameter_values`, injecting the forecast.
    3. Pass everything to the underlying :class:`AcadosDiffMpcTorch` solver.
    """

    def forward(
        self,
        obs: dict[str, Any],
        action: torch.Tensor | None = None,
        param: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the MPC problem for a batch of initial conditions and forecasts.

        Args:
            obs: Dict with keys:
                - ``"state"``: initial room temperature, shape ``(batch_size, 1)``.
                - ``"outdoor_temp_forecast"``: ambient temperature forecast,
                  shape ``(batch_size, N_stages, 1)``.
            action: Optional warm-start action (not used here).
            param: Learnable parameter tensor, shape ``(batch_size, N_learnable)``.
                If ``None``, defaults from the parameter manager are used.
            ctx: Optional context from a previous call for warm-starting.

        Returns:
            Tuple ``(ctx, u0, x, u, value)`` as returned by
            :class:`AcadosDiffMpcTorch`.
        """
        state = obs["state"]
        temp_forecast = obs["outdoor_temp_forecast"]  # (batch_size, N_stages, 1)
        batch_size = state.shape[0]

        # Build per-stage non-learnable params with the actual forecast injected.
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=batch_size,
            outdoor_temp=temp_forecast,
        )

        return self.diff_mpc(state, action, param, p_stagewise, ctx=ctx)

    def default_param(self, obs: ndarray | None) -> ndarray:
        return self.param_manager.learnable_parameters_default.cat.full().flatten()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    N_stages = N_HORIZON + 1  # 11 stages (0 .. 10)

    # ── Build manager, OCP, and custom planner ────────────────────────────────
    manager = AcadosParameterManager(make_params(N_HORIZON), N_horizon=N_HORIZON)
    ocp = build_ocp(manager, N_HORIZON)
    diff_mpc = AcadosDiffMpcTorch(ocp)
    planner = TempCtrlPlanner(param_manager=manager, diff_mpc=diff_mpc)

    # ── Build observation dict ────────────────────────────────────────────────
    state = torch.tensor(rng.uniform(15.0, 25.0, size=(BATCH_SIZE, 1)))

    # Outdoor temperature forecast provided per-call as part of the observation.
    # Shape: (batch_size, N_stages, 1)
    temp_forecast = rng.uniform(5.0, 25.0, size=(BATCH_SIZE, N_stages, 1))

    obs = {
        "state": state,
        "outdoor_temp_forecast": temp_forecast,
    }

    # ── Non-default learnable parameters ─────────────────────────────────────
    # comfort_setpoint: constant across stages, shape (batch_size, 1)
    # Each batch element gets a different setpoint preference.
    comfort_values = np.array([[19.0], [21.0], [23.0], [22.5]])  # (batch_size, 1)

    # price: stage-varying (two blocks), shape (batch_size, N_stages, 1).
    # The manager picks stage 0 for block 0 (stages 0-4) and stage 5 for block 1 (stages 5-10).
    price_forecast = rng.uniform(0.05, 0.40, size=(BATCH_SIZE, N_stages, 1))

    # combine_default_learnable_parameter_values starts from the default values
    # and replaces the entries for the named parameters with the provided arrays.
    param = torch.tensor(
        manager.combine_default_learnable_parameter_values(
            comfort_setpoint=comfort_values,
            price=price_forecast,
        )
    )
    print(f"param shape: {param.shape}")  # (4, N_learnable)

    # ── Solve ─────────────────────────────────────────────────────────────────
    ctx, u0, x, u, value = planner.forward(obs=obs, param=param)

    print(f"ctx.status:     {ctx.status}")  # [0 0 0 0] means all solves succeeded
    print(f"u0.shape:       {u0.shape}")
    print(f"value:          {value.squeeze().tolist()}")
    print(f"comfort_values: {comfort_values.squeeze().tolist()}")
    print(f"first control:  {u0.squeeze().tolist()}")
