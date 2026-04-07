"""Parameter manager tutorial - basic usage with default non-learnable parameters.

Demonstrates:
- Building an AcadosParameterManager from a parameter list
- combine_non_learnable_parameter_values: packing per-stage non-learnable values
- combine_default_learnable_parameter_values: packing a batch of learnable values
  with a stage-varying price overwrite
- Running AcadosPlanner.forward with the default p_stagewise (outdoor_temp at its
  default value for every stage)
"""

import numpy as np
import torch
from utils import BATCH_SIZE, N_HORIZON, build_ocp, make_params

from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)

    # ── Build manager, OCP, and planner ──────────────────────────────────────
    manager = AcadosParameterManager(make_params(N_HORIZON), N_horizon=N_HORIZON)
    ocp = build_ocp(manager, N_HORIZON)
    diff_mpc = AcadosDiffMpcTorch(ocp)
    planner = AcadosPlanner(param_manager=manager, diff_mpc=diff_mpc)

    N_stages = N_HORIZON + 1  # 11 stages (0 .. 10)

    # ── Illustrate combine_non_learnable_parameter_values ────────────────────
    # Pass an outdoor temperature forecast; shape must be (batch_size, N_stages, param_dim).
    temp_forecast = rng.uniform(5.0, 25.0, size=(BATCH_SIZE, N_stages, 1))
    p_stagewise = manager.combine_non_learnable_parameter_values(
        batch_size=BATCH_SIZE,
        outdoor_temp=temp_forecast,
    )
    # p_stagewise[:, k, :] is the full non-learnable parameter vector at stage k.
    print(f"p_stagewise shape: {p_stagewise.shape}")  # (4, 11, n_nonlearnable)

    # ── Illustrate combine_default_learnable_parameter_values ────────────────
    # Overwrite the stage-varying price with a forecast.
    # For stage-varying params the shape is (batch_size, N_stages, param_dim);
    # the manager picks the value at the start of each block.
    price_forecast = rng.uniform(0.05, 0.40, size=(BATCH_SIZE, N_stages, 1))
    p_global = manager.combine_default_learnable_parameter_values(
        batch_size=BATCH_SIZE,
        price=price_forecast,
    )
    print(f"p_global shape:    {p_global.shape}")     # (4, N_learnable)

    # ── Run the planner ───────────────────────────────────────────────────────
    # AcadosPlanner.forward calls combine_non_learnable_parameter_values internally
    # using the default outdoor_temp (20 degC every stage).  To supply a real
    # forecast at solve time, see pm_tutorial_forecast.py.
    x0_batch = torch.tensor(rng.uniform(15.0, 25.0, size=(BATCH_SIZE, 1)))
    param = torch.tensor(
        manager.combine_default_learnable_parameter_values(batch_size=BATCH_SIZE)
    )

    ctx, u0, x, u, value = planner.forward(obs=x0_batch, param=param)

    print(f"ctx.status: {ctx.status}")   # [0 0 0 0] means all solves succeeded
    print(f"u0.shape:   {u0.shape}")
    print(f"value:      {value.squeeze().tolist()}")
