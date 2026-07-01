"""Parameter manager tutorial - forecast-aware solve with non-default learnable params.

Demonstrates:
- Using ``AcadosDiffMpcTorch`` directly to solve with per-call outdoor temperature
  forecast via ``combine_non_learnable_parameters``
- Passing non-default learnable parameter values (custom comfort setpoint and
  stage-varying price) via ``combine_learnable_parameters_torch``
"""

import numpy as np
import torch
from utils import BATCH_SIZE, N_HORIZON, build_manager, build_ocp

from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    N_stages = N_HORIZON + 1

    manager = build_manager(N_HORIZON)
    ocp = build_ocp(manager, N_HORIZON)
    diff_mpc = AcadosDiffMpcTorch(ocp, manager)

    # ── Build initial state and outdoor temperature forecast ─────────────────
    state = torch.tensor(rng.uniform(15.0, 25.0, size=(BATCH_SIZE, 1)), dtype=torch.float64)
    temp_forecast = rng.uniform(5.0, 25.0, size=(BATCH_SIZE, N_stages, 1))

    # ── Non-default learnable parameters ─────────────────────────────────────
    comfort_values = torch.tensor(np.array([[19.0], [21.0], [23.0], [22.5]]), dtype=torch.float64)
    price_forecast = torch.tensor(
        rng.uniform(0.05, 0.40, size=(BATCH_SIZE, N_stages, 1)), dtype=torch.float64
    )

    p_global = manager.combine_learnable_parameters_torch(
        batch_size=BATCH_SIZE,
        device=torch.device("cpu"),
        dtype=torch.float64,
        comfort_setpoint=comfort_values,
        price=price_forecast,
    )
    print(f"p_global shape: {p_global.shape}")

    # ── Solve ─────────────────────────────────────────────────────────────────
    ctx, u0, x, u, value = diff_mpc(
        x0=state,
        params={
            "outdoor_temp": temp_forecast,
            "comfort_setpoint": comfort_values,
            "price": price_forecast,
        },
    )

    print(f"ctx.status:     {ctx.status}")
    print(f"u0.shape:       {u0.shape}")
    print(f"value:          {value.squeeze().tolist()}")
    print(f"comfort_values: {comfort_values.squeeze().tolist()}")
    print(f"first control:  {u0.squeeze().tolist()}")
