"""Parameter manager tutorial - basic usage with default non-differentiable parameters.

Demonstrates:
- Building an AcadosParameterManager via ``register_parameter``
- ``combine_non_differentiable_parameters``: packing per-stage non-differentiable values
- ``combine_differentiable_parameters_torch``: packing a batch of differentiable values
  with a stage-varying price overwrite
- Running ``AcadosDiffMpcTorch.forward`` with the default p_stagewise (outdoor_temp
  at its default value for every stage)
"""

import numpy as np
import torch
from utils import BATCH_SIZE, N_HORIZON, build_manager, build_ocp

from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)

    manager = build_manager(N_HORIZON)
    ocp = build_ocp(manager, N_HORIZON)
    diff_mpc = AcadosDiffMpcTorch(ocp, manager)

    N_stages = N_HORIZON + 1

    # ── Illustrate combine_non_differentiable_parameters ────────────────────
    temp_forecast = rng.uniform(5.0, 25.0, size=(BATCH_SIZE, N_stages, 1))
    p_stagewise = manager.combine_non_differentiable_parameters(
        batch_size=BATCH_SIZE,
        outdoor_temp=temp_forecast,
    )
    print(f"p_stagewise shape: {p_stagewise.shape}")

    # ── Illustrate combine_differentiable_parameters_torch ───────────────────
    price_forecast = rng.uniform(0.05, 0.40, size=(BATCH_SIZE, N_stages, 1))
    p_global = manager.combine_differentiable_parameters_torch(
        batch_size=BATCH_SIZE,
        device=torch.device("cpu"),
        dtype=torch.float64,
        price=torch.as_tensor(price_forecast, dtype=torch.float64),
    )
    print(f"p_global shape:    {p_global.shape}")

    # ── Run the solver ────────────────────────────────────────────────────────
    x0_batch = torch.tensor(rng.uniform(15.0, 25.0, size=(BATCH_SIZE, 1)), dtype=torch.float64)
    p_global_default = manager.combine_differentiable_parameters_torch(
        batch_size=BATCH_SIZE, device=torch.device("cpu"), dtype=torch.float64
    )

    ctx, u0, x, u, value = diff_mpc(x0=x0_batch, params={})

    print(f"ctx.status: {ctx.status}")
    print(f"u0.shape:   {u0.shape}")
    print(f"value:      {value.squeeze().tolist()}")
