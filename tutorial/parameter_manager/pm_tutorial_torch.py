"""Parameter manager tutorial - register API with differentiable parameters.

Demonstrates the ``AcadosParameterManager`` and ``AcadosDiffMpcTorch`` API
where parameters are registered with ``manager.register_parameter()``, which
returns a CasADi symbolic for immediate use in the OCP model.

Two parameter types:
  - **Non-learnable** (``differentiable=False``): changed at runtime but no
    gradient flows through them (e.g. weather forecast).
  - **Learnable** (``differentiable=True``): differentiable; gradients from
    ``value.sum().backward()`` flow back to the input tensor.

Stage-varying learnable parameters (e.g. electricity price with two blocks)
use the ``splits`` argument.
"""

import casadi as ca
import gymnasium as gym
import numpy as np
import torch
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import stage_expanded_box, AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch


# ── Constants ────────────────────────────────────────────────────────────────

N_HORIZON = 20  # MPC horizon length; stages 0 .. N_HORIZON (inclusive)
BATCH_SIZE = 4  # number of parallel problem instances

# Thermal model constants [RC circuit]
R_THERMAL = 2.0  # thermal resistance [h·degC/kW]
C_THERMAL = 1.5  # thermal capacitance [kWh/degC]

if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)

    # ── Build parameter manager and register parameters ──────────────────────
    manager = AcadosParameterManager(N_horizon=N_HORIZON)

    # Non-learnable: ambient temperature, changed per call, no gradient.
    outdoor_temp = manager.register_parameter(
        name="outdoor_temp",
        default=np.array([20.0]),
        differentiable=False,
    )
    # The learnable parameter space is built externally as a gym.spaces.Dict, keyed by
    # parameter name and in registration order. Stage-varying boxes are tiled across their
    # stage blocks with stage_expanded_box so flatten_space matches the flat learnable vector.
    spaces: list[tuple[str, gym.spaces.Box]] = []

    # Learnable: comfort setpoint, constant across all stages.
    comfort_ref = manager.register_parameter(
        name="comfort_setpoint",
        default=np.array([21.0]),
        differentiable=True,
    )
    spaces.append(
        (
            "comfort_setpoint",
            gym.spaces.Box(low=np.array([15.0]), high=np.array([28.0]), dtype=np.float64),
        )
    )
    # Learnable and stage-varying: two price blocks.
    # Block 0 (stages 0-3) and block 1 (stages 4-N_HORIZON).
    price = manager.register_parameter(
        name="price",
        default=np.array([0.15]),
        differentiable=True,
        splits=[4, N_HORIZON],
    )
    spaces.append(
        (
            "price",
            stage_expanded_box(
                gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float64),
                [4, N_HORIZON],
                N_HORIZON,
            ),
        )
    )

    dt = 0.25  # 15-minute time step [h]

    # ── Build OCP ────────────────────────────────────────────────────────────
    ocp = AcadosOcp()
    ocp.model.name = "temp_ctrl"

    # State: room temperature [degC]
    T = ca.SX.sym("T")
    ocp.model.x = T

    # Control: heating power [kW]
    q = ca.SX.sym("q")
    ocp.model.u = q

    # Discrete-time RC dynamics with registered parameter symbols
    ocp.model.disc_dyn_expr = T + dt * (
        (outdoor_temp - T) / (R_THERMAL * C_THERMAL) + q / C_THERMAL
    )

    # Costs
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (T - comfort_ref) ** 2 + price * q
    ocp.model.cost_expr_ext_cost_e = (T - comfort_ref) ** 2

    ocp.constraints.x0 = np.array([20.0])

    ocp.constraints.lbu = np.array([-1.0])
    ocp.constraints.ubu = np.array([1.0])
    ocp.constraints.idxbu = np.array([0])

    ocp.solver_options.tf = N_HORIZON * dt
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "DISCRETE"

    # ── Build differentiable MPC planner ─────────────────────────────────────
    diff_mpc = AcadosDiffMpcTorch(ocp, manager, param_space=gym.spaces.Dict(spaces))

    # ── Solve with non-default learnable parameters ─────────────────────────
    x0_batch = torch.tensor(rng.uniform(10.0, 30.0, size=(BATCH_SIZE, 1)))

    # Price forecast: stage-varying, shape (batch_size, N_horizon + 1, 1).
    # With splits=[4, N_HORIZON]:
    #   Block 0 (stages 0-4) reads val[:, 0]
    #   Block 1 (stages 5-20) reads val[:, 5]
    price_array = np.zeros((BATCH_SIZE, N_HORIZON + 1))
    price_array[:, 0] = [0.015, 0.20, 0.25, 0.30]
    price_array[:, 5] = [0.020, 0.25, 0.30, 0.35]
    price_tensor = torch.tensor(price_array, dtype=torch.float64, requires_grad=True)

    ctx, u0, x, u, value = diff_mpc(x0_batch, params={"price": price_tensor})

    # Backpropagate through the solver
    value.sum().backward()

    print(f"ctx.status: {ctx.status}")  # [0 0 0 0] means all solves succeeded
    print(f"u0.shape:   {u0.shape}")
    print(f"value:      {value.squeeze().tolist()}")
    print(f"price.grad: {price_tensor.grad}")
