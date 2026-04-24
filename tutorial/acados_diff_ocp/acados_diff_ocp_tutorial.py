import casadi as ca
import gymnasium as gym
import numpy as np
import torch

from leap_c.ocp.acados.diff_ocp import AcadosDiffOcp
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

# ── Constants ────────────────────────────────────────────────────────────────

N_HORIZON = 10  # MPC horizon length; stages 0 .. N_HORIZON (inclusive)
BATCH_SIZE = 4  # number of parallel problem instances

# Thermal model constants [RC circuit]
R_THERMAL = 2.0  # thermal resistance [h·degC/kW]
C_THERMAL = 1.5  # thermal capacitance [kWh/degC]

if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)

    # ── Build differentiable OCP, and planner ──────────────────────────────────────
    ocp = AcadosDiffOcp(N_horizon=N_HORIZON)

    outdoor_temp = ocp.register_param(
        name="outdoor_temp",
        default=np.array([20.0]),
        differentiable=False,
    )
    comfort_ref = ocp.register_param(
        name="comfort_setpoint",
        default=np.array([21.0]),
        space=gym.spaces.Box(low=np.array([15.0]), high=np.array([28.0]), dtype=np.float64),
        differentiable=True,
    )
    price = ocp.register_param(
        name="price",
        default=np.array([0.15]),
        space=gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float64),
        differentiable=True,
        end_stages=[4, N_HORIZON],
    )
    dt = 0.25  # 15-minute time step [h]

    ocp.model.name = "temp_ctrl"

    # State: room temperature [degC]
    T = ca.SX.sym("T")
    ocp.model.x = T

    # Control: heating power [kW]
    q = ca.SX.sym("q")
    ocp.model.u = q

    # Discrete-time RC dynamics
    ocp.model.disc_dyn_expr = T + dt * (
        (outdoor_temp - T) / (R_THERMAL * C_THERMAL) + q / C_THERMAL
    )

    # Costs
    ocp.model.cost_expr_ext_cost = (T - comfort_ref) ** 2 + price * q
    ocp.model.cost_expr_ext_cost_e = (T - comfort_ref) ** 2

    ocp.constraints.x0 = np.array([20.0])

    ocp.solver_options.tf = N_HORIZON * dt
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "DISCRETE"

    diff_mpc = AcadosDiffMpcTorch(ocp)

    # ── Run the planner ───────────────────────────────────────────────────────
    # using the default outdoor_temp (20 degC every stage).  To supply a real
    # forecast at solve time, see pm_tutorial_forecast.py.
    x0_batch = torch.tensor(rng.uniform(15.0, 25.0, size=(BATCH_SIZE, 1)))
    p_global = torch.tensor(
        ocp.parameter_manager.combine_default_learnable_parameter_values(batch_size=BATCH_SIZE)
    )

    ctx, u0, x, u, value = diff_mpc(x0_batch, p_global=p_global)

    print(f"ctx.status: {ctx.status}")  # [0 0 0 0] means all solves succeeded
    print(f"u0.shape:   {u0.shape}")
    print(f"value:      {value.squeeze().tolist()}")
