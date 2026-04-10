"""Shared constants and OCP construction for the parameter manager tutorials."""

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosModel, AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

# ── Constants ────────────────────────────────────────────────────────────────

N_HORIZON = 10  # MPC horizon length; stages 0 .. N_HORIZON (inclusive)
BATCH_SIZE = 4  # number of parallel problem instances

# Thermal model constants [RC circuit]
R_THERMAL = 2.0  # thermal resistance [h·degC/kW]
C_THERMAL = 1.5  # thermal capacitance [kWh/degC]


# ── Parameter list ────────────────────────────────────────────────────────────


def make_params(N_horizon: int = N_HORIZON) -> list[AcadosParameter]:
    """Return the canonical parameter list for the temperature-control tutorial OCP.

    Parameters:
        - ``dt``               fixed time step [h]  (interface: "fix")
        - ``outdoor_temp``     ambient temperature [degC]  (interface: "non-learnable")
        - ``comfort_setpoint`` comfort reference temperature [degC]  (interface: "learnable")
        - ``price``            electricity price [EUR/kWh], two stage blocks
          (interface: "learnable")
    """
    return [
        # Fixed: known constant, baked into the solver at compile time.
        AcadosParameter(
            name="dt",
            default=np.array([0.25]),  # 15-minute time step [h]
            interface="fix",
        ),
        # Non-learnable: changes every solver call (e.g. a weather forecast),
        # but is NOT differentiated through.
        AcadosParameter(
            name="outdoor_temp",
            default=np.array([20.0]),  # ambient temperature [degC]
            interface="non-learnable",
        ),
        # Learnable, constant across stages: a single scalar the learning
        # algorithm can adjust.
        AcadosParameter(
            name="comfort_setpoint",
            default=np.array([21.0]),
            space=gym.spaces.Box(low=np.array([15.0]), high=np.array([28.0]), dtype=np.float64),
            interface="learnable",
        ),
        # Learnable, stage-varying: two price blocks.
        # block 0: stages 0-4, block 1: stages 5-N_horizon
        # The manager handles the indicator mechanism automatically.
        AcadosParameter(
            name="price",
            default=np.array([0.15]),  # electricity price [EUR/kWh]
            space=gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float64),
            interface="learnable",
            end_stages=[4, N_horizon],
        ),
    ]


# ── OCP builder ───────────────────────────────────────────────────────────────


def build_ocp(manager: AcadosParameterManager, N_horizon: int = N_HORIZON) -> AcadosOcp:
    """Build and return the acados OCP for the temperature-control tutorial.

    The OCP has:
    - State: room temperature T [degC]
    - Control: heating power q [kW]
    - Dynamics: discrete-time RC model
    - Stage cost: (T - comfort_setpoint)^2 + price * q
    - Terminal cost: (T - comfort_setpoint)^2

    Args:
        manager: Configured :class:`AcadosParameterManager` for the OCP.
        N_horizon: Horizon length (number of shooting intervals).

    Returns:
        A configured :class:`acados_template.AcadosOcp` ready for solver creation.
    """
    ocp = AcadosOcp()
    model = AcadosModel()
    model.name = "temp_ctrl"

    # State: room temperature [degC]
    T = ca.SX.sym("T")
    model.x = T

    # Control: heating power [kW]
    q = ca.SX.sym("q")
    model.u = q

    # Retrieve parameters.
    # "fix" returns a numpy array; index [0] to get a plain scalar for CasADi arithmetic.
    # All other interfaces return CasADi SX expressions.
    dt = manager.get("dt")[0]  # numpy scalar
    outdoor_temp = manager.get("outdoor_temp")  # CasADi SX, from p (per-stage)
    comfort_ref = manager.get("comfort_setpoint")  # CasADi SX, from p_global
    price = manager.get("price")  # CasADi SX, stage-aware weighted sum

    # Discrete-time RC dynamics
    model.disc_dyn_expr = T + dt * ((outdoor_temp - T) / (R_THERMAL * C_THERMAL) + q / C_THERMAL)

    # Costs
    model.cost_expr_ext_cost = (T - comfort_ref) ** 2 + price * q
    model.cost_expr_ext_cost_e = (T - comfort_ref) ** 2

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model = model
    manager.assign_to_ocp(ocp)  # wires p_global and p into the ocp object

    # Provide a nominal x0 so acados allocates lbx/ubx at stage 0.
    # The actual value is overwritten at each solve call by AcadosDiffMpcTorch.
    ocp.constraints.x0 = np.array([20.0])

    ocp.solver_options.tf = N_horizon * dt
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "DISCRETE"
    return ocp
