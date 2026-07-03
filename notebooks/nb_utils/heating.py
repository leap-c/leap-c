"""The R1C1 room-heating OCP used by notebooks 04 and 05.

A single thermal resistance R to the outdoors and a single thermal
capacitance C for the room (the model from the parameter-management guide in
``docs/source/parameter_management.md``):

    T_next = T + dt * ((outdoor_temp - T) / (R * C) + q / C)

NOTE: keep in sync with the inline builder taught in
``notebooks/04_heating_parameter_management.py`` (pedagogical inline copy).
"""

import casadi as ca
import numpy as np
from acados_template import AcadosOcp

from leap_c.parameters import AcadosParameterManager
from leap_c.parameters.utils import ParamSplits

R_THERMAL = 2.0  # thermal resistance to the outdoors [K/kW]
C_THERMAL = 1.5  # thermal capacitance of the room [kWh/K]


def build_heating_ocp(
    N_horizon: int,
    dt: float = 0.25,
    price_splits: ParamSplits = "global",
    q_max: float = 8.0,
    name: str = "heating",
) -> tuple[AcadosOcp, AcadosParameterManager]:
    """Build the parametric heating OCP and its parameter manager.

    Args:
        N_horizon: Horizon length (number of shooting intervals).
        dt: Time step [h].
        price_splits: How the electricity price varies across stages —
            ``"global"`` (one price for the whole horizon), a list of segment
            end stages, or ``"stagewise"`` (one price per stage).
        q_max: Upper bound on the heating power [kW].
        name: acados model name. Use distinct names when building several
            instances in one session so their generated code does not collide.

    Always builds the OCP and the manager together, fresh: a manager is
    finalized by ``AcadosDiffMpcTorch`` (via ``assign_to_ocp``) and must not
    be reused for a second OCP.
    """
    manager = AcadosParameterManager(N_horizon=N_horizon)

    # Weather forecast: changeable per stage at runtime, but no gradients.
    outdoor_temp = manager.register_parameter(
        name="outdoor_temp", default=np.array([10.0]), differentiable=False
    )
    # Comfort reference: one differentiable value shared by all stages.
    comfort_setpoint = manager.register_parameter(
        name="comfort_setpoint", default=np.array([21.0]), differentiable=True
    )
    # Electricity price: differentiable, stage structure set by `price_splits`.
    price = manager.register_parameter(
        name="price", default=np.array([0.15]), differentiable=True, splits=price_splits
    )

    ocp = AcadosOcp()
    ocp.model.name = name

    T = ca.SX.sym("T")  # room temperature [degC]
    q = ca.SX.sym("q")  # heating power [kW]
    ocp.model.x = T
    ocp.model.u = q

    # Discrete-time R1C1 dynamics: heat leaks to the outdoors through R,
    # the heater injects q into the capacitance C.
    ocp.model.disc_dyn_expr = T + dt * (
        (outdoor_temp - T) / (R_THERMAL * C_THERMAL) + q / C_THERMAL
    )

    # Comfort deviation plus price-weighted energy; terminal cost comfort only.
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (T - comfort_setpoint) ** 2 + price * q
    ocp.model.cost_expr_ext_cost_e = (T - comfort_setpoint) ** 2

    # Initial state — a nominal value, overwritten on every solve.
    ocp.constraints.x0 = np.array([20.0])

    # The heater can only heat, up to q_max.
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([q_max])

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon * dt
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp, manager


def step_room(T: float, q: float, outdoor_temp: float, dt: float = 0.25) -> float:
    """The true R1C1 update — identical to the model inside the OCP."""
    return T + dt * ((outdoor_temp - T) / (R_THERMAL * C_THERMAL) + q / C_THERMAL)
