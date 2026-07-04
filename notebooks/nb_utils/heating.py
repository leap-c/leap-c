"""The R1C1 room-heating OCPs used throughout the tutorial notebooks.

A single thermal resistance R to the outdoors and a single thermal
capacitance C for the room (the model from the parameter-management guide in
``docs/source/parameter_management.md``):

    T_next = T + dt * ((outdoor_temp - T) / (R * C) + q / C)

NOTE: ``build_heating_ocp`` is a synced copy of the inline builder taught in
``notebooks/getting_started/02_from_acados_to_diff_mpc.py``.
``build_heating_ocp_comfort_band`` lives only here; notebook 06 shows the
lines it adds as a static excerpt.
"""

import casadi as ca
import numpy as np
from acados_template import ACADOS_INFTY, AcadosOcp

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

    Registered parameters: ``outdoor_temp`` (non-differentiable, stagewise at
    runtime), ``R`` (differentiable — the insulation question),
    ``comfort_setpoint`` (differentiable) and ``price`` (differentiable, stage
    structure set by ``price_splits``).

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
    # Envelope quality: differentiable, so the solver can answer
    # "would better insulation (a larger R) pay off?". Omitting `splits`
    # means splits="global": one value shared by all stages.
    R = manager.register_parameter(
        name="R", default=np.array([R_THERMAL]), differentiable=True
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
        (outdoor_temp - T) / (R * C_THERMAL) + q / C_THERMAL
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
    """The nominal R1C1 update — identical to the model inside the OCP."""
    return T + dt * ((outdoor_temp - T) / (R_THERMAL * C_THERMAL) + q / C_THERMAL)


def build_heating_ocp_comfort_band(
    N_horizon: int,
    dt: float = 0.25,
    q_max: float = 12.0,
    eps_reg: float = 1e-3,
    slack_weight: float = 1e2,
    name: str = "heating_band",
) -> tuple[AcadosOcp, AcadosParameterManager]:
    """Build the comfort-band heating OCP used by the planner notebooks (06-08).

    Instead of tracking a setpoint, the room temperature must stay inside a
    *time-varying* comfort band ``[t_lower + comfort_margin, t_upper]`` while
    the pure energy cost ``price_weight * price * q`` is minimized. Because
    leap-c writes state bounds only at stage 0 on each solve, the band is
    encoded as a slacked nonlinear constraint ``h(x, p) >= 0`` whose bounds
    are stagewise *parameters* — this is the supported way to get
    time-varying, softly-enforced state bounds.

    Registered parameters:
        - ``outdoor_temp``, ``price``, ``t_lower``, ``t_upper``:
          non-differentiable, passed per solve as ``(B, N+1, 1)`` windows.
        - ``R`` (the planner's belief about the envelope), ``price_weight``
          (how much the occupant cares about cost) and ``comfort_margin``
          (how far above the scheduled lower bound they actually live):
          differentiable scalars — the learnable knobs of the imitation- and
          reinforcement-learning notebooks.

    The quadratic regularization ``eps_reg * q**2`` keeps the Hessian positive
    definite (the economic cost alone is linear in ``q``). The slack weight is
    a fixed numeric: acados slack penalties (``cost.Zl``) cannot be
    parameters, which is exactly why the learnable comfort knob is a bound
    *margin* rather than a bound *weight*.

    Args:
        N_horizon: Horizon length (number of shooting intervals).
        dt: Time step [h].
        q_max: Upper bound on the heating power [kW].
        eps_reg: Quadratic control regularization weight.
        slack_weight: Quadratic penalty on comfort-band violations.
        name: acados model name (keep distinct per notebook).
    """
    manager = AcadosParameterManager(N_horizon=N_horizon)

    outdoor_temp = manager.register_parameter(
        name="outdoor_temp", default=np.array([10.0]), differentiable=False
    )
    price = manager.register_parameter(
        name="price", default=np.array([0.15]), differentiable=False
    )
    t_lower = manager.register_parameter(
        name="t_lower", default=np.array([17.0]), differentiable=False
    )
    t_upper = manager.register_parameter(
        name="t_upper", default=np.array([21.0]), differentiable=False
    )
    R = manager.register_parameter(
        name="R", default=np.array([R_THERMAL]), differentiable=True
    )
    price_weight = manager.register_parameter(
        name="price_weight", default=np.array([1.0]), differentiable=True
    )
    comfort_margin = manager.register_parameter(
        name="comfort_margin", default=np.array([0.0]), differentiable=True
    )

    ocp = AcadosOcp()
    ocp.model.name = name

    T = ca.SX.sym("T")  # room temperature [degC]
    q = ca.SX.sym("q")  # heating power [kW]
    ocp.model.x = T
    ocp.model.u = q

    ocp.model.disc_dyn_expr = T + dt * (
        (outdoor_temp - T) / (R * C_THERMAL) + q / C_THERMAL
    )

    # Pure economic stage cost; eps_reg makes it strictly convex in q.
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = price_weight * price * q + eps_reg * q**2

    # Comfort band as a slacked two-row inequality, h(x, p) >= 0:
    #   T >= t_lower + comfort_margin   and   T <= t_upper.
    band_expr = ca.vertcat(T - t_lower - comfort_margin, t_upper - T)
    # acados applies con_h_expr at stages 1..N-1; stage 0 gets no
    # h-constraint (x0 is fixed anyway; con_h_expr_0 would add one).
    ocp.model.con_h_expr = band_expr
    ocp.constraints.lh = np.zeros(2)
    ocp.constraints.uh = np.full(2, ACADOS_INFTY)
    ocp.constraints.idxsh = np.array([0, 1])
    ocp.cost.Zl = np.full(2, slack_weight)
    ocp.cost.Zu = np.full(2, slack_weight)
    ocp.cost.zl = np.zeros(2)
    ocp.cost.zu = np.zeros(2)

    # The same band at the terminal stage (mirroring _e is mandatory).
    ocp.model.con_h_expr_e = band_expr
    ocp.constraints.lh_e = np.zeros(2)
    ocp.constraints.uh_e = np.full(2, ACADOS_INFTY)
    ocp.constraints.idxsh_e = np.array([0, 1])
    ocp.cost.Zl_e = np.full(2, slack_weight)
    ocp.cost.Zu_e = np.full(2, slack_weight)
    ocp.cost.zl_e = np.zeros(2)
    ocp.cost.zu_e = np.zeros(2)

    ocp.constraints.x0 = np.array([20.0])

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
