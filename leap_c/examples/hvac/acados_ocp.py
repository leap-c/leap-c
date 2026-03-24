from dataclasses import asdict
from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp
from scipy.constants import convert_temperature

from leap_c.examples.hvac.dynamics import (
    HydronicDynamicsParameters,
    HydronicParameters,
    transcribe_discrete_state_space,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

HvacAcadosParamInterface = Literal["reference", "reference_dynamics"]
"""Determines the exposed parameter interface of the planner.
- "reference": Only reference parameters are exposed (e.g., ambient temperature,
  solar radiation, electricity price, comfort bounds, etc.).
- "reference_dynamics": Both reference parameters and dynamics parameters are exposed.
"""

HvacAcadosParamGranularity = Literal["global", "stagewise"] | int


def make_default_hvac_params(
    interface: HvacAcadosParamInterface,
    granularity: HvacAcadosParamGranularity,
    N_horizon: int = 96,
    hydronic_params: HydronicParameters | None = None,
) -> tuple[AcadosParameter, ...]:
    """Return a tuple of default parameters for the hvac planner.

    Args:
        interface: The parameter interface type.
        granularity: The granularity of the parameters.
        N_horizon: The number of steps in the MPC horizon
            (default: 96, i.e., 24 hours in 15-minute steps).
        hydronic_params: Optional HydronicParameters object.

    Returns:
        Tuple of AcadosParameter objects for the HVAC system.
    """
    hydronic_params = HydronicParameters() if hydronic_params is None else hydronic_params

    # TODO: The default parameter shifts are currently hardcoded. This will be
    # improved once the environment interface is refactored.
    scaling_shifts = {
        "gAw": 0.8,
        "Rea": 0.8,
        "Rhi": 0.8,
        "Rie": 0.8,
        "Ch": 0.8,
        "Ci": 0.8,
        "Ce": 0.8,
    }
    params = []

    for k, v in asdict(hydronic_params.dynamics).items():
        default_value = scaling_shifts[k] * np.array([v])
        if k in scaling_shifts and "dynamics" in interface:
            param_interface = "learnable"
        else:
            param_interface = "fix"

        params.append(
            AcadosParameter(
                name=k,
                default=default_value,
                space=gym.spaces.Box(
                    low=0.7 * np.array([v]), high=1.3 * np.array([v]), dtype=np.float64
                ),
                interface=param_interface,
            )
        )

    if isinstance(granularity, int):
        assert 1 <= granularity <= N_horizon, (
            "Granularity must be between 0 and N_horizon (inclusive) when specified as an integer."
        )
        step = max(1, N_horizon // granularity)
        end_stages = list(range(0, N_horizon + 1, step))
        if end_stages[-1] != N_horizon:
            end_stages.append(N_horizon)
    else:
        end_stages = list(range(N_horizon + 1)) if granularity == "stagewise" else []

    params.extend(
        [
            AcadosParameter(
                name="temperature",  # Ambient temperature in Kelvin
                default=np.array([convert_temperature(20.0, "celsius", "kelvin")]),
                space=gym.spaces.Box(
                    low=np.array([convert_temperature(-20.0, "celsius", "kelvin")]),
                    high=np.array([convert_temperature(40.0, "celsius", "kelvin")]),
                    dtype=np.float64,
                ),
                interface="non-learnable",
                end_stages=list(range(N_horizon + 1)),
            ),
            AcadosParameter(
                name="solar",
                default=np.array([200.0]),  # Solar radiation in W/m²
                space=gym.spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([2000.0]),
                    dtype=np.float64,
                ),
                interface="non-learnable",
                end_stages=list(range(N_horizon + 1)),
            ),
            AcadosParameter(
                name="price",
                default=np.array([0.15]),  # Electricity price in €/kWh
                space=gym.spaces.Box(
                    low=np.array([0.00]),
                    high=np.array([10.0]),
                    dtype=np.float64,
                ),
                interface="non-learnable",
                end_stages=list(range(N_horizon + 1)),
            ),
        ]
    )

    # Comfort constraints for indoor temperature
    # Note: lb_Ti and ub_Ti are set at runtime via constraints_set() in the planner,
    # NOT as OCP parameters — they have no effect as parameters since the OCP model
    # never calls param_manager.get("lb_Ti"/"ub_Ti").
    params.extend(
        [
            AcadosParameter(
                name="ref_Ti",  # Reference indoor temperature in Kelvin
                default=np.array([convert_temperature(21.0, "celsius", "kelvin")]),
                space=gym.spaces.Box(
                    low=np.array([convert_temperature(10.0, "celsius", "kelvin")]),
                    high=np.array([convert_temperature(30.0, "celsius", "kelvin")]),
                    dtype=np.float64,
                ),
                interface="learnable",
                end_stages=end_stages,
            ),
            AcadosParameter(
                name="backoff_Ti",  # Backoff temperature for indoor temperature in Kelvin
                default=np.array([0.0]),
                space=gym.spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([2.0]),
                    dtype=np.float64,
                ),
                interface="fix",
                end_stages=end_stages,
            ),
        ]
    )

    params.extend(
        [
            AcadosParameter(
                name="log_q_Ti",
                default=np.array([-1.9]),  # log10 of the weight for indoor temperature residuals
                space=gym.spaces.Box(low=np.array([-2.0]), high=np.array([1.0]), dtype=np.float64),
                interface="learnable",
                end_stages=end_stages,
            ),
            AcadosParameter(
                name="q_dqh",
                default=np.array([0.001]),  # weight for residuals of rate of change of heater power
                space=gym.spaces.Box(low=np.array([0.0]), high=np.array([0.01]), dtype=np.float64),
                interface="fix",
                end_stages=end_stages,
            ),
        ]
    )

    return tuple(params)


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    N_horizon: int,
    name: str = "hvac",
    x0: np.ndarray | None = None,
    qh_max: float = 5.0,
) -> AcadosOcp:
    """Export the HVAC OCP.

    Augments the state with the previous input to encode the rate penalty.

    State:   x = [Ti, Th, Te, qh]  (K, K, K, kW)
        qh is the heating power applied at the previous step.
    Input:   u = [dqh, t]          (kW, kW)
        dqh is the increment; the actually applied power is qh_new = qh + dqh.
        t >= |qh_new| is an auxiliary variable for the LP absolute-value reformulation.
    Params per stage: [Ta, solar, price]
    Bounds on Ti as box constraints.

    The cost is price * t + q_dqh * dqh², which is purely quadratic in dqh and
    linear in t, giving a constant Hessian and making the OCP a true QP.

    Args:
        param_manager: The parameter manager containing the parameters for the OCP.
        N_horizon: Number of time steps in the horizon.
        name: Name of the OCP model.
        x0: Initial state. If None, a default value is used.
        qh_max: Maximum absolute heating power in kW (box constraint).


    Returns:
        AcadosOcp: The configured OCP object.
    """
    dt: float = 900.0  # Time step in seconds (15 minutes)

    ocp = AcadosOcp()

    param_manager.assign_to_ocp(ocp)

    # Model
    ocp.model.name = name

    # ── State: x_aug = [Ti, Th, Te, qh] ──────────────────────────────────────
    Ti = ca.SX.sym("Ti")
    Th = ca.SX.sym("Th")
    Te = ca.SX.sym("Te")
    qh = ca.SX.sym("qh")
    ocp.model.x = ca.vertcat(Ti, Th, Te, qh)

    # ── Input: [dqh, t] where qh_new = qh + dqh and t = |qh_new| via LP ─────
    dqh = ca.SX.sym("dqh")
    t = ca.SX.sym("t")
    ocp.model.u = ca.vertcat(dqh, t)

    # ── Discrete dynamics ─────────────────────────────────────────────────────
    Ad, Bd, Ed = transcribe_discrete_state_space(
        dt=dt,
        params=param_manager.recreate_dataclass(HydronicDynamicsParameters),
    )
    Bd_kW = Bd * 1e3  # W → kW for numerical conditioning

    x_phys = ca.vertcat(Ti, Th, Te)
    d_vec = ca.vertcat(
        param_manager.get("temperature"),  # Ambient temperature
        param_manager.get("solar"),  # Solar radiation
    )
    qh_new = qh + dqh
    x_next_phys = Ad @ x_phys + Bd_kW * qh_new + Ed @ d_vec
    ocp.model.disc_dyn_expr = ca.vertcat(x_next_phys, qh_new)

    # ── Cost ──────────────────────────────────────────────────────────────────
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # dqh² has constant Hessian; t enters linearly → overall constant Hessian.
    ocp.model.cost_expr_ext_cost = (
        0.25 * param_manager.get("price") * t + param_manager.get("q_dqh") * dqh**2
    )
    ocp.model.cost_expr_ext_cost_e = 1e-3 * (Ti - convert_temperature(20.0, "C", "K")) ** 2

    # ── Constraints ───────────────────────────────────────────────────────────
    Ti_ic = convert_temperature(20.0, "C", "K")
    ocp.constraints.x0 = np.array([Ti_ic, Ti_ic, Ti_ic, 0.0])

    # Slacked comfort box constraint on Ti; hard box constraint on qh.
    ocp.constraints.lbx = np.array([convert_temperature(12.0, "C", "K"), -qh_max])
    ocp.constraints.ubx = np.array([convert_temperature(25.0, "C", "K"), qh_max])
    ocp.constraints.idxbx = np.array([0, 3])

    ocp.constraints.idxsbx = np.array([0])  # slack only on Ti (first entry of idxbx)
    ocp.cost.zl = 1e2 * np.ones((1,))
    ocp.cost.Zl = 1e2 * np.ones((1,))
    ocp.cost.zu = 1e2 * np.ones((1,))
    ocp.cost.Zu = 1e2 * np.ones((1,))

    ocp.constraints.lbx_e = np.array([convert_temperature(12.0, "C", "K"), -qh_max])
    ocp.constraints.ubx_e = np.array([convert_temperature(25.0, "C", "K"), qh_max])
    ocp.constraints.idxbx_e = np.array([0, 3])

    # Box bounds: dqh in [-10, 10] kW (full swing), t >= 0
    ocp.constraints.lbu = np.array([-10.0, 0.0])
    ocp.constraints.ubu = np.array([10.0, qh_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # Linear constraints encoding t >= |qh_new| = |qh + dqh|:
    #   (qh + dqh) - t <= 0  →  C=[0,0,0,1], D=[1,-1]
    #  -(qh + dqh) - t <= 0  →  C=[0,0,0,-1], D=[-1,-1]
    # lg <= C @ x + D @ u <= ug
    ocp.constraints.C = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]])
    ocp.constraints.D = np.array([[1.0, -1.0], [-1.0, -1.0]])
    ocp.constraints.lg = np.array([-1e9, -1e9])
    ocp.constraints.ug = np.array([0.0, 0.0])

    # ── Solver options ────────────────────────────────────────────────────────
    ocp.solver_options.tf = N_horizon * dt
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp
