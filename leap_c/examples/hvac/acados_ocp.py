from dataclasses import asdict
from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import ACADOS_INFTY, AcadosOcp
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

    params = [
        AcadosParameter(
            name=k,
            default=np.array([v]),
            space=gym.spaces.Box(
                low=0.95 * np.array([v]), high=1.05 * np.array([v]), dtype=np.float64
            ),
            interface="fix" if "dynamics" not in interface else "learnable",
        )
        for k, v in asdict(hydronic_params.dynamics).items()
    ]

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
    params.extend(
        [
            AcadosParameter(
                name="lb_Ti",  # Lower bound on indoor temperature in Kelvin
                default=np.array([convert_temperature(17.0, "celsius", "kelvin")]),
                space=gym.spaces.Box(
                    low=np.array([convert_temperature(15.0, "celsius", "kelvin")]),
                    high=np.array([convert_temperature(19.0, "celsius", "kelvin")]),
                    dtype=np.float64,
                ),
                interface="non-learnable",
                end_stages=list(range(N_horizon + 1)),
            ),
            AcadosParameter(
                name="ub_Ti",  # Upper bound on indoor temperature in Kelvin
                default=np.array([convert_temperature(23.0, "celsius", "kelvin")]),
                space=gym.spaces.Box(
                    low=np.array([convert_temperature(21.0, "celsius", "kelvin")]),
                    high=np.array([convert_temperature(25.0, "celsius", "kelvin")]),
                    dtype=np.float64,
                ),
                interface="non-learnable",
                end_stages=list(range(N_horizon + 1)),
            ),
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
                default=np.array([0.7]),
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
                name="q_Ti",
                default=np.array([0.2]),  # weight for indoor temperature residuals
                space=gym.spaces.Box(
                    low=np.array([0.01]), high=np.array([0.55]), dtype=np.float64
                ),
                interface="learnable",
                end_stages=end_stages,
            ),
            AcadosParameter(
                name="q_dqh",
                default=np.array([1.0]),  # weight for residuals of rate of change of heater power
                space=gym.spaces.Box(low=np.array([0.5]), high=np.array([1.5]), dtype=np.float64),
                interface="fix",
                end_stages=end_stages,
            ),
            AcadosParameter(
                name="q_ddqh",
                default=np.array([1.0]),  # weight for residuals of acceleration of heater power
                space=gym.spaces.Box(low=np.array([0.5]), high=np.array([1.5]), dtype=np.float64),
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
) -> AcadosOcp:
    """Export the HVAC OCP.

    Args:
        param_manager: The parameter manager containing the parameters for the OCP.
        N_horizon: Number of time steps in the horizon.
        name: Name of the OCP model.
        x0: Initial state. If None, a default value is used.

    Returns:
        AcadosOcp: The configured OCP object.
    """
    dt: float = 900.0  # Time step in seconds (15 minutes)

    ocp = AcadosOcp()

    param_manager.assign_to_ocp(ocp)

    # Model
    ocp.model.name = name

    ocp.model.x = ca.vertcat(
        ca.SX.sym("Ti"),  # Indoor air temperature
        ca.SX.sym("Th"),  # Radiator temperature
        ca.SX.sym("Te"),  # Envelope temperature
    )

    qh = ca.SX.sym("qh")  # Heat input to radiator
    dqh = ca.SX.sym("dqh")  # Velocity of heat input to radiator
    ddqh = ca.SX.sym("ddqh")  # Acceleration of heat input to radiator

    Ad, Bd, Ed = transcribe_discrete_state_space(
        Ad=np.zeros((3, 3)),
        Bd=np.zeros((3, 1)),
        Ed=np.zeros((3, 2)),
        dt=dt,
        params=param_manager.recreate_dataclass(HydronicDynamicsParameters),
    )

    d = ca.vertcat(
        param_manager.get("temperature"),  # Ambient temperature
        param_manager.get("solar"),  # Solar radiation
    )
    ocp.model.disc_dyn_expr = Ad @ ocp.model.x + Bd @ qh + Ed @ d

    # Augment the model with double integrator for the control input
    ocp.model.x = ca.vertcat(ocp.model.x, qh, dqh)
    ocp.model.disc_dyn_expr = ca.vertcat(
        ocp.model.disc_dyn_expr,
        qh + dt * dqh + 0.5 * dt**2 * ddqh,
        dqh + dt * ddqh,
    )
    ocp.model.u = ddqh

    # Cost function
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (
        0.25 * param_manager.get("price") * qh
        + param_manager.get("q_Ti") * (param_manager.get("ref_Ti") - ocp.model.x[0]) ** 2
        + param_manager.get("q_dqh") * (dqh) ** 2
        + param_manager.get("q_ddqh") * (ddqh) ** 2
    )

    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = (
        0.25 * param_manager.get("price") * qh
        + param_manager.get("q_Ti") * (param_manager.get("ref_Ti") - ocp.model.x[0]) ** 2
        + param_manager.get("q_dqh") * (dqh) ** 2
    )

    # Constraints
    ocp.constraints.x0 = x0 or np.array(
        [convert_temperature(20.0, "celsius", "kelvin")] * 3 + [0.0, 0.0]
    )

    # Comfort constraints
    ocp.model.con_h_expr = ca.vertcat(
        ocp.model.x[0] - param_manager.get("lb_Ti") - param_manager.get("backoff_Ti"),
        param_manager.get("ub_Ti") - ocp.model.x[0] - param_manager.get("backoff_Ti"),
    )
    ocp.constraints.lh = np.array([0.0, 0.0])
    ocp.constraints.uh = np.array([ACADOS_INFTY, ACADOS_INFTY])

    ocp.constraints.idxsh = np.array([0, 1])
    ocp.cost.zl = 1e3 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.Zl = 1e3 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.zu = 1e3 * np.ones((ocp.constraints.idxsh.size,))
    ocp.cost.Zu = 1e3 * np.ones((ocp.constraints.idxsh.size,))

    ocp.constraints.lbx = np.array([0.0])  # Can only consume power
    ocp.constraints.ubx = np.array([5000.0])  # Watt
    ocp.constraints.idxbx = np.array([3])  # qh

    # Solver options
    ocp.solver_options.tf = N_horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp
