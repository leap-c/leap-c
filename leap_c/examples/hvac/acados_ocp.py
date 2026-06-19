from dataclasses import fields
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
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosParameterManagerTorch

HvacAcadosParamInterface = Literal["reference", "reference_dynamics"]
"""Determines the exposed parameter interface of the planner.
- "reference": Only reference parameters are exposed (e.g., ambient temperature,
  solar radiation, electricity price, comfort bounds, etc.).
- "reference_dynamics": Both reference parameters and dynamics parameters are exposed.
"""

HvacAcadosParamGranularity = Literal["global", "stagewise"] | int


def export_parametric_ocp(
    interface: HvacAcadosParamInterface,
    granularity: HvacAcadosParamGranularity,
    N_horizon: int,
    name: str = "hvac",
    hydronic_params: HydronicParameters | None = None,
    ta_learnable: bool = False,
    solar_learnable: bool = False,
    price_learnable: bool = False,
) -> tuple[AcadosOcp, AcadosParameterManager]:
    """Export the HVAC OCP.

    Augments the state with the previous input to encode the rate penalty.

    State:   x = [Ti, Th, Te, qh_prev]  (K, K, K, kW)
    Input:   u = qh                      (kW)
    Params per stage: [Ta, solar, price]
    Bounds on Ti as box constraints.

    Args:
        interface: The parameter interface type.
        granularity: The granularity of the parameters.
        N_horizon: Number of time steps in the horizon.
        name: Name of the OCP model.
        hydronic_params: Optional HydronicParameters object.
        ta_learnable: Whether ambient temperature is learnable.
        solar_learnable: Whether solar radiation is learnable.
        price_learnable: Whether electricity price is learnable.

    Returns:
        tuple[AcadosOcp, AcadosParameterManager]: The OCP object and its parameter manager.
    """
    hydronic_params = HydronicParameters() if hydronic_params is None else hydronic_params

    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon

    manager = AcadosParameterManagerTorch(N_horizon=N_horizon)

    if isinstance(granularity, int):
        assert 1 <= granularity <= N_horizon, (
            "Granularity must be between 0 and N_horizon (inclusive) when specified as an integer."
        )
        step = max(1, N_horizon // granularity)
        splits_list = list(range(0, N_horizon + 1, step))
        if splits_list[-1] != N_horizon:
            splits_list.append(N_horizon)
        splits = splits_list
    elif granularity == "stagewise":
        splits = "stagewise"
    else:
        splits = "global"

    # Register non-learnable forecast parameters
    temperature = manager.register_parameter(
        "temperature",
        default=np.array([convert_temperature(20.0, "celsius", "kelvin")]),
        space=gym.spaces.Box(
            low=np.array([convert_temperature(-20.0, "celsius", "kelvin")]),
            high=np.array([convert_temperature(40.0, "celsius", "kelvin")]),
            dtype=np.float64,
        ),
        differentiable=ta_learnable,
        splits=list(range(N_horizon + 1)),
    )
    solar = manager.register_parameter(
        "solar",
        default=np.array([200.0]),
        space=gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([2000.0]),
            dtype=np.float64,
        ),
        differentiable=solar_learnable,
        splits=list(range(N_horizon + 1)),
    )
    price = manager.register_parameter(
        "price",
        default=np.array([0.15]),
        space=gym.spaces.Box(
            low=np.array([0.00]),
            high=np.array([10.0]),
            dtype=np.float64,
        ),
        differentiable=price_learnable,
        splits=list(range(N_horizon + 1)),
    )

    # Register learnable reference parameters
    manager.register_parameter(
        "ref_Ti",
        default=np.array([convert_temperature(21.0, "celsius", "kelvin")]),
        space=gym.spaces.Box(
            low=np.array([convert_temperature(10.0, "celsius", "kelvin")]),
            high=np.array([convert_temperature(30.0, "celsius", "kelvin")]),
            dtype=np.float64,
        ),
        differentiable=True,
        splits=splits,
    )

    # Sunsetted "fix" parameters - they are now local python constants
    q_dqh = np.array([0.001])

    # Dynamics parameters (can be learnable if interface is reference_dynamics)
    scaling_shifts = {
        "gAw": 0.8,
        "Rea": 0.8,
        "Rhi": 0.8,
        "Rie": 0.8,
        "Ch": 0.8,
        "Ci": 0.8,
        "Ce": 0.8,
    }

    dyn_fields = {}
    for f in fields(HydronicDynamicsParameters):
        k = f.name
        v = getattr(hydronic_params.dynamics, k)
        default_value = scaling_shifts[k] * np.array([v]) if k in scaling_shifts else np.array([v])

        if k in scaling_shifts and "dynamics" in interface:
            # Register as learnable parameter
            dyn_fields[k] = manager.register_parameter(
                name=k,
                default=default_value,
                space=gym.spaces.Box(
                    low=0.7 * np.array([v]), high=1.3 * np.array([v]), dtype=np.float64
                ),
                differentiable=True,
            )
        else:
            # Keep as a fixed value constant
            dyn_fields[k] = v

    ######## Model ########
    ocp.model.name = name

    # State: x_aug = [Ti, Th, Te, qh_prev]
    Ti = ca.SX.sym("Ti")
    Th = ca.SX.sym("Th")
    Te = ca.SX.sym("Te")
    qh_prev = ca.SX.sym("qh_prev")
    ocp.model.x = ca.vertcat(Ti, Th, Te, qh_prev)

    # Input: qh [kW]
    qh = ca.SX.sym("qh")
    ocp.model.u = qh

    # Discrete dynamics
    Ad, Bd, Ed = transcribe_discrete_state_space(
        dt=900.0,
        params=HydronicDynamicsParameters(**dyn_fields),
    )
    Bd_kW = Bd * 1e3  # W → kW for numerical conditioning

    x_phys = ca.vertcat(Ti, Th, Te)
    d_vec = ca.vertcat(temperature, solar)
    x_next_phys = Ad @ x_phys + Bd_kW * qh + Ed @ d_vec
    ocp.model.disc_dyn_expr = ca.vertcat(x_next_phys, qh)

    # Cost parameters
    manager.register_parameter(
        "log_q_Ti",
        default=np.array([-1.9]),
        space=gym.spaces.Box(low=np.array([-2.0]), high=np.array([1.0]), dtype=np.float64),
        differentiable=True,
        splits=splits,
    )

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model.cost_expr_ext_cost = 0.25 * price * qh + q_dqh * (qh - qh_prev) ** 2
    ocp.model.cost_expr_ext_cost_e = 1e-3 * (Ti - convert_temperature(20.0, "C", "K")) ** 2

    ######## Constraints ########
    Ti_ic = convert_temperature(20.0, "C", "K")
    ocp.constraints.x0 = np.array([Ti_ic, Ti_ic, Ti_ic, 0.0])

    # Slacked comfort box constraint on Ti
    ocp.constraints.lbx = np.array([convert_temperature(12.0, "C", "K")])
    ocp.constraints.ubx = np.array([convert_temperature(30.0, "C", "K")])
    ocp.constraints.idxbx = np.array([0])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zl = 1e3 * np.ones((1,))
    ocp.cost.Zl = 1e3 * np.ones((1,))
    ocp.cost.zu = 1e3 * np.ones((1,))
    ocp.cost.Zu = 1e3 * np.ones((1,))

    ocp.constraints.lbx_e = np.array([convert_temperature(12.0, "C", "K")])
    ocp.constraints.ubx_e = np.array([convert_temperature(30.0, "C", "K")])
    ocp.constraints.idxbx_e = np.array([0])

    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([5.0])
    ocp.constraints.idxbu = np.array([0])

    ######## Solver options ########
    ocp.solver_options.tf = N_horizon * 900.0
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp, manager
