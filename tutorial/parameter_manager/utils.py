"""Shared constants and OCP construction for the parameter manager tutorials."""

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameterManager

N_HORIZON = 10
BATCH_SIZE = 4

R_THERMAL = 2.0
C_THERMAL = 1.5


def build_manager(N_horizon: int = N_HORIZON) -> AcadosParameterManager:
    """Build and return the parameter manager for the temperature-control tutorial.

    Parameters:
        - ``dt``               fixed time step [h]  (non-differentiable)
        - ``outdoor_temp``     ambient temperature [degC]  (non-differentiable)
        - ``comfort_setpoint`` comfort reference temperature [degC]  (differentiable)
        - ``price``            electricity price [EUR/kWh], two stage blocks
          (differentiable)
    """
    manager = AcadosParameterManager(N_horizon=N_horizon)

    manager.register_parameter(
        name="dt",
        default=np.array([0.25]),
        differentiable=False,
    )
    manager.register_parameter(
        name="outdoor_temp",
        default=np.array([20.0]),
        differentiable=False,
    )
    manager.register_parameter(
        name="comfort_setpoint",
        default=np.array([21.0]),
        differentiable=True,
    )
    manager.register_parameter(
        name="price",
        default=np.array([0.15]),
        differentiable=True,
        splits=[4, N_horizon],
    )
    return manager


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

    T = ca.SX.sym("T")
    model.x = T

    q = ca.SX.sym("q")
    model.u = q

    dt = manager.get("dt")
    outdoor_temp = manager.get("outdoor_temp")
    comfort_ref = manager.get("comfort_setpoint")
    price = manager.get("price")

    model.disc_dyn_expr = T + dt * ((outdoor_temp - T) / (R_THERMAL * C_THERMAL) + q / C_THERMAL)

    model.cost_expr_ext_cost = (T - comfort_ref) ** 2 + price * q
    model.cost_expr_ext_cost_e = (T - comfort_ref) ** 2

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model = model

    ocp.constraints.x0 = np.array([20.0])

    ocp.solver_options.tf = N_horizon * dt
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "DISCRETE"
    return ocp
