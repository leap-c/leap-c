from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi.tools import struct_symSX

from leap_c.examples.hvac.util import transcribe_discrete_state_space, get_f_expl_expr
from leap_c.acados.mpc import Mpc

import matplotlib.pyplot as plt

from leap_c.examples.hvac.util import (
    BestestParameters,
    BestestHydronicParameters,
    BestestHydronicHeatpumpParameters,
)

from scipy.constants import convert_temperature


class HvacMpc(Mpc):
    """docstring for PointMassMPC."""

    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 288,
        T_horizon: float = 24 * 3600.0,
        discount_factor: float = 0.99,
        n_batch: int = 64,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
    ):
        # Default parameters if none provided
        self.params = (
            params if params is not None else BestestHydronicParameters().to_dict()
        )

        learnable_params = learnable_params if learnable_params is not None else []

        print("learnable_params: ", learnable_params)

        ocp = export_parametric_ocp(
            params=self.params,
            learnable_params=learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
        )

        set_ocp_solver_options(ocp=ocp)

        self.ocp = ocp

        # TODO: Comment in again after providing p_global
        # self.given_default_param_dict = params

        # super().__init__(
        #     ocp=ocp,
        #     n_batch=n_batch,
        #     export_directory=export_directory,
        #     export_directory_sensitivity=export_directory_sensitivity,
        #     throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        # )


def export_parametric_ocp(
    params: dict[str, np.ndarray],
    name: str = "hvac",
    learnable_params: list[str] | None = None,
    N_horizon: int = 288,
    tf: float = 24.0 * 3600.0,
    x0: np.ndarray | None = None,
) -> AcadosOcp:
    if x0 is None:
        x0 = np.array([17.0] * 3)

    ocp = AcadosOcp()

    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = N_horizon

    ocp.model.name = name

    ocp.model.x = ca.vertcat(
        ca.SX.sym("Ti"),  # Indoor air temperature
        ca.SX.sym("Th"),  # Radiator temperature
        ca.SX.sym("Te"),  # Envelope temperature
    )

    ocp.model.xdot = ca.vertcat(
        ca.SX.sym("dTi_dt"),  # Indoor air temperature derivative
        ca.SX.sym("dTh_dt"),  # Radiator temperature derivative
        ca.SX.sym("dTe_dt"),  # Envelope temperature derivative
    )

    ocp.model.u = ca.SX.sym("qh")  # Heat input to radiator

    ocp.model.p = ca.vertcat(
        ca.SX.sym("Ta"),  # Ambient temperature
        ca.SX.sym("Phi_s"),  # Solar radiation
        ca.SX.sym("price"),  # Electricity price
    )

    Ad, Bd, Ed = transcribe_discrete_state_space(
        Ad=np.zeros((3, 3)),
        Bd=np.zeros((3, 1)),
        Ed=np.zeros((3, 2)),
        dt=tf / N_horizon,
        params=params,
    )

    ocp.model.disc_dyn_expr = Ad @ ocp.model.x + Bd @ ocp.model.u + Ed @ ocp.model.p[:2]

    ocp.parameter_values = np.array([convert_temperature(10.0, "c", "k"), 0.0, 0.01])

    # Cost function
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = ocp.model.p[2] * ocp.model.u

    ocp.constraints.x0 = x0

    # Box constraints on u
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([2000.0])  # [W]
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = convert_temperature(np.array([17.0]), "celsius", "kelvin")
    ocp.constraints.ubx = convert_temperature(np.array([25.0]), "celsius", "kelvin")
    ocp.constraints.idxbx = np.array([0])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zl = 1e5 * np.ones((ocp.constraints.idxsbx.size,))
    ocp.cost.Zl = 1e5 * np.ones((ocp.constraints.idxsbx.size,))
    ocp.cost.zu = 1e5 * np.ones((ocp.constraints.idxsbx.size,))
    ocp.cost.Zu = 1e5 * np.ones((ocp.constraints.idxsbx.size,))

    # #############################
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp


def set_ocp_solver_options(ocp: AcadosOcp) -> None:
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    ocp.solver_options.hpipm_mode = "ROBUST"


if __name__ == "__main__":
    dt = 900.0  # sampling time in seconds
    N_horizon = 28
    T_horizon = N_horizon * dt  # Total time horizon in seconds

    # T_horizon = 4 * 3600.0  # Total time horizon in seconds
    # N_horizon = int(T_horizon / dt)  # Number of time steps in the horizon

    mpc = HvacMpc(N_horizon=N_horizon, T_horizon=T_horizon)

    ocp = mpc.ocp
    ocp_solver = AcadosOcpSolver(acados_ocp=ocp)

    # Exogenous parameters
    Ta = convert_temperature(10.0, "celsius", "kelvin")  # Ambient temperature in K
    Phi_s = 0.0  # Solar radiation in W/m^2
    price = 0.01  # Electricity price in EUR/kWh

    # Set electricity prices
    parameter_values = np.tile(
        np.array([Ta, Phi_s, price]),
        (ocp_solver.acados_ocp.solver_options.N_horizon, 1),
    )

    ########## Set electricity prices ##########
    price_type = "constant_prices"  # Options: "random_prices", "sinusoidal_prices", "constant_prices"
    rng = np.random.default_rng(seed=42)

    if price_type == "random_prices":
        parameter_values[:, -1] = rng.uniform(
            0.01, 0.2, size=(ocp_solver.acados_ocp.solver_options.N_horizon)
        )
    elif price_type == "sinusoidal_prices":
        parameter_values[:, -1] = 0.01 + 0.1 * np.sin(
            np.linspace(0, 2 * np.pi, ocp_solver.acados_ocp.solver_options.N_horizon)
        )
    elif price_type == "constant_prices":
        parameter_values[:, -1] = 0.01

    for stage in range(ocp.solver_options.N_horizon):
        ocp_solver.set(stage, "p", parameter_values[stage, :])

    ########## Set ambient temperature and solar radiation ##########
    # TODO: Set weather forecast (p[: 0] ambient temperature, p[:, 1] solar radiation)

    ###### Set initial state ##########
    x0 = convert_temperature(np.array([17.1, 17.1, 17.1]), "celsius", "kelvin")
    for stage in range(ocp.solver_options.N_horizon):
        ocp_solver.set(stage, "x", x0)

    # Solve the OCP
    _ = ocp_solver.solve_for_x0(x0_bar=x0)
    status = ocp_solver.get_status()

    x = np.array([ocp_solver.get(stage, "x") for stage in range(ocp.dims.N + 1)])
    u = np.array([ocp_solver.get(stage, "u") for stage in range(ocp.dims.N)])

    lb_indoor = ocp_solver.acados_ocp.constraints.lbx[0] * np.ones_like(x[:, 0])

    time_steps = np.arange(len(x)) * dt / 3600.0  # Convert to hours

    plt.figure()
    ax1 = plt.subplot(3, 1, 1)
    plt.step(time_steps, x[:, 0], label="Indoor temperature", color="red")
    plt.plot(
        time_steps, lb_indoor, label="Indoor temperature lower bound", linestyle="--"
    )
    plt.step(time_steps, x[:, 2], label="Envelope temperature")
    # plt.step(time_steps, Ta * np.ones_like(x[:, 2]), label="Ambient temperature")
    plt.ylabel("Temperature (K)")
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2, sharex=ax1)
    plt.step(time_steps, x[:, 1], label="Radiator temperature")
    plt.grid()
    plt.legend()
    plt.ylabel("Temperature (K)")
    plt.tight_layout()

    plt.subplot(3, 1, 3, sharex=ax1)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.step(
        time_steps[: len(u)], u[:, 0], label="Heat input to radiator", color="tab:blue"
    )
    ax2.step(
        time_steps[: len(parameter_values)],
        parameter_values[:, -1],
        label="Electricity price",
        linestyle="--",
        color="tab:orange",
    )

    ax1.set_ylabel("Heat input to radiator (W)", color="tab:blue")
    ax2.set_ylabel("Electricity price (EUR/kWh)", color="tab:orange")

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.grid()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.xlabel("Time step")
    plt.tight_layout()
    plt.show()
