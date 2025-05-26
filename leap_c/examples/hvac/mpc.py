from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi.tools import struct_symSX

from leap_c.examples.hvac.util import transcribe_discrete_state_space, get_f_expl_expr
from leap_c.acados.mpc import Mpc

import matplotlib.pyplot as plt


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
        if params is None:
            params = {
                # Effective window area [m²]
                "gAw": 40.344131392192,
                # Thermal capacitances [J/K]
                "Ch": 10447262.2318648,  # Radiator
                "Ci": 14827137.0377258,  # Indoor air
                "Ce": 50508258.9032192,  # Building envelope
                # Noise parameters
                "e11": -30.0936560706053,  # Measurement noise
                "sigmai": -23.3175423490014,
                "sigmah": -19.5274067368137,
                "sigmae": -5.07591222090641,
                # Thermal resistances [K/W]
                "Rea": 0.00163027389197229,  # Envelope to outdoor
                "Rhi": 0.000437603769897038,  # Radiator to indoor air
                "Rie": 0.000855786902577802,  # Indoor air to envelope
                # Heater parameters
                "eta": 0.98,  # Efficiency for electric heater
            }

        # Store parameters
        self.params = params

        learnable_params = learnable_params if learnable_params is not None else []

        print("learnable_params: ", learnable_params)

        ocp = export_parametric_ocp(
            params=params,
            learnable_params=learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
        )

        set_ocp_solver_options(ocp=ocp, exact_hess_dyn=True)

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
        dt=300.0,  # 5 min sampling time
        params=params,
    )

    ocp.model.disc_dyn_expr = Ad @ ocp.model.x + Bd @ ocp.model.u + Ed @ ocp.model.p[:2]

    ocp.model.f_expl_expr = get_f_expl_expr(
        x=ocp.model.x,
        u=ocp.model.u,
        d=ocp.model.p,
        params=params,
    )

    ocp.parameter_values = np.array([10.0, 0.0, 0.01])  # [Ta, Phi_s, price]

    # ocp = translate_learnable_param_to_p_global(
    #     nominal_param=params, learnable_param=learnable_params, ocp=ocp
    # )

    if False:
        # Cost function
        ocp.cost.cost_type = "EXTERNAL"
        # ocp.model.cost_expr_ext_cost = (ocp.model.x[0] - 21.0) ** 2 + ocp.model.p[
        #     2
        # ] * ocp.model.u
        ocp.model.cost_expr_ext_cost = ocp.model.p[2] * ocp.model.u
        # ocp.cost.cost_type_e = "EXTERNAL"
        # ocp.model.cost_expr_ext_cost_e = (ocp.model.x[0] - 21.0) ** 2
    else:
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = ca.vertcat(
            ocp.model.x[0],  # Indoor temperature
            ocp.model.u,  # Heat input to radiator
        )
        ocp.cost.yref = np.array([21.0, 0.0])
        ocp.cost.W = np.array([[1.0, 0.0], [0.0, 1.0]])
        ocp.cost.W_e = np.array([[1.0]])
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = ocp.model.x[0]
        ocp.cost.yref_e = np.array([21.0])

    ocp.constraints.x0 = x0

    # Box constraints on u
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([5000.0])  # [W]
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([17.0])  # [°C] min indoor temperature
    ocp.constraints.ubx = np.array([25.0])  # [°C] max indoor temperature
    ocp.constraints.idxbx = np.array([0])

    if True:
        ocp.constraints.idxsbx = np.array([0])
        ns = ocp.constraints.idxsbx.size
        ocp.cost.zl = 1e5 * np.ones((ns,))
        ocp.cost.Zl = 1e5 * np.ones((ns,))
        ocp.cost.zu = 1e5 * np.ones((ns,))
        ocp.cost.Zu = 1e5 * np.ones((ns,))

    # #############################
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp


def set_ocp_solver_options(ocp: AcadosOcp, exact_hess_dyn: bool):
    # ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    # ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver_ric_alg = 1
    # ocp.solver_options.with_value_sens_wrt_params = False
    # ocp.solver_options.with_solution_sens_wrt_params = False


if __name__ == "__main__":
    mpc = HvacMpc()

    ocp = mpc.ocp

    ocp_solver = AcadosOcpSolver(acados_ocp=ocp)

    # Set electricity prices
    parameter_values = np.tile(
        np.array([10.0, 0.0, 0.01]), (ocp_solver.acados_ocp.solver_options.N_horizon, 1)
    )

    # random_prices = True
    price_type = "random_prices"

    if price_type == "random_prices":
        parameter_values[:, -1] = np.random.uniform(
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

    # TODO: Set weather forecast (p[: 0] ambient temperature, p[:, 1] solar radiation)

    # TODO: Set initial state

    x0 = np.array([18.0, 18.0, 18.0])
    for stage in range(ocp.solver_options.N_horizon):
        ocp_solver.set(stage, "x", x0)

    # Solve the OCP

    _ = ocp_solver.solve_for_x0(x0_bar=x0)
    status = ocp_solver.get_status()

    x = np.array([ocp_solver.get(stage, "x") for stage in range(ocp.dims.N)])
    u = np.array([ocp_solver.get(stage, "u") for stage in range(ocp.dims.N)])

    lb_indoor = ocp_solver.acados_ocp.constraints.lbx[0] * np.ones(ocp.dims.N)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.stairs(x[:, 0], label="Indoor temperature")
    plt.stairs(lb_indoor, label="Indoor temperature lower bound", linestyle="--")
    plt.stairs(x[:, 1], label="Radiator temperature")
    plt.stairs(x[:, 2], label="Envelope temperature")
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.stairs(u[:, 0], label="Heat input to radiator", color="tab:blue")
    ax2.stairs(parameter_values[:, -1], label="Electricity price", color="tab:orange")

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
