from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi.tools import struct_symSX

from leap_c.examples.hvac.util import transcribe_discrete_state_space
from leap_c.mpc import Mpc

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
        x0 = np.array([20.0, 20.0, 20.0])

    ocp = AcadosOcp()

    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = N_horizon

    ocp.model.name = name

    ocp.dims.nu = 1
    ocp.dims.nx = 3
    ocp.dims.np = 3

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)  # [Ti, Th, Te]
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)  # [qh]
    ocp.model.p = ca.SX.sym("p", ocp.dims.np)  # [Ta, Phi_s, price]

    Ad, Bd, Ed = transcribe_discrete_state_space(
        Ad=np.zeros((3, 3)),
        Bd=np.zeros((3, 1)),
        Ed=np.zeros((3, 2)),
        dt=300.0,  # 5 min sampling time
        params=params,
    )

    ocp.model.disc_dyn_expr = Ad @ ocp.model.x + Bd @ ocp.model.u + Ed @ ocp.model.p[:2]

    ocp.parameter_values = np.array([10.0, 0.0, 0.01])  # [Ta, Phi_s, price]

    # ocp = translate_learnable_param_to_p_global(
    #     nominal_param=params, learnable_param=learnable_params, ocp=ocp
    # )

    # Cost function
    ocp.cost.cost_type = "EXTERNAL"
    # ocp.model.cost_expr_ext_cost = (ocp.model.x[0] - 21.0) ** 2 + ocp.model.p[
    #     2
    # ] * ocp.model.u
    ocp.model.cost_expr_ext_cost = ocp.model.p[2] * ocp.model.u
    # ocp.cost.cost_type_e = "EXTERNAL"
    # ocp.model.cost_expr_ext_cost_e = (ocp.model.x[0] - 21.0) ** 2

    ocp.constraints.x0 = x0

    # Box constraints on u
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([2000.0])  # [W]
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([17.0])  # [°C] min indoor temperature
    ocp.constraints.ubx = np.array([25.0])  # [°C] max indoor temperature
    ocp.constraints.idxbx = np.array([0])

    ocp.constraints.idxsbx = np.array([0])

    ns = ocp.constraints.idxsbx.size
    ocp.cost.zl = 100 * np.ones((ns,))
    ocp.cost.Zl = 0 * np.ones((ns,))
    ocp.cost.zu = 100 * np.ones((ns,))
    ocp.cost.Zu = 0 * np.ones((ns,))

    # #############################
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp


def set_ocp_solver_options(ocp: AcadosOcp, exact_hess_dyn: bool):
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = False
    ocp.solver_options.with_solution_sens_wrt_params = False


if __name__ == "__main__":
    mpc = HvacMpc()

    ocp = mpc.ocp

    ocp_solver = AcadosOcpSolver(acados_ocp=ocp)

    # Set electricity prices
    parameter_values = np.tile(
        np.array([10.0, 0.0, 0.01]), (ocp_solver.acados_ocp.solver_options.N_horizon, 1)
    )

    parameter_values[:, -1] = np.random.uniform(
        0.01, 0.2, size=(ocp_solver.acados_ocp.solver_options.N_horizon)
    )

    for stage in range(ocp.solver_options.N_horizon):
        ocp_solver.set(stage, "p", parameter_values[stage, :])

    # TODO: Set weather forecast (p[: 0] ambient temperature, p[:, 1] solar radiation)

    # TODO: Set initial state

    # Solve the OCP

    status = ocp_solver.solve()

    x = np.array([ocp_solver.get(stage, "x") for stage in range(ocp.dims.N)])
    u = np.array([ocp_solver.get(stage, "u") for stage in range(ocp.dims.N)])

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.stairs(x[:, 0], label="Indoor temperature")
    plt.stairs(x[:, 1], label="Radiator temperature")
    plt.stairs(x[:, 2], label="Envelope temperature")
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.stairs(u[:, 0], label="Heat input to radiator")
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.stairs(parameter_values[:, -1], label="Electricity price")
    plt.grid()
    plt.legend()
    plt.xlabel("Time step")
    plt.tight_layout()
    plt.show()
