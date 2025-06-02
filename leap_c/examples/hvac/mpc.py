import time
from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi.tools import struct_symSX
from scipy.constants import convert_temperature

from leap_c.acados.mpc import Mpc
from leap_c.examples.hvac.util import (
    BestestHydronicParameters,
    EnergyPriceProfile,
    create_constant_comfort_bounds,
    create_constant_disturbance,
    create_disturbance_from_weather,
    create_price_profile_from_spot_sprices,
    create_realistic_comfort_bounds,
    create_time_of_use_energy_costs,
    load_price_data,
    load_weather_data,
    plot_ocp_results,
    transcribe_discrete_state_space,
)


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
        dt=900.0,  # 15 minutes in seconds
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
    ocp.constraints.ubu = np.array([5000.0])  # [W]
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = convert_temperature(np.array([17.0]), "celsius", "kelvin")
    ocp.constraints.ubx = convert_temperature(np.array([25.0]), "celsius", "kelvin")
    ocp.constraints.idxbx = np.array([0])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zl = 1e4 * np.ones((ocp.constraints.idxsbx.size,))
    ocp.cost.Zl = 1e4 * np.ones((ocp.constraints.idxsbx.size,))
    ocp.cost.zu = 1e4 * np.ones((ocp.constraints.idxsbx.size,))
    ocp.cost.Zu = 1e4 * np.ones((ocp.constraints.idxsbx.size,))

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
    ocp.solver_options.qp_solver_cond_N = 4


def create_random_price_profile(
    N: int,
    rng: np.random.Generator,
    price_min: float = 0.1,
    price_max: float = 0.2,
) -> EnergyPriceProfile:
    """Create a random price profile for the given number of horizons."""
    return EnergyPriceProfile(price=rng.uniform(price_min, price_max, size=(N,)))


def get_solution(ocp_solver: AcadosOcpSolver) -> dict[str, np.ndarray]:
    """Extract the solution from the OCP solver."""
    ocp = ocp_solver.acados_ocp
    return {
        "success": ocp_solver.get_status() == 0,
        "states": np.array(
            [ocp_solver.get(stage, "x") for stage in range(ocp.dims.N + 1)]
        ),
        "controls": np.array(
            [ocp_solver.get(stage, "u") for stage in range(ocp.dims.N)]
        ),
        "slack_lower": np.array(
            [ocp_solver.get(stage, "sl") for stage in range(1, ocp.dims.N)]
        ),
        "slack_upper": np.array(
            [ocp_solver.get(stage, "su") for stage in range(1, ocp.dims.N)]
        ),
        "indoor_temperatures": np.array(
            [ocp_solver.get(stage, "x")[0] for stage in range(ocp.dims.N + 1)]
        ),
        "radiator_temperatures": np.array(
            [ocp_solver.get(stage, "x")[1] for stage in range(ocp.dims.N + 1)]
        ),
        "envelope_temperatures": np.array(
            [ocp_solver.get(stage, "x")[2] for stage in range(ocp.dims.N + 1)]
        ),
        "ambient_temperature": np.array(
            [ocp_solver.get(stage, "p")[0] for stage in range(ocp.dims.N + 1)]
        ),
        "solar_radiation": np.array(
            [ocp_solver.get(stage, "p")[1] for stage in range(ocp.dims.N + 1)]
        ),
        "electricity_price": np.array(
            [ocp_solver.get(stage, "p")[2] for stage in range(ocp.dims.N + 1)]
        ),
        # NOTE: This is the total cost, not energy-consumption cost
        "cost": ocp_solver.get_cost(),
    }


if __name__ == "__main__":
    dt = 1.0  # time step in quarterly hours
    horizon_hours = 24  # prediction horizon in hours
    N_horizon = horizon_hours * 4

    # Exogenous parameters
    weather_data_path = Path(__file__).parent / "weather.csv"
    if weather_data_path.exists():
        weather_data = load_weather_data(weather_data_path)

        start_time = weather_data.index[0]

        disturbance = create_disturbance_from_weather(
            weather_df=weather_data,
            start_time=start_time,
            horizon_hours=horizon_hours,
        )
    else:
        start_time = (0.0,)
        disturbance = create_constant_disturbance(
            N=N_horizon,
            T_outdoor=convert_temperature(10.0, "celsius", "kelvin"),
            solar_radiation=20.0,
        )

    price_profile_type = "historical"  # or "random"
    if price_profile_type == "historical":
        price_data = load_price_data(Path(__file__).parent / "spot_prices.csv")
        energy_price_profile = create_price_profile_from_spot_sprices(
            price_df=price_data,
            start_time=start_time,
            horizon_hours=horizon_hours,
        )
    elif price_profile_type == "time_of_use":
        energy_price_profile = create_time_of_use_energy_costs(
            N=N_horizon,
            start_time=start_time,
        )
    elif price_profile_type == "random":
        energy_price_profile = create_random_price_profile(
            N=N_horizon,
            rng=np.random.default_rng(seed=22),
            price_min=0.1,
            price_max=0.2,
        )
    else:
        raise ValueError(
            f"Unknown price profile type: {price_profile_type}. "
            "Use 'random' or 'historical'."
        )

    mpc = HvacMpc(N_horizon=N_horizon, T_horizon=N_horizon)

    ocp = mpc.ocp
    ocp_solver = AcadosOcpSolver(acados_ocp=ocp)

    parameter_values = np.hstack(
        (
            disturbance.T_outdoor.reshape(-1, 1),
            disturbance.solar_radiation.reshape(-1, 1),
            energy_price_profile.price.reshape(-1, 1),
        )
    )

    for stage in range(ocp.solver_options.N_horizon):
        ocp_solver.set(stage, "p", parameter_values[stage, :])

    # Comfort bounds
    comfort_bounds_type = "realistic"  # or "constant"
    if comfort_bounds_type == "realistic":
        comfort_bounds = create_realistic_comfort_bounds(
            N=N_horizon,
            start_time=start_time,
        )
    elif comfort_bounds_type == "constant":
        const_comfort_bounds = create_constant_comfort_bounds(
            N=N_horizon,
            T_lower=convert_temperature(17.0, "celsius", "kelvin"),
            T_upper=convert_temperature(23.0, "celsius", "kelvin"),
        )
    else:
        raise ValueError(
            f"Unknown comfort bounds type: {comfort_bounds_type}. "
            "Use 'constant' or 'realistic'."
        )

    for stage in range(1, ocp.solver_options.N_horizon):
        ocp_solver.constraints_set(stage, "lbx", comfort_bounds.T_lower[stage])
        ocp_solver.constraints_set(stage, "ubx", comfort_bounds.T_upper[stage])

    # Solve the OCP
    start_time = time.time()
    print("Solving OCP...")
    _ = ocp_solver.solve_for_x0(
        x0_bar=convert_temperature(np.array([17.1, 17.1, 17.1]), "c", "k")
    )
    print(f"OCP solved in {time.time() - start_time:.5f} seconds")

    solution = get_solution(ocp_solver=ocp_solver)

    fig = plot_ocp_results(
        solution,
        disturbance,
        energy_price_profile,
        comfort_bounds,
        dt=15 * 60,
    )

    plt.show()
