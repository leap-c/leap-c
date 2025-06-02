import time
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import convert_temperature
from scipy.optimize import linprog

from leap_c.examples.hvac.util import (
    BestestHydronicParameters,
    BestestParameters,
    ComfortBounds,
    DisturbanceProfile,
    EnergyPriceProfile,
    create_constant_comfort_bounds,
    create_constant_disturbance,
    create_constant_energy_price,
    plot_comfort_violations,
    plot_ocp_results,
    transcribe_discrete_state_space,
)

NX = 3  # Number of state variables (Ti, Th, Te)
NS = 2  # Number of slack variables (delta_l, delta_u)


class ThermalControlLpSolver:
    """
    Optimal Control Problem solver for thermal building control.

    Solves the linear program defined in equation (8) of the PDF:
    - Minimizes energy costs and comfort violations
    - Subject to thermal dynamics and comfort constraints
    """

    def __init__(
        self,
        params: BestestParameters,
        dt: float = 15 * 60,  # 15 minutes in seconds
        N: int = 96,  # 24-hour horizon with 15-min intervals
        u_min: float = 0.0,  # Minimum heat input [W]
        u_max: float = 2000.0,  # Maximum heat input [W]
        rho_l: float = 1e4,  # Penalty for lower comfort bound violation
        rho_u: float = 1e4,  # Penalty for upper comfort bound violation
    ) -> None:
        self.params = params.to_dict()
        self.dt = dt
        self.N = N
        self.u_min = u_min
        self.u_max = u_max
        self.rho_l = rho_l
        self.rho_u = rho_u

        # Compute discrete-time state-space matrices
        self.Ad, self.Bd, self.Ed = transcribe_discrete_state_space(
            Ad=np.zeros((3, 3)),
            Bd=np.zeros((3, 1)),
            Ed=np.zeros((3, 2)),
            dt=dt,
            params=self.params,
        )

        # Flatten matrices for easier use
        self.Bd = self.Bd.flatten()

    def setup_linear_program(
        self,
        x0: np.ndarray,
        disturbance_profile: DisturbanceProfile,
        comfort_bounds: ComfortBounds,
        energy_prices: EnergyPriceProfile,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Set up the linear program matrices for the OCP.

        Returns:
            c: Cost vector
            A_ub: Inequality constraint matrix
            b_ub: Inequality constraint bounds
            bounds: Variable bounds
        """
        assert disturbance_profile.horizon_length >= self.N, (
            f"Disturbance profile too short: {disturbance_profile.horizon_length} < {self.N}"
        )
        assert len(comfort_bounds.T_lower) >= self.N + 1, (
            f"Comfort bounds too short: {len(comfort_bounds.T_lower)} < {self.N + 1}"
        )
        assert len(energy_prices.price) >= self.N, (
            f"Energy costs too short: {len(energy_prices.price)} < {self.N}"
        )

        # Decision variables: [x0, x1, ..., xN, delta_l0, ..., delta_lN, delta_u0, ..., delta_uN, u0, ..., uN-1]
        # State variables: 3*(N+1) = 3*97 = 291
        # Slack variables: 2*(N+1) = 2*97 = 194
        # Control variables: N = 96
        # Total variables: 291 + 194 + 96 = 581

        n_states = NX * (self.N + 1)
        n_slack = NS * (self.N + 1)
        n_controls = self.N
        n_vars = n_states + n_slack + n_controls

        # Cost vector
        c = np.zeros(n_vars)

        # Energy costs for controls
        control_start_idx = n_states + n_slack
        c[control_start_idx : control_start_idx + self.N] = energy_prices.price[
            : self.N
        ]

        # Slack variable penalties
        slack_l_start_idx = n_states
        slack_u_start_idx = n_states + (self.N + 1)
        c[slack_l_start_idx : slack_l_start_idx + (self.N + 1)] = self.rho_l
        c[slack_u_start_idx : slack_u_start_idx + (self.N + 1)] = self.rho_u

        print("n_vars:", n_vars)
        print("n_states:", n_states)
        print("n_slack:", n_slack)
        print("n_controls:", n_controls)
        print("Control start index:", control_start_idx)
        print("Slack lower start index:", slack_l_start_idx)
        print("Slack upper start index:", slack_u_start_idx)

        # Constraint matrices
        # Variable bounds
        bounds = [(None, None)] * n_vars

        # State variables: no explicit bounds
        for i in range(n_states):
            bounds[i] = (None, None)

        # Slack variables: non-negative
        for i in range(n_slack):
            bounds[n_states + i] = (0, None)

        # Control variables: bounded
        for i in range(n_controls):
            bounds[control_start_idx + i] = (self.u_min, self.u_max)

        # Equality constraints for dynamics: x_{k+1} = Ad*x_k + Bd*u_k + Ed*d_k
        A_eq_list = []
        b_eq_list = []

        for k in range(self.N):
            # Get indices
            x_k_idx = NX * k
            x_k1_idx = NX * (k + 1)
            u_k_idx = control_start_idx + k

            # Create constraint: x_{k+1} - Ad*x_k - Bd*u_k = Ed*d_k
            constraint_row = np.zeros((NX, n_vars))

            # x_{k+1} coefficient: I
            constraint_row[:, x_k1_idx : x_k1_idx + NX] = np.eye(NX)

            # x_k coefficient: -Ad
            constraint_row[:, x_k_idx : x_k_idx + NX] = -self.Ad

            # u_k coefficient: -Bd
            constraint_row[:, u_k_idx] = -self.Bd

            # right-hand side: Ed*d_k
            d_k = disturbance_profile.get_disturbance_vector(k)
            rhs = self.Ed @ d_k

            A_eq_list.append(constraint_row)
            b_eq_list.append(rhs)

        # Initial state constraint: x_0 = x0
        constraint_row = np.zeros((NX, n_vars))
        constraint_row[:, :3] = np.eye(NX)
        A_eq_list.append(constraint_row.reshape(NX, -1))
        b_eq_list.append(x0)

        # Comfort constraints as inequalities:
        # T_i_k <= T_upper_k + delta_u_k  =>  T_i_k - delta_u_k <= T_upper_k
        # T_i_k >= T_lower_k - delta_l_k  =>  -T_i_k - delta_l_k <= -T_lower_k

        A_ub_list = []
        b_ub_list = []

        for k in range(self.N + 1):
            Ti_idx = NX * k  # Indoor temperature is first state
            delta_l_idx = slack_l_start_idx + k
            delta_u_idx = slack_u_start_idx + k

            # Upper bound constraint: Ti_k - delta_u_k <= T_upper_k
            constraint_row = np.zeros(n_vars)
            constraint_row[Ti_idx] = 1.0  # Ti coefficient
            constraint_row[delta_u_idx] = -1.0  # delta_u coefficient
            A_ub_list.append(constraint_row)
            b_ub_list.append(comfort_bounds.T_upper[k])

            # Lower bound constraint: -Ti_k - delta_l_k <= -T_lower_k
            constraint_row = np.zeros(n_vars)
            constraint_row[Ti_idx] = -1.0  # Ti coefficient
            constraint_row[delta_l_idx] = -1.0  # delta_l coefficient
            A_ub_list.append(constraint_row)
            b_ub_list.append(-comfort_bounds.T_lower[k])

        # Convert to matrices
        if A_eq_list:
            A_eq = np.vstack(
                [A.reshape(-1, n_vars) if A.ndim > 1 else A for A in A_eq_list]
            )
            b_eq = np.hstack(b_eq_list)
        else:
            A_eq = None
            b_eq = None

        if A_ub_list:
            A_ub = np.vstack(A_ub_list)
            b_ub = np.array(b_ub_list)
        else:
            A_ub = None
            b_ub = None

        return c, A_ub, b_ub, A_eq, b_eq, bounds

    def solve(
        self,
        x0: np.ndarray,
        disturbance_profile: DisturbanceProfile,
        comfort_bounds: ComfortBounds,
        energy_costs: EnergyPriceProfile,
        method: str = "highs",
    ) -> dict[str, Any]:
        """
        Solve the optimal control problem.

        Args:
            x0: Initial state [Ti, Th, Te] in Kelvin
            disturbance_profile: Disturbance profile over horizon
            comfort_bounds: Temperature comfort bounds
            energy_costs: Energy cost profile
            method: Solver method ('highs' or 'revised simplex')

        Returns:
            Dictionary containing solution results
        """
        # Validate inputs
        assert x0.shape == (NX,), (
            f"Initial state must have shape ({NX},), got {x0.shape}"
        )

        # Set up linear program
        start_time = time.time()
        print("Setting up linear program...")
        c, A_ub, b_ub, A_eq, b_eq, bounds = self.setup_linear_program(
            x0, disturbance_profile, comfort_bounds, energy_costs
        )
        print(
            f"Linear program setup completed in {time.time() - start_time:.5f} seconds"
        )

        # Solve using scipy.optimize.linprog
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            print("Solving linear program...")
            start_time = time.time()
            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=method,
                options={"disp": False},
            )
            print(f"Linear program solved in {time.time() - start_time:.5f} seconds")

        if not result.success:
            print(f"Optimization failed: {result.message}")
            return {
                "success": False,
                "message": result.message,
                "states": None,
                "controls": None,
                "slack_lower": None,
                "slack_upper": None,
                "cost": np.inf,
            }

        # Extract solution
        x_opt = result.x

        # Parse solution
        n_states = NX * (self.N + 1)
        n_slack = NS * (self.N + 1)

        # States: reshape to (N+1, 3)
        states = x_opt[:n_states].reshape((self.N + 1, NX))

        # Slack variables
        slack_lower = x_opt[n_states : n_states + (self.N + 1)]
        slack_upper = x_opt[n_states + (self.N + 1) : n_states + n_slack]

        # Controls
        controls = x_opt[n_states + n_slack :]

        return {
            "success": True,
            "message": result.message,
            "states": states,  # [Ti, Th, Te] for each time step
            "controls": controls,  # Heat input for each time step
            "slack_lower": slack_lower,  # Lower bound violations
            "slack_upper": slack_upper,  # Upper bound violations
            "cost": result.fun,
            "indoor_temperatures": states[:, 0],  # Extract Ti trajectory
            "radiator_temperatures": states[:, 1],  # Extract Th trajectory
            "envelope_temperatures": states[:, 2],  # Extract Te trajectory
        }


if __name__ == "__main__":
    # Parameters
    params = BestestHydronicParameters()

    # Create OCP solver
    ocp = ThermalControlLpSolver(
        params=params,
        dt=15 * 60,  # 15 minutes
        N=96,  # 24-hour horizon
        u_min=0.0,
        u_max=2000.0,
        rho_l=1e4,
        rho_u=1e4,
    )

    # Initial state: [Ti=20°C, Th=50°C, Te=15°C]
    x0 = convert_temperature(np.array([20.0, 50.0, 15.0]), "celsius", "kelvin")

    # Create profiles
    disturbance = create_constant_disturbance(
        N=96,
        T_outdoor=convert_temperature(5.0, "celsius", "kelvin"),
        solar_radiation=100.0,
    )

    comfort = create_constant_comfort_bounds(
        N=96,
        T_lower=convert_temperature(19.0, "celsius", "kelvin"),
        T_upper=convert_temperature(23.0, "celsius", "kelvin"),
    )

    energy_prices = create_constant_energy_price(N=96, cost=0.1)  # €/kWh scaled

    rng = np.random.default_rng(seed=42)

    energy_prices.price = rng.uniform(0.1, 0.2, size=energy_prices.price.shape[0])

    disturbance.T_outdoor[30:40] = convert_temperature(
        10.0, "celsius", "kelvin"
    )  # Simulate a warm period
    disturbance.solar_radiation[80:90] = 300.0  # Another high solar radiation period

    # Solve OCP
    print("Solving OCP...")
    start_time = time.time()
    # Solve the OCP
    solution = ocp.solve(x0, disturbance, comfort, energy_prices)

    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.5f} seconds")

    if solution["success"]:
        print("Optimization successful!")
        print(f"Total cost: {solution['cost']:.2f}")
        print(
            f"Indoor temperature range: {convert_temperature(solution['indoor_temperatures'].min(), 'kelvin', 'celsius'):.1f}°C to {convert_temperature(solution['indoor_temperatures'].max(), 'kelvin', 'celsius'):.1f}°C"
        )
        print(
            f"Total heating energy: {solution['controls'].sum() * 15 * 60 / 1000:.1f} kJ"
        )
        print(f"Max comfort violation (lower): {solution['slack_lower'].max():.2f} K")
        print(f"Max comfort violation (upper): {solution['slack_upper'].max():.2f} K")

        # Plot results
        print("\nGenerating plots...")
        fig = plot_ocp_results(
            solution, disturbance, energy_prices, comfort, dt=15 * 60
        )

        # Also plot comfort violations if there are any
        if solution["slack_lower"].max() > 0.01 or solution["slack_upper"].max() > 0.01:
            fig_violations = plot_comfort_violations(solution, dt=15 * 60)

        plt.show()

    else:
        print(f"Optimization failed: {solution['message']}")
