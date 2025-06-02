import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import convert_temperature
from scipy.optimize import linprog

from leap_c.examples.hvac.util import (
    BestestHydronicParameters,
    BestestParameters,
    transcribe_discrete_state_space,
)

NX = 3  # Number of state variables (Ti, Th, Te)
NS = 2  # Number of slack variables (delta_l, delta_u)


@dataclass
class ComfortBounds:
    """Temperature comfort bounds over the prediction horizon."""

    T_lower: np.ndarray  # Lower temperature bounds [K] for each time step
    T_upper: np.ndarray  # Upper temperature bounds [K] for each time step

    def __post_init__(self) -> None:
        assert len(self.T_lower) == len(self.T_upper), (
            "Lower and upper bounds must have same length"
        )
        assert np.all(self.T_lower <= self.T_upper), (
            "Lower bounds must be <= upper bounds"
        )


@dataclass
class DisturbanceProfile:
    """Disturbance profile over the prediction horizon."""

    T_outdoor: np.ndarray  # Outdoor temperature [K] for each time step
    solar_radiation: np.ndarray  # Solar radiation [W/m²] for each time step

    def __post_init__(self) -> None:
        assert len(self.T_outdoor) == len(self.solar_radiation), (
            "Outdoor temp and solar radiation must have same length"
        )

    @property
    def horizon_length(self) -> int:
        return len(self.T_outdoor)

    def get_disturbance_vector(self, k: int) -> np.ndarray:
        """Get disturbance vector at time step k."""
        return np.array([self.T_outdoor[k], self.solar_radiation[k]])


@dataclass
class EnergyCostProfile:
    """Energy cost profile over the prediction horizon."""

    costs: np.ndarray  # Energy costs for each time step

    def __post_init__(self) -> None:
        assert np.all(self.costs >= 0), "Energy costs must be non-negative"


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
        energy_costs: EnergyCostProfile,
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
        assert len(energy_costs.costs) >= self.N, (
            f"Energy costs too short: {len(energy_costs.costs)} < {self.N}"
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
        c[control_start_idx : control_start_idx + self.N] = energy_costs.costs[: self.N]

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
        energy_costs: EnergyCostProfile,
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
        c, A_ub, b_ub, A_eq, b_eq, bounds = self.setup_linear_program(
            x0, disturbance_profile, comfort_bounds, energy_costs
        )

        # Solve using scipy.optimize.linprog
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

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


# Example usage and helper functions
def create_constant_disturbance(
    N: int, T_outdoor: float, solar_radiation: float
) -> DisturbanceProfile:
    """Create a constant disturbance profile."""
    return DisturbanceProfile(
        T_outdoor=np.full(N, T_outdoor), solar_radiation=np.full(N, solar_radiation)
    )


def create_constant_comfort_bounds(
    N: int, T_lower: float, T_upper: float
) -> ComfortBounds:
    """Create constant comfort bounds."""
    return ComfortBounds(
        T_lower=np.full(N + 1, T_lower), T_upper=np.full(N + 1, T_upper)
    )


def create_constant_energy_costs(N: int, cost: float) -> EnergyCostProfile:
    """Create constant energy costs."""
    return EnergyCostProfile(costs=np.full(N, cost))


def plot_ocp_results(
    solution: dict[str, Any],
    disturbance_profile: DisturbanceProfile,
    energy_costs: EnergyCostProfile,
    comfort_bounds: ComfortBounds,
    dt: float = 15 * 60,
    figsize: tuple[float, float] = (12, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot the OCP solution results in a figure with three vertically stacked subplots.

    Args:
        solution: Solution dictionary from ThermalControlOCP.solve()
        disturbance_profile: Disturbance profile used in optimization
        comfort_bounds: Comfort bounds used in optimization
        dt: Time step in seconds
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    if not solution["success"]:
        print("Cannot plot: optimization was not successful")
        return None

    # Create time vectors
    N = len(solution["controls"])
    time_hours_states = np.arange(N + 1) * dt / 3600  # For states (N+1 points)
    time_hours_controls = np.arange(N) * dt / 3600  # For controls (N points)
    time_hours_disturbance = (
        np.arange(min(len(disturbance_profile.T_outdoor), N)) * dt / 3600
    )

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        "Thermal Building Control - OCP Solution", fontsize=16, fontweight="bold"
    )

    # Subplot 1: Thermal States
    ax0 = axes[0]

    # Convert temperatures to Celsius for plotting
    Ti_celsius = convert_temperature(
        solution["indoor_temperatures"], "kelvin", "celsius"
    )
    Th_celsius = convert_temperature(
        solution["radiator_temperatures"], "kelvin", "celsius"
    )
    Te_celsius = convert_temperature(
        solution["envelope_temperatures"], "kelvin", "celsius"
    )

    # Plot comfort bounds
    T_lower_celsius = convert_temperature(
        comfort_bounds.T_lower[: N + 1], "kelvin", "celsius"
    )
    T_upper_celsius = convert_temperature(
        comfort_bounds.T_upper[: N + 1], "kelvin", "celsius"
    )

    ax0.fill_between(
        time_hours_states,
        T_lower_celsius,
        T_upper_celsius,
        alpha=0.2,
        color="lightgreen",
        label="Comfort zone",
    )

    # Plot state trajectories
    ax0.plot(
        time_hours_states, Ti_celsius, "b-", linewidth=2, label="Indoor temp. (Ti)"
    )
    ax0.plot(
        time_hours_states,
        Te_celsius,
        "orange",
        linewidth=2,
        label="Envelope temp. (Te)",
    )

    # Plot comfort bounds as dashed lines
    ax0.plot(time_hours_states, T_lower_celsius, "g--", alpha=0.7, label="Lower bound")
    ax0.plot(time_hours_states, T_upper_celsius, "g--", alpha=0.7, label="Upper bound")

    ax0.set_ylabel("Temperature [°C]", fontsize=12)
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Indoor/Envelope Temperature", fontsize=14, fontweight="bold")

    # Subplot 1: Heater Temperature
    ax1 = axes[1]
    ax1.plot(
        time_hours_states, Th_celsius, "r-", linewidth=2, label="Radiator temp. (Th)"
    )
    ax1.set_ylabel("Temperature [°C]", fontsize=12)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Heater Temperature", fontsize=14, fontweight="bold")

    # Subplot 2: Disturbance Signals (twin axes)
    ax2 = axes[2]

    # Outdoor temperature (left y-axis)
    To_celsius = convert_temperature(
        disturbance_profile.T_outdoor[: len(time_hours_disturbance)],
        "kelvin",
        "celsius",
    )
    line1 = ax2.plot(
        time_hours_disturbance, To_celsius, "b-", linewidth=2, label="Outdoor temp."
    )
    ax2.set_ylabel("Outdoor Temperature [°C]", color="b", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="b")

    # Solar radiation (right y-axis)
    ax2_twin = ax2.twinx()
    solar_rad = disturbance_profile.solar_radiation[: len(time_hours_disturbance)]
    line2 = ax2_twin.plot(
        time_hours_disturbance,
        solar_rad,
        "orange",
        linewidth=2,
        label="Solar radiation",
    )
    ax2_twin.set_ylabel("Solar Radiation [W/m²]", color="orange", fontsize=12)
    ax2_twin.tick_params(axis="y", labelcolor="orange")

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="best")

    ax2.grid(True, alpha=0.3)
    ax2.set_title("Disturbance Signals", fontsize=14, fontweight="bold")

    # Subplot 3: Control Input
    ax3 = axes[3]

    # Plot control as step function
    ax3.step(
        time_hours_controls,
        solution["controls"],
        "r-",
        where="post",
        linewidth=2,
        label="Heat input",
    )

    # Add control bounds as horizontal lines
    ax3.axhline(y=0, color="k", linestyle="--", alpha=0.5, label="Min power")

    ax3.set_xlabel("Time [hours]", fontsize=12)
    ax3.set_ylabel("Heat Input [W]", fontsize=12)
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Control Input", fontsize=14, fontweight="bold")

    # Set y-axis lower limit to 0 for better visualization
    ax3.set_ylim(bottom=0)

    ax3_twin = ax3.twinx()
    # Add energy cost as a secondary y-axis
    ax3_twin.step(
        time_hours_controls,
        energy_costs.costs,
        color="orange",
        where="post",
        linewidth=2,
        label="Energy cost (scaled)",
    )
    ax3_twin.set_ylabel("Energy Cost [scaled]", color="orange", fontsize=12)
    ax3_twin.tick_params(axis="y", labelcolor="orange")
    # ax3_twin.legend(loc="lower right")
    ax3_twin.grid(False)  # Disable grid for twin axis
    ax3_twin.set_ylim(bottom=0)  # Set lower limit to 0 for energy cost

    # Adjust layout
    plt.tight_layout()

    # Add summary statistics as text
    total_energy_kWh = solution["controls"].sum() * dt / 3600 / 1000  # Convert to kWh
    max_comfort_violation = max(
        solution["slack_lower"].max(), solution["slack_upper"].max()
    )

    stats_text = (
        f"Total Energy: {total_energy_kWh:.1f} kWh | "
        f"Total Cost: {solution['cost']:.2f} | "
        f"Max Comfort Violation: {max_comfort_violation:.2f} K"
    )

    fig.text(
        0.78,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_comfort_violations(
    solution: dict[str, Any],
    dt: float = 15 * 60,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Plot comfort violations (slack variables) over time.

    Args:
        solution: Solution dictionary from ThermalControlOCP.solve()
        dt: Time step in seconds
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    if not solution["success"]:
        print("Cannot plot: optimization was not successful")
        return None

    N = len(solution["controls"])
    time_hours = np.arange(N + 1) * dt / 3600

    fig, ax = plt.subplots(figsize=figsize)

    # Plot slack variables
    ax.plot(
        time_hours,
        solution["slack_lower"],
        "b-",
        linewidth=2,
        label="Lower bound violation (δ_l)",
    )
    ax.plot(
        time_hours,
        solution["slack_upper"],
        "r-",
        linewidth=2,
        label="Upper bound violation (δ_u)",
    )

    ax.set_xlabel("Time [hours]", fontsize=12)
    ax.set_ylabel("Temperature Violation [K]", fontsize=12)
    ax.set_title("Comfort Constraint Violations", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis lower limit to 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


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

    energy_costs = create_constant_energy_costs(N=96, cost=0.1)  # €/kWh scaled

    rng = np.random.default_rng(seed=42)

    energy_costs.costs = rng.uniform(0.1, 0.2, size=energy_costs.costs.shape[0])

    disturbance.T_outdoor[30:40] = convert_temperature(
        10.0, "celsius", "kelvin"
    )  # Simulate a warm period
    disturbance.solar_radiation[80:90] = 300.0  # Another high solar radiation period

    import time

    # Solve OCP
    print("Solving OCP...")
    start_time = time.time()
    # Solve the OCP
    solution = ocp.solve(x0, disturbance, comfort, energy_costs)

    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")

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
        fig = plot_ocp_results(solution, disturbance, energy_costs, comfort, dt=15 * 60)

        # Also plot comfort violations if there are any
        if solution["slack_lower"].max() > 0.01 or solution["slack_upper"].max() > 0.01:
            fig_violations = plot_comfort_violations(solution, dt=15 * 60)

        plt.show()

    else:
        print(f"Optimization failed: {solution['message']}")
