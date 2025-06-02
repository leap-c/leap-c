from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
from scipy.constants import convert_temperature


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
class EnergyPriceProfile:
    """Energy price profile over the prediction horizon."""

    price: np.ndarray  # Energy price for each time step

    def __post_init__(self) -> None:
        assert np.all(self.price >= 0), "Energy price must be non-negative"


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


def create_constant_energy_price(N: int, cost: float) -> EnergyPriceProfile:
    """Create constant energy costs."""
    return EnergyPriceProfile(price=np.full(N, cost))


def plot_ocp_results(
    solution: dict[str, Any],
    disturbance_profile: DisturbanceProfile,
    energy_prices: EnergyPriceProfile,
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
    ax0.step(
        time_hours_states, Ti_celsius, "b-", linewidth=2, label="Indoor temp. (Ti)"
    )
    ax0.step(
        time_hours_states,
        Te_celsius,
        "orange",
        linewidth=2,
        label="Envelope temp. (Te)",
    )

    # Plot comfort bounds as dashed lines
    ax0.step(time_hours_states, T_lower_celsius, "g--", alpha=0.7, label="Lower bound")
    ax0.step(time_hours_states, T_upper_celsius, "g--", alpha=0.7, label="Upper bound")

    ax0.set_ylabel("Temperature [°C]", fontsize=12)
    ax0.legend(loc="best")
    ax0.grid(visible=True, alpha=0.3)
    ax0.set_title("Indoor/Envelope Temperature", fontsize=14, fontweight="bold")

    # Subplot 1: Heater Temperature
    ax1 = axes[1]
    ax1.step(
        time_hours_states, Th_celsius, "b-", linewidth=2, label="Radiator temp. (Th)"
    )
    ax1.set_ylabel("Temperature [°C]", fontsize=12)
    ax1.grid(visible=True, alpha=0.3)
    ax1.set_title("Heater Temperature", fontsize=14, fontweight="bold")

    # Subplot 2: Disturbance Signals (twin axes)
    ax2 = axes[2]

    # Outdoor temperature (left y-axis)
    To_celsius = convert_temperature(
        disturbance_profile.T_outdoor[: len(time_hours_disturbance)],
        "kelvin",
        "celsius",
    )
    ax2.plot(
        time_hours_disturbance, To_celsius, "b-", linewidth=2, label="Outdoor temp."
    )
    ax2.set_ylabel("Outdoor Temperature [°C]", color="b", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="b")

    # Solar radiation (right y-axis)
    ax2_twin = ax2.twinx()
    solar_rad = disturbance_profile.solar_radiation[: len(time_hours_disturbance)]
    ax2_twin.plot(
        time_hours_disturbance,
        solar_rad,
        "orange",
        linewidth=2,
        label="Solar radiation",
    )
    ax2_twin.set_ylabel("Solar Radiation [W/m²]", color="orange", fontsize=12)
    ax2_twin.tick_params(axis="y", labelcolor="orange")

    ax2.grid(visible=True, alpha=0.3)
    ax2.set_title("Exogeneous Signals", fontsize=14, fontweight="bold")

    # Subplot 3: Control Input
    ax3 = axes[3]

    # Plot control as step function
    ax3.step(
        time_hours_controls,
        solution["controls"],
        "b-",
        where="post",
        linewidth=2,
        label="Heat input",
    )

    ax3.set_xlabel("Time [hours]", fontsize=12)
    ax3.set_ylabel("Heat Input [W]", color="b", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Control Input", fontsize=14, fontweight="bold")

    # Set y-axis lower limit to 0 for better visualization
    ax3.set_ylim(bottom=0)

    ax3_twin = ax3.twinx()
    # Add energy cost as a secondary y-axis
    ax3_twin.step(
        time_hours_controls,
        energy_prices.price,
        color="orange",
        where="post",
        linewidth=2,
        label="Energy cost (scaled)",
    )
    ax3_twin.set_ylabel("Energy Price [EUR/kWh]", color="orange", fontsize=12)
    ax3_twin.tick_params(axis="y", labelcolor="orange")
    ax3_twin.grid(visible=False)  # Disable grid for twin axis
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
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.8},
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


@dataclass
class BestestParameters:
    """Base class for hydronic system parameters."""

    # Effective window area [m²]
    gAw: float  # noqa: N815

    # Thermal capacitances [J/K]
    Ch: float  # Heating system thermal capacity
    Ci: float  # Indoor thermal capacity
    Ce: float  # External thermal capacity

    # Noise parameters
    e11: float  # Measurement noise
    sigmai: float
    sigmah: float
    sigmae: float

    # Thermal resistances [K/W]
    Rea: float  # Resistance external-ambient
    Rhi: float  # Resistance heating-indoor
    Rie: float  # Resistance indoor-external

    # Heater parameters
    eta: float  # Efficiency for electric heater

    def to_dict(self) -> dict[str, float]:
        """Convert parameters to a dictionary with string keys and float values."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, params_dict: dict[str, float]) -> "BestestParameters":
        """Create an instance from a dictionary."""
        return cls(**params_dict)


@dataclass
class BestestHydronicParameters(BestestParameters):
    """Standard hydronic system parameters."""

    gAw: float = 10.1265729225269  # noqa: N815
    Ch: float = 4015.39425109821
    Ci: float = 1914908.30860716
    Ce: float = 15545663.6743828
    e11: float = -9.49409438095981
    sigmai: float = -37.8538482163307
    sigmah: float = -50.4867241844347
    sigmae: float = -5.57887704511886
    Rea: float = 0.00751396226986365
    Rhi: float = 0.0761996125919563
    Rie: float = 0.00135151763922409
    eta: float = 0.98


@dataclass
class BestestHydronicHeatpumpParameters(BestestParameters):
    """Heat pump system parameters for a hydronic heating system."""

    gAw: float = 40.344131392192  # noqa: N815
    Ch: float = 10447262.2318648
    Ci: float = 14827137.0377258
    Ce: float = 50508258.9032192
    e11: float = -30.0936560706053
    sigmai: float = -23.3175423490014
    sigmah: float = -19.5274067368137
    sigmae: float = -5.07591222090641
    Rea: float = 0.00163027389197229
    Rhi: float = 0.000437603769897038
    Rie: float = 0.000855786902577802
    eta: float = 0.98


def get_f_expl_expr(
    x: ca.SX | np.ndarray,
    u: ca.SX | np.ndarray,
    e: ca.SX | np.ndarray,
    params: dict[str, float],
) -> ca.SX | np.ndarray:
    """
    Get the state derivatives.

    Args:
        x: Current state vector [Ti, Th, Te]
        u: Heat input to radiator [W]
        e: Exogeneous inputs [Ta, Phi_s]
        params: Parameters
    Returns:
        State derivatives [dTi/dt, dTh/dt, dTe/dt]
    """
    assert x.shape in ((3,), (3, 1)), "State vector x must have shape (3,) or (3, 1)."
    assert e.shape in ((2,), (2, 1)), "Ex. in. vector e must have shape (2,) or (2, 1)."

    Ac = np.zeros((3, 3)) if isinstance(x, np.ndarray) else ca.SX.zeros(3, 3)
    Bc = np.zeros((3, 1)) if isinstance(u, np.ndarray) else ca.SX.zeros(3, 1)
    Ec = np.zeros((3, 2)) if isinstance(e, np.ndarray) else ca.SX.zeros(3, 2)

    Ac, Bc, Ec = transcribe_continuous_state_space(Ac=Ac, Bc=Bc, Ec=Ec, params=params)

    if isinstance(x, np.ndarray):
        return Ac @ x + Bc @ u + Ec @ e

    return ca.mtimes(Ac, x) + ca.mtimes(Bc, u) + ca.mtimes(Ec, e)


def get_disc_dyn_expr(
    x: ca.SX | np.ndarray,
    u: ca.SX | np.ndarray,
    e: ca.SX | np.ndarray,
    dt: float,
    params: dict[str, float],
) -> ca.SX | np.ndarray:
    assert x.shape in ((3,), (3, 1)), "State vector x must have shape (3,) or (3, 1)."
    assert u.shape in ((1,), (1, 1)), "Control input u must have shape (1,) or (1, 1)."
    assert e.shape in ((2,), (2, 1)), "Ex. in. vector e must have shape (2,) or (2, 1)."
    assert dt > 0, "Sampling time dt must be positive."

    Ad = np.zeros((3, 3)) if isinstance(x, np.ndarray) else ca.SX.zeros(3, 3)
    Bd = np.zeros((3, 1)) if isinstance(u, np.ndarray) else ca.SX.zeros(3, 1)
    Ed = np.zeros((3, 2)) if isinstance(e, np.ndarray) else ca.SX.zeros(3, 2)

    Ad, Bd, Ed = transcribe_discrete_state_space(
        Ad=Ad, Bd=Bd, Ed=Ed, dt=dt, params=params
    )

    if isinstance(x, np.ndarray):
        return Ad @ x + Bd @ u + Ed @ e

    return ca.mtimes(Ad, x) + ca.mtimes(Bd, u) + ca.mtimes(Ed, e)


def rk4_step(
    f: Callable,
    x: Iterable,
    u: float,
    d: Iterable,
    p: dict[str, float],
    h: float,
) -> np.ndarray:
    """
    Perform a single RK4 step.

    Args:
        f: Function to integrate
        x: Current state
        u: Control input
        d: Disturbance input
        p: Parameters
        h: Step size
    Returns:
        Next state

    """
    k1 = f(x, u, d, p)
    k2 = f(x + 0.5 * h * k1, u, d, p)
    k3 = f(x + 0.5 * h * k2, u, d, p)
    k4 = f(x + h * k3, u, d, p)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def transcribe_continuous_state_space(
    Ac: ca.SX | np.ndarray,
    Bc: ca.SX | np.ndarray,
    Ec: ca.SX | np.ndarray,
    params: dict[str, float],
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """
    Create continuous-time state-space matrices Ac, Bc, Ec as per equation (6).

    Args:
        Ac: State-space matrix (system dynamics)
        Bc: State-space matrix (control input)
        Ec: State-space matrix (disturbances)
        params: Dictionary with thermal parameters

    Returns:
        Ac, Bc, Ec: State-space matrices

    """
    # Extract parameters
    Ch = params["Ch"]  # Radiator thermal capacitance
    Ci = params["Ci"]  # Indoor air thermal capacitance
    Ce = params["Ce"]  # Envelope thermal capacitance
    Rhi = params["Rhi"]  # Radiator to indoor air resistance
    Rie = params["Rie"]  # Indoor air to envelope resistance
    Rea = params["Rea"]  # Envelope to outdoor resistance
    gAw = params["gAw"]  # Effective window area

    # Create Ac matrix (system dynamics)
    # Indoor air temperature equation coefficients [Ti, Th, Te]
    Ac[0, 0] = -(1 / (Ci * Rhi) + 1 / (Ci * Rie))  # Ti coefficient
    Ac[0, 1] = 1 / (Ci * Rhi)  # Th coefficient
    Ac[0, 2] = 1 / (Ci * Rie)  # Te coefficient

    # Radiator temperature equation coefficients [Ti, Th, Te]
    Ac[1, 0] = 1 / (Ch * Rhi)  # Ti coefficient
    Ac[1, 1] = -1 / (Ch * Rhi)  # Th coefficient
    Ac[1, 2] = 0  # Te coefficient

    # Envelope temperature equation coefficients [Ti, Th, Te]
    Ac[2, 0] = 1 / (Ce * Rie)  # Ti coefficient
    Ac[2, 1] = 0  # Th coefficient
    Ac[2, 2] = -(1 / (Ce * Rie) + 1 / (Ce * Rea))  # Te coefficient

    # Create Bc matrix (control input)
    Bc[0, 0] = 0  # No direct effect on indoor temperature
    Bc[1, 0] = 1 / Ch  # Effect on radiator temperature
    Bc[2, 0] = 0  # No direct effect on envelope temperature

    # Create Ec matrix (disturbances: outdoor temperature and solar radiation)
    Ec[0, 0] = 0  # No direct effect of outdoor temperature on indoor temp
    Ec[0, 1] = gAw / Ci  # Effect of solar radiation on indoor temperature

    Ec[1, 0] = 0  # No effect of outdoor temp or solar on radiator
    Ec[1, 1] = 0

    Ec[2, 0] = 1 / (Ce * Rea)  # Effect of outdoor temperature on envelope
    Ec[2, 1] = 0  # No direct effect of solar radiation on envelope

    return Ac, Bc, Ec


def transcribe_discrete_state_space(
    Ad: ca.SX | np.ndarray,
    Bd: ca.SX | np.ndarray,
    Ed: ca.SX | np.ndarray,
    dt: float,
    params: dict[str, float],
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """
    Create discrete-time state-space matrices Ad, Bd, Ed as per equation (7).

    Args:
        Ad: State-space matrix (system dynamics)
        Bd: State-space matrix (control input)
        Ed: State-space matrix (disturbances)
        dt: Sampling time
        params: Dictionary with thermal parameters

    Returns:
        Ad, Bd, Ed: Discrete-time state-space matrices

    """
    # Extract type of Ad
    if isinstance(Ad, np.ndarray):
        # Create continuous-time state-space matrices
        Ac, Bc, Ec = transcribe_continuous_state_space(
            Ac=np.zeros((3, 3)),
            Bc=np.zeros((3, 1)),
            Ec=np.zeros((3, 2)),
            params=params,
        )

        # Discretize the continuous-time state-space representation
        Ad = scipy.linalg.expm(Ac * dt)  # Discrete-time state matrix
        Bd = np.linalg.solve(Ac, (Ad - np.eye(3))) @ Bc
        Ed = np.linalg.solve(Ac, (Ad - np.eye(3))) @ Ec

    elif isinstance(Ad, ca.SX):
        # Create continuous-time state-space matrices
        Ac, Bc, Ec = transcribe_continuous_state_space(
            Ac=ca.SX.zeros(3, 3),
            Bc=ca.SX.zeros(3, 1),
            Ec=ca.SX.zeros(3, 2),
            params=params,
        )

        # Discretize the continuous-time state-space representation
        Ad = ca.expm(Ac * dt)  # Discrete-time state matrix
        Bd = ca.mtimes(
            ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Bc
        )  # Discrete-time input matrix
        Ed = ca.mtimes(
            ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Ec
        )  # Discrete-time disturbance matrix

    return Ad, Bd, Ed


if __name__ == "__main__":
    params = BestestHydronicParameters().to_dict()
    dt = 6 * 300.0  # 5 min sampling time

    # Define a discrete integrator function
    Ad, Bd, Ed = transcribe_discrete_state_space(
        Ad=np.zeros((3, 3)),
        Bd=np.zeros((3, 1)),
        Ed=np.zeros((3, 2)),
        dt=dt,
        params=params,
    )

    def discrete_step(x: np.ndarray, u: np.ndarray, e: np.ndarray) -> np.ndarray:
        return Ad @ x + Bd @ u + Ed @ e

    def continuous_step(x: np.ndarray, u: np.ndarray, e: np.ndarray) -> np.ndarray:
        return rk4_step(f=get_f_expl_expr, x=x, u=u, d=e, p=params, h=dt)

    # Initial state
    x = convert_temperature(np.array([20.0, 50.0, 15.0]), "celsius", "kelvin")

    # Heat input to radiator in W
    u = np.array([1000.0])

    # Exogeneous inputs: outdoor temperature and solar radiation
    e = np.array([convert_temperature(10.0, "celsius", "kelvin"), 200.0])  # [Ta, Phi_s]

    print("Initial state:", x)

    # Perform a discrete step
    x_next_discrete = discrete_step(x, u, e)
    print("Next state (discrete step):", x_next_discrete)
    # Perform a continuous step using RK4
    x_next_continuous = continuous_step(x, u, e)
    print("Next state (continuous step):", x_next_continuous)
    # Compare the results
    print(
        "Difference between discrete and continuous steps:",
        x_next_discrete - x_next_continuous,
    )

    # Use
