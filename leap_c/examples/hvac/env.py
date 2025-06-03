from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.constants import convert_temperature

from leap_c.examples.hvac.util import (
    BestestHydronicParameters,
    BestestParameters,
    transcribe_continuous_state_space,
    transcribe_discrete_state_space,
)

# Constants
DAYLIGHT_START_HOUR = 6
DAYLIGHT_END_HOUR = 18
MEAN_AMBIENT_TEMPERATURE = convert_temperature(0, "celsius", "kelvin")
MAGNITUDE_AMBIENT_TEMPERATURE = 5
MAGNITUDE_SOLAR_RADIATION = 200


class StochasticThreeStateRcEnv(gym.Env):
    """
    Simulator for a three-state RC thermal model with exact discretization of Gaussian noise.

    This environment uses the matrix exponential approach to exactly discretize both the
    deterministic dynamics and the stochastic noise terms.
    """

    def __init__(
        self,
        params: None | BestestParameters = None,
        step_size: float = 900.0,  # Default 15 minutes
        ambient_temperature_function: Callable | None = None,
        solar_radiation_function: Callable | None = None,
        enable_noise: bool = True,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the stochastic simulator.

        Args:
            params: Dictionary of thermal parameters
            step_size: Time step for the simulation in seconds
            ambient_temperature_function: Function for ambient temperature
            solar_radiation_function: Function for solar radiation
            enable_noise: Whether to include stochastic noise
            random_seed: Random seed for reproducibility
        """
        # Store parameters
        self.params = (
            params if params is not None else BestestHydronicParameters().to_dict()
        )

        self.step_size = step_size
        self.enable_noise = enable_noise

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initial state variables [K]
        self.Ti = convert_temperature(20.0, "celsius", "kelvin")
        self.Th = convert_temperature(20.0, "celsius", "kelvin")
        self.Te = convert_temperature(20.0, "celsius", "kelvin")
        self.state_0 = np.array([self.Ti, self.Th, self.Te])

        # Precompute discrete-time matrices including noise covariance
        self.Ad, self.Bd, self.Ed, self.Qd = self._compute_discrete_matrices()

        # Set default functions if not provided
        if ambient_temperature_function is None:
            self.ambient_temperature_function = self._get_default_ambient_temperature
        else:
            self.ambient_temperature_function = ambient_temperature_function

        if solar_radiation_function is None:
            self.solar_radiation_function = self._get_default_solar_radiation
        else:
            self.solar_radiation_function = solar_radiation_function

    def _compute_discrete_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute discrete-time matrices using exact discretization via matrix exponential.
        This includes both deterministic dynamics and noise covariance.
        """
        # Get continuous-time matrices
        Ac, Bc, Ec = transcribe_continuous_state_space(
            Ac=np.zeros((3, 3)),
            Bc=np.zeros((3, 1)),
            Ec=np.zeros((3, 2)),
            params=self.params,
        )

        # Compute discrete-time state-space matrices
        Ad, Bd, Ed = transcribe_discrete_state_space(
            Ad=np.zeros((3, 3)),
            Bd=np.zeros((3, 1)),
            Ed=np.zeros((3, 2)),
            dt=self.step_size,
            params=self.params,
        )

        # Create noise intensity matrix Σ from parameters
        # The stochastic terms are σᵢω̇ᵢ, σₕω̇ₕ, σₑω̇ₑ
        sigma_i = np.exp(self.params["sigmai"])
        sigma_h = np.exp(self.params["sigmah"])
        sigma_e = np.exp(self.params["sigmae"])

        Qd = self._compute_noise_covariance(
            Ac=Ac,
            Sigma=np.diag([sigma_i, sigma_h, sigma_e]),
            dt=self.step_size,
        )

        return Ad, Bd, Ed, Qd

    def _compute_noise_covariance(
        self, Ac: np.ndarray, Sigma: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        TODO: Check if this is correct. See, e.g., Farrell, J. Sec 4.7.2
        Compute the exact discrete-time noise covariance matrix using matrix exponential.
        Q_d = ∫₀^Δt e^(Aτ) Σ Σᵀ e^(Aᵀτ) dτ.

        Args:
            Ac: Continuous-time system matrix
            Sigma: Noise intensity matrix (diagonal)
            dt: Sampling time

        Returns:
            Qd: Discrete-time noise covariance matrix
        """
        n = Ac.shape[0]  # State dimension (3)

        # Create the augmented matrix for computing the noise covariance integral
        # [ A    Σ Σᵀ ]
        # [ 0      -Aᵀ ]
        SigmaSigmaT = Sigma @ Sigma.T

        # Augmented matrix (6x6)
        M = np.block([[Ac, SigmaSigmaT], [np.zeros((n, n)), -Ac.T]])

        # Matrix exponential of augmented system
        exp_M = scipy.linalg.expm(M * dt)

        # Extract the noise covariance from the upper-right block
        # Qd = e^(A*dt) * (upper-right block of exp_M)
        Ad = exp_M[:n, :n]
        Phi = exp_M[:n, n:]

        # The discrete-time covariance is Qd = Ad @ Phi
        return Ad @ Phi

    def step(self, action: np.ndarray, time: float) -> np.ndarray:
        """
        Perform a simulation step with exact discrete-time dynamics including noise.

        Args:
            action: Control input (heat input to radiator)
            time: Current time in seconds since the start of the simulation

        Returns:
            next_state: Next state after applying control input and noise
        """
        # Get exogenous inputs
        exog = np.array(
            [
                self.ambient_temperature_function(time),
                self.solar_radiation_function(time),
            ]
        )

        # Deterministic state update
        x_next = self.Ad @ self.state + self.Bd @ action + self.Ed @ exog

        # Add Gaussian noise if enabled
        if self.enable_noise:
            # Sample from multivariate normal distribution with exact covariance
            noise = np.random.default_rng().multivariate_normal(
                mean=np.zeros(3), cov=self.Qd
            )
            x_next += noise

        self.state = x_next
        return self.state

    def reset(self, state_0: np.ndarray | None = None) -> None:
        """Reset the model state to initial values."""
        if state_0 is None:
            state_0 = self.state_0
        self.state = state_0.copy()

    def get_noise_statistics(self) -> dict:
        """
        Get statistics about the noise model.

        Returns:
            Dictionary with noise statistics
        """
        sigma_i = np.exp(self.params["sigmai"])
        sigma_h = np.exp(self.params["sigmah"])
        sigma_e = np.exp(self.params["sigmae"])

        return {
            "continuous_noise_intensities": {
                "sigma_i": sigma_i,
                "sigma_h": sigma_h,
                "sigma_e": sigma_e,
            },
            "discrete_noise_covariance": self.Qd,
            "discrete_noise_std": np.sqrt(np.diag(self.Qd)),
            "step_size": self.step_size,
        }

    def _get_default_ambient_temperature(self, t: float) -> float:
        """Get default ambient temperature function value."""
        MEAN_AMBIENT_TEMPERATURE = convert_temperature(0, "celsius", "kelvin")
        MAGNITUDE_AMBIENT_TEMPERATURE = 5
        return MEAN_AMBIENT_TEMPERATURE + MAGNITUDE_AMBIENT_TEMPERATURE * np.sin(
            2 * np.pi * t / (24 * 3600)
        )

    def _get_default_solar_radiation(self, t: float) -> float:
        """Get default solar radiation function value."""
        DAYLIGHT_START_HOUR = 6
        DAYLIGHT_END_HOUR = 18
        MAGNITUDE_SOLAR_RADIATION = 200

        hour = (t % (24 * 3600)) / 3600
        if DAYLIGHT_START_HOUR <= hour <= DAYLIGHT_END_HOUR:
            return MAGNITUDE_SOLAR_RADIATION * np.sin(np.pi * (hour - 6) / 12)
        return 0.0


def get_alternating_heating_power(t) -> float:
    if (t // 3600) % 10 == 0:
        return np.array([2000.0])  # 2 kW
    return np.array([0.0])


def get_constant_heating_power(t: float) -> float:
    # Example: Constant heating power
    return np.array([2000.0])


# Example usage
def plot_hvac_results(
    get_ambient_temperature: Callable,
    get_solar_radiation: Callable,
    get_heating_power: Callable,
    time: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
) -> plt.Figure:
    """
    Plot the results of an HVAC simulation, including temperatures, heating power,
    and solar radiation over time.

    Parameters:
    -----------
    get_ambient_temperature : callable
        A function that takes a time value (in seconds) and returns the ambient
        (outdoor) temperature at that time.
    get_solar_radiation : callable
        A function that takes a time value (in seconds) and returns the solar
        radiation (in W/m²) at that time.
    get_heating_power : callable
        A function that takes a time value (in seconds) and returns the heating
        power (in W) at that time.
    time : numpy.ndarray
        A 1D array of time values (in seconds) corresponding to the simulation time steps.
    x : numpy.ndarray
        A 2D array where each row corresponds to the state variables at a given time step:
        - x[:, 0]: Indoor temperature (°C)
        - x[:, 1]: Radiator temperature (°C)
        - x[:, 2]: Envelope temperature (°C)
    u : numpy.ndarray
        A 1D array of heating power values (in W) at each time step.

    Returns:
    --------
    None
        This function does not return any value. It generates and displays plots
        of the simulation results.

    Notes:
    ------
    - The time values are converted from seconds to hours for plotting.
    - The function creates two subplots:
        1. Temperatures (indoor, radiator, envelope, and outdoor) over time.
        2. Heating power and solar radiation over time.
    - The plots include legends, grid lines, and appropriate labels for clarity.
    """
    results = {
        "Ti": convert_temperature(x[:, 0], "kelvin", "celsius"),  # Indoor temperature
        "Th": convert_temperature(x[:, 1], "kelvin", "celsius"),  # Radiator temperature
        "Te": convert_temperature(x[:, 2], "kelvin", "celsius"),  # Envelope temperature
        "Ta": convert_temperature(
            np.array([get_ambient_temperature(t) for t in time]), "kelvin", "celsius"
        ),  # Outdoor temperature
        "time": time,  # Time array
        "qh": [get_heating_power(t) for t in time],  # Heating power
        "Phi_s": [get_solar_radiation(t) for t in time],  # Solar radiation
    }

    dt = time[1] - time[0]  # Time step in seconds:w

    # Plot results
    plt.figure(figsize=(12, 8))
    # Convert time from seconds to hours for plotting
    time_hours = time / 3600

    # Plot temperatures (first subplot - unchanged)
    plt.subplot(3, 1, 1)
    plt.plot(time_hours, results["Ti"], "b-", label="Indoor Temperature")
    plt.plot(time_hours, results["Te"], "g-", label="Envelope Temperature")
    plt.plot(time_hours, results["Ta"], "k--", label="Outdoor Temperature")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time_hours, results["Th"], "r-", label="Radiator Temperature")
    plt.legend()
    plt.ylabel("Temperature (°C)")
    plt.grid()

    # Plot heating power and solar radiation (second subplot - with twin axes)
    ax1 = plt.subplot(3, 1, 3)
    # Primary y-axis (left) for heating power
    (line1,) = ax1.plot(
        time_hours,
        u,
        "r-",
        label="Heating Power",
    )
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Heating Power (W)", color="r")
    ax1.tick_params(axis="y", labelcolor="r")
    ax1.grid()

    # Secondary y-axis (right) for solar radiation
    ax2 = ax1.twinx()
    solar_data = [get_solar_radiation(t) for t in results["time"]]
    (line2,) = ax2.plot(
        time_hours,
        solar_data,
        "y-",
        label="Solar Radiation",
    )
    ax2.set_ylabel("Solar Radiation (W/m²)", color="y")
    ax2.tick_params(axis="y", labelcolor="y")

    # Create a combined legend for the second subplot
    ax1.legend([line1, line2], ["Heating Power", "Solar Radiation"], loc="upper left")

    plt.xlabel("Time (hours)")
    plt.tight_layout()

    # Add super title with dt
    plt.suptitle(f"HVAC Simulation Results (dt = {dt:.2f} seconds)", fontsize=16)

    # Save the figure
    plt.savefig(f"hvac_simulation_results_{int(dt)}.png")

    return plt.gcf()


if __name__ == "__main__":
    # Create the model

    # Simulation parameters

    # ambient_temperate_function = get_ambient_temperature
    # solar_radiation_function = get_solar_radiation
    ambient_temperate_function = lambda t: convert_temperature(10, "c", "k")
    solar_radiation_function = lambda t: 0.0  # No solar radiation for simplicity

    heating_power_function = get_alternating_heating_power

    for time_step in [900.0]:
        days = 7
        time_span = (0, days * 24 * 3600)
        time = np.arange(time_span[0], time_span[1], time_step)

        env = StochasticThreeStateRcEnv(
            step_size=time_step,
            ambient_temperature_function=ambient_temperate_function,
            solar_radiation_function=solar_radiation_function,
            enable_noise=True,
        )

        env.reset()

        x = [env.state]
        u = []

        # Run simulation
        for t in time:
            # Get current inputs
            action = heating_power_function(t)

            # Update state using the integrator
            state = env.step(action, t)

            x.append(state)
            u.append(action)

        # Pop the last state
        x.pop()

        x = np.array(x)
        u = np.array(u)

        fig = plot_hvac_results(
            ambient_temperate_function,
            solar_radiation_function,
            heating_power_function,
            time,
            x,
            u,
        )

    plt.show()
