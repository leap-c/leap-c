from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import convert_temperature

from dataclasses import dataclass

from leap_c.examples.hvac.util import (
    BestestParameters,
    BestestHydronicParameters,
    BestestHydronicHeatpumpParameters,
)

# Constants
DAYLIGHT_START_HOUR = 6
DAYLIGHT_END_HOUR = 18
MEAN_AMBIENT_TEMPERATURE = convert_temperature(0, "celsius", "kelvin")
MAGNITUDE_AMBIENT_TEMPERATURE = 5
MAGNITUDE_SOLAR_RADIATION = 200


class ThreeStateRcEnv(gym.Env):
    """Simulator for a three-state RC thermal model of a building."""

    def __init__(
        self,
        params: None | BestestParameters = None,
        step_size: float = 60.0,
        ambient_temperature_function: Callable | None = None,
        solar_radiation_function: Callable | None = None,
    ) -> None:
        """
        Initialize the simulator with thermal parameters.

        Args:
            params: Dictionary of thermal parameters
            step_size: Time step for the simulation in seconds

        """
        # Store parameters
        self.params = (
            params if params is not None else BestestHydronicParameters().to_dict()
        )

        # Initial state variables [°C]
        self.Ti = convert_temperature(20.0, "celsius", "kelvin")
        self.Th = convert_temperature(20.0, "celsius", "kelvin")
        self.Te = convert_temperature(20.0, "celsius", "kelvin")

        self.state_0 = np.array([self.Ti, self.Th, self.Te])

        self.integrator = lambda x, u, d, p: rk4_step(dynamics, x, u, d, p, step_size)

        if ambient_temperature_function is None:
            self.ambient_temperature_function = get_ambient_temperature
        else:
            self.ambient_temperature_function = ambient_temperature_function

        if solar_radiation_function is None:
            self.solar_radiation_function = get_solar_radiation
        else:
            self.solar_radiation_function = solar_radiation_function

    def step(self, action: float, time: float) -> np.ndarray:
        """
        Perform a simulation step.

        Args:
            action: Control input (heat input to radiator)
            time: Current time in seconds since the start of the simulation
        Returns:
            next_state: Next state after applying the control input

        """
        self.state = self.integrator(
            self.state,
            action,
            (
                self.ambient_temperature_function(time),
                self.solar_radiation_function(time),
            ),
            self.params,
        )

        return self.state

    def reset(self, state_0: np.ndarray | None = None) -> None:
        """Reset the model state to initial values."""
        if state_0 is None:
            state_0 = self.state_0
        self.state = state_0


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


def dynamics(x: Iterable, u: float, d: Iterable, p: dict[str, float]) -> np.ndarray:
    """
    Calculate the state derivatives for the thermal model.

    Args:
        x: Current state vector [Ti, Th, Te]
        u: Heat input to radiator [W]
        d: Disturbance inputs [Ta, Phi_s]
        p: Parameters

    Returns:
        State derivatives [dTi/dt, dTh/dt, dTe/dt]

    """
    Ti, Th, Te = x
    qh = u
    Ta, Phi_s = d

    # TODO: Use envrionment random number generator
    # State derivatives according to equations (1a)-(1c)
    dTi_dt = (
        (1 / (p["Ci"] * p["Rhi"])) * (Th - Ti)
        + (1 / (p["Ci"] * p["Rie"])) * (Te - Ti)
        + (1 / p["Ci"]) * p["gAw"] * Phi_s
        + np.exp(p["sigmai"]) * np.random.randn()
    )
    dTh_dt = (
        (1 / (p["Ch"] * p["Rhi"])) * (Ti - Th)
        + (qh / p["Ch"])
        + np.exp(p["sigmah"]) * np.random.randn()
    )
    dTe_dt = (
        (1 / (p["Ce"] * p["Rie"])) * (Ti - Te)
        + (1 / (p["Ce"] * p["Rea"])) * (Ta - Te)
        + np.exp(p["sigmae"]) * np.random.randn()
    )

    return np.array([dTi_dt, dTh_dt, dTe_dt])


def get_ambient_temperature(t: float) -> float:
    """
    Get the ambient temperature at time t.

    Args:
        t: Time in seconds since the start of the simulation
    Returns:
        Ambient temperature in °C

    """
    return MEAN_AMBIENT_TEMPERATURE + MAGNITUDE_AMBIENT_TEMPERATURE * np.sin(
        2 * np.pi * t / (24 * 3600)
    )


def get_solar_radiation(t: float) -> float:
    """
    Get the solar radiation at time t.

    Args:
        t: Time in seconds since the start of the simulation
    Returns:
        Solar radiation in W/m²

    """
    hour = (t % (24 * 3600)) / 3600
    if DAYLIGHT_START_HOUR <= hour <= DAYLIGHT_END_HOUR:  # Daylight hours
        return MAGNITUDE_SOLAR_RADIATION * np.sin(
            np.pi * (hour - 6) / 12
        )  # Peak at noon
    return 0.0  # Night


def get_alternating_heating_power(t) -> float:
    # Example: Heater is on for 1 hour, then off for 1 hour
    if (t // 3600) % 2 == 0:
        return 200.0  # 2 kW
    return 0.0


def get_constant_heating_power(t: float) -> float:
    # Example: Constant heating power
    return 2000.0


# Example usage
def plot_hvac_results(
    get_ambient_temperature: Callable,
    get_solar_radiation: Callable,
    get_heating_power: Callable,
    time: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
) -> None:
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
    plt.show()


if __name__ == "__main__":
    # Create the model

    # Simulation parameters

    days = 3
    time_span = (0, days * 24 * 3600)
    time_step = 5.0  # in seconds
    time = np.arange(time_span[0], time_span[1], time_step)

    ambient_temperate_function = get_ambient_temperature
    solar_radiation_function = get_solar_radiation
    heating_power_function = get_constant_heating_power

    env = ThreeStateRcEnv(
        step_size=time_step,
        ambient_temperature_function=ambient_temperate_function,
        solar_radiation_function=solar_radiation_function,
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

    plot_hvac_results(
        ambient_temperate_function,
        solar_radiation_function,
        heating_power_function,
        time,
        x,
        u,
    )
