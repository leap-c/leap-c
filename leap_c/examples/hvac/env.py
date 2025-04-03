from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Constants
DAYLIGHT_START_HOUR = 6  # Start of daylight hours
DAYLIGHT_END_HOUR = 18  # End of daylight hours
MEAN_AMBIENT_TEMPERATURE = 15  # Mean ambient temperature in °C
MAGNITUDE_AMBIENT_TEMPERATURE = 5  # Amplitude of ambient temperature variation in °C
MAGNITUDE_SOLAR_RADIATION = 200  # Peak solar radiation in W/m²


class ThreeStateRcEnv(gym.Env):
    """Simulator for a three-state RC thermal model of a building."""

    def __init__(
        self, params: dict[str, float] | None = None, step_size: float = 60.0
    ) -> None:
        """Initialize the simulator with thermal parameters.

        Args:
            params: Dictionary of thermal parameters
            step_size: Time step for the simulation in seconds

        """
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

        # Initial state variables [°C]
        self.Ti = 19.98391155668  # Indoor temperature
        self.Th = 15.9962126581082  # Radiator temperature
        self.Te = 19.3135718989064  # Envelope temperature

        self.state_0 = np.array([self.Ti, self.Th, self.Te])

        self.integrator = lambda x, u, d, p: rk4_step(dynamics, x, u, d, p, step_size)

    def step(self, action: float, time: float) -> np.ndarray:
        """Perform a simulation step.

        Args:
            action: Control input (heat input to radiator)
            time: Current time in seconds since the start of the simulation
        Returns:
            next_state: Next state after applying the control input

        """
        self.state = self.integrator(
            self.state,
            action,
            (get_ambient_temperature(time), get_solar_radiation(time)),
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
    """Perform a single RK4 step.

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
    """Calculate the state derivatives for the thermal model.

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


def get_ambient_temperature(t) -> float:
    """Get the ambient temperature at time t.

    Args:
        t: Time in seconds since the start of the simulation
    Returns:
        Ambient temperature in °C

    """
    return MEAN_AMBIENT_TEMPERATURE + MAGNITUDE_AMBIENT_TEMPERATURE * np.sin(
        2 * np.pi * t / (24 * 3600)
    )


def get_solar_radiation(t) -> float:
    """Get the solar radiation at time t.

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


def get_heating_power(t) -> float:
    # Example: Heater is on for 1 hour, then off for 1 hour
    if (t // 3600) % 2 == 0:
        return 2000.0  # 2 kW
    return 0.0


# Example usage
if __name__ == "__main__":
    # Create the model

    # Simulation parameters
    time_span = (0, 72 * 3600)  # 24 hours in seconds
    time_step = 5.0  # in seconds
    time = np.arange(time_span[0], time_span[1], time_step)

    env = ThreeStateRcEnv(step_size=time_step)

    env.reset()

    x = [env.state]

    # Run simulation
    for t in time:
        # Get current inputs
        action = get_heating_power(t)

        # Update state using the integrator
        state = env.step(action, t)

        x.append(state)

    # Pop the last state
    x.pop()

    x = np.array(x)

    # Convert x to a named tuple for better readability
    results = {
        "Ti": x[:, 0],  # Indoor temperature
        "Th": x[:, 1],  # Radiator temperature
        "Te": x[:, 2],  # Envelope temperature
        "time": time,  # Time array
        "qh": [get_heating_power(t) for t in time],  # Heating power
        "Ta": [get_ambient_temperature(t) for t in time],  # Outdoor temperature
        "Phi_s": [get_solar_radiation(t) for t in time],  # Solar radiation
    }

    # Plot results
    plt.figure(figsize=(12, 8))

    # Convert time from seconds to hours for plotting
    time_hours = time / 3600

    # Plot temperatures
    plt.subplot(2, 1, 1)
    plt.plot(time_hours, results["Ti"], "b-", label="Indoor Temperature")
    plt.plot(time_hours, results["Th"], "r-", label="Radiator Temperature")
    plt.plot(time_hours, results["Te"], "g-", label="Envelope Temperature")
    plt.plot(
        time_hours,
        [get_ambient_temperature(t) for t in results["time"]],
        "k--",
        label="Outdoor Temperature",
    )
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid()

    # Plot heating power and solar radiation
    plt.subplot(2, 1, 2)
    plt.plot(
        time_hours,
        [get_heating_power(t) for t in results["time"]],
        "r-",
        label="Heating Power (W)",
    )
    plt.plot(
        time_hours,
        [get_solar_radiation(t) for t in results["time"]],
        "y-",
        label="Solar Radiation (W/m²)",
    )
    plt.xlabel("Time (hours)")
    plt.ylabel("Power (W) / Radiation (W/m²)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
