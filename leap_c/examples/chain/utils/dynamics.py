"""Chain dynamics functions."""

from typing import Callable

import casadi as ca
from casadi import SX, norm_2, vertcat
from casadi.tools.structure3 import ssymStruct
import numpy as np


def get_f_expl_expr(
    x: ssymStruct,
    u: ca.SX,
    p: dict[str, np.ndarray | ca.SX],
    x0: ca.SX = ca.SX.zeros(3),
) -> ca.SX:
    """CasADi symbolic chain dynamics.

    This version accepts parameters as a dictionary for compatibility with
    the RestingChainSolver.

    Args:
        x: State vector containing positions and velocities
        u: Control input (velocity of last mass)
        p: Parameter dictionary with keys ["m", "D", "L", "C", "w"]
        x0: Fixed point position (anchor)

    Returns:
        State derivative as CasADi expression
    """
    n_masses = p["m"].shape[0] + 1

    xpos = vertcat(*x["pos"])
    xvel = vertcat(*x["vel"])

    # Force on intermediate masses
    f = SX.zeros(3 * (n_masses - 2), 1)

    # Gravity force on intermediate masses
    for i in range(int(f.shape[0] / 3)):
        f[3 * i + 2] = -9.81

    n_link = n_masses - 1

    # Spring force
    for i in range(n_link):
        if i == 0:
            dist = xpos[i * 3 : (i + 1) * 3] - x0
        else:
            dist = xpos[i * 3 : (i + 1) * 3] - xpos[(i - 1) * 3 : i * 3]

        F = ca.SX.zeros(3, 1)
        for j in range(F.shape[0]):
            F[j] = (
                p["D"][i + j] / p["m"][i] * (1 - p["L"][i + j] / norm_2(dist)) * dist[j]
            )

        # mass on the right
        if i < n_link - 1:
            f[i * 3 : (i + 1) * 3] -= F

        # mass on the left
        if i > 0:
            f[(i - 1) * 3 : i * 3] += F

    # Damping force
    for i in range(n_link):
        if i == 0:
            vel = xvel[i * 3 : (i + 1) * 3]
        elif i == n_link - 1:
            vel = u - xvel[(i - 1) * 3 : i * 3]
        else:
            vel = xvel[i * 3 : (i + 1) * 3] - xvel[(i - 1) * 3 : i * 3]

        F = ca.SX.zeros(3, 1)
        # Damping force
        for j in range(3):
            F[j] = p["C"][i + j] * ca.norm_1(vel[j]) * vel[j]

        # mass on the right
        if i < n_masses - 2:
            f[i * 3 : (i + 1) * 3] -= F

        # mass on the left
        if i > 0:
            f[(i - 1) * 3 : i * 3] += F

    # Disturbance on intermediate masses
    for i in range(n_masses - 2):
        f[i * 3 : (i + 1) * 3] += p["w"][i]

    return vertcat(xvel, u, f)


def rk4_integrator_casadi(
    f_expl_expr: ca.SX,
    x: ca.SX,
    u: ca.SX,
    p: ca.SX,
    dt: float,
) -> ca.SX:
    """Runge-Kutta 4th order integrator for CasADi symbolic dynamics.

    Args:
        f_expl_expr: CasADi expression for the explicit dynamics (dx/dt = f(x, u, p))
        x: State variable (CasADi SX)
        u: Control input (CasADi SX)
        p: Parameters (CasADi SX)
        dt: Time step

    Returns:
        Next state after time step dt using RK4 integration
    """
    ode = ca.Function("ode", [x, u, p], [f_expl_expr])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)
    k3 = ode(x + dt / 2 * k2, u, p)
    k4 = ode(x + dt * k3, u, p)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def create_discrete_numpy_dynamics(n_mass: int, dt: float) -> Callable:
    """Create discrete-time NumPy dynamics from CasADi implementation.

    Args:
        n_mass: Number of masses in the chain
        dt: Time step for integration

    Returns:
        A function with signature f(x, u, p, fix_point) -> x_next
    """
    from casadi.tools import entry, struct_symSX

    # Create symbolic variables
    x = struct_symSX(
        [
            entry("pos", shape=(3, 1), repeat=n_mass - 1),
            entry("vel", shape=(3, 1), repeat=n_mass - 2),
        ]
    )

    u = ca.SX.sym("u", 3, 1)

    # Parameter dictionary
    p_dict = {
        "m": ca.SX.sym("m", n_mass - 1),
        "D": ca.SX.sym("D", 3 * (n_mass - 1)),
        "L": ca.SX.sym("L", 3 * (n_mass - 1)),
        "C": ca.SX.sym("C", 3 * (n_mass - 1)),
        "w": ca.SX.sym("w", 3 * (n_mass - 2)),
    }

    fix_point = ca.SX.sym("fix_point", 3)

    # Get the continuous dynamics expression
    f_expr = get_f_expl_expr(x, u, p_dict, fix_point)

    # Create CasADi integrator using built-in RK4
    all_params = ca.vertcat(
        p_dict["m"], p_dict["D"], p_dict["L"], p_dict["C"], p_dict["w"]
    )

    # Define the ODE system
    ode = {"x": x.cat, "p": ca.vertcat(u, all_params, fix_point), "ode": f_expr}

    # Create integrator with RK4 method
    integrator = ca.integrator("integrator", "rk", ode, {"tf": dt})

    def discrete_dynamics(
        x: np.ndarray,
        u: np.ndarray,
        p: dict[str, np.ndarray],
        fix_point: np.ndarray | None = None,
    ) -> np.ndarray:
        """Discrete-time NumPy dynamics using CasADi's RK4 integrator.

        Args:
            x: State vector [pos1, pos2, ..., vel1, vel2, ...]
            u: Control input (velocity of last mass)
            p: Parameter dictionary with keys ["m", "D", "L", "C", "w"]
            fix_point: Fixed point position (anchor)

        Returns:
            Next state after time step dt
        """
        if fix_point is None:
            fix_point = np.zeros(3)

        # Flatten and concatenate parameters in the expected order
        p_flat = np.concatenate(
            [
                p["m"].flatten(),
                p["D"].flatten(),
                p["L"].flatten(),
                p["C"].flatten(),
                p["w"].flatten(),
            ]
        )

        # Combine control, parameters, and fix_point for the integrator
        p_combined = np.concatenate([u.flatten(), p_flat, fix_point.flatten()])

        # Call the CasADi integrator
        result = integrator(x0=x, p=p_combined)
        return np.array(result["xf"]).flatten()

    return discrete_dynamics
