"""Chain dynamics functions."""

import casadi as ca
import numpy as np
from casadi import SX, norm_2, vertcat
from casadi.tools import entry, struct_symSX

from ..utils.casadi import integrate_erk4


def define_f_expl_expr(
    x: struct_symSX,
    u: ca.SX,
    p: dict[str, np.ndarray | ca.SX],
    fix_point: ca.SX = ca.SX.zeros(3),
) -> ca.SX:
    """Define CasADi symbolic chain dynamics.

    This version accepts parameters as a dictionary for compatibility with
    the RestingChainSolver.

    Args:
        x: State vector containing positions and velocities
        u: Control input (velocity of last mass)
        p: Parameter dictionary with keys ["m", "D", "L", "C", "w"]
        fix_point: Fixed point position (anchor)

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
            dist = xpos[i * 3 : (i + 1) * 3] - fix_point
        else:
            dist = xpos[i * 3 : (i + 1) * 3] - xpos[(i - 1) * 3 : i * 3]

        F = ca.SX.zeros(3, 1)
        for j in range(F.shape[0]):
            F[j] = p["D"][i + j] / p["m"][i] * (1 - p["L"][i + j] / norm_2(dist)) * dist[j]

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


def create_discrete_casadi_dynamics(n_mass: int, dt: float) -> ca.Function:
    """Create a discrete dynamics function for the chain system.

    Args:
        n_mass: Number of masses in the chain.
        dt: Time step for the discrete dynamics.

    Returns:
        A CasADi function representing the discrete dynamics.
    """
    x = struct_symSX(
        [
            entry("pos", shape=(3, 1), repeat=n_mass - 1),
            entry("vel", shape=(3, 1), repeat=n_mass - 2),
        ]
    )
    u = ca.SX.sym("u", 3, 1)

    p = {
        "m": ca.SX.sym("m", n_mass - 1),
        "D": ca.SX.sym("D", 3 * (n_mass - 1)),
        "L": ca.SX.sym("L", 3 * (n_mass - 1)),
        "C": ca.SX.sym("C", 3 * (n_mass - 1)),
        "w": ca.SX.sym("w", 3 * (n_mass - 2)),
    }

    f_expl = define_f_expl_expr(x=x, u=u, p=p)

    p_cat = ca.vertcat(*p.values())

    disc_dyn_expr = integrate_erk4(
        f_expl,
        x.cat,
        u,
        p_cat,
        dt,
    )

    return ca.Function(
        "disc_dyn",
        [x.cat, u, *p.values()],
        [disc_dyn_expr],
        ["x", "u", *p.keys()],
        ["x_next"],
    )
