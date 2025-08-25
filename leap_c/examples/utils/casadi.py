import casadi as ca


def integrate_rk4(f_expl: ca.SX, x: ca.SX, u: ca.SX, p: ca.SX, dt: float) -> ca.SX:
    """Integrate dynamics using the RK4 method.

    Args:
        f_expl: The explicit dynamics function.
        x: The state vector.
        u: The control input vector.
        p: The parameter vector.
        dt: The time step for integration.

    Returns:
        The updated state vector after integration.
    """
    ode = ca.Function("ode", [x, u, p], [f_expl])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)
    k3 = ode(x + dt / 2 * k2, u, p)
    k4 = ode(x + dt * k3, u, p)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
