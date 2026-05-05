import casadi as ca


def integrate_erk4(
    f_expl: ca.SX,
    x: ca.SX,
    u: ca.SX,
    p: ca.SX,
    dt: float,
    num_substeps: int = 1,
) -> ca.SX:
    """Integrate dynamics using the explicit RK4 method.

    Args:
        f_expl: The explicit dynamics function.
        x: The state vector.
        u: The control input vector.
        p: The parameter vector.
        dt: The time step for integration.
        num_substeps: Number of internal RK4 sub-steps over ``dt``. Matches acados'
            ``sim_method_num_steps`` for parity with an ``ERK`` integrator.

    Returns:
        The updated state vector after integration.
    """
    ode = ca.Function("ode", [x, u, p], [f_expl])
    h = dt / num_substeps
    x_next = x
    for _ in range(num_substeps):
        k1 = ode(x_next, u, p)
        k2 = ode(x_next + h / 2 * k1, u, p)
        k3 = ode(x_next + h / 2 * k2, u, p)
        k4 = ode(x_next + h * k3, u, p)
        x_next = x_next + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next
