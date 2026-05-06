import casadi as ca


def integrate_erk4(
    f_expl: ca.SX,
    x: ca.SX,
    u: ca.SX,
    p: ca.SX,
    dt: float,
    n_steps: int = 1,
) -> ca.SX:
    """Integrate dynamics using the explicit Runge-Kutta 4 method.

    Args:
        f_expl: The explicit dynamics expression.
        x: The state vector.
        u: The control input vector.
        p: The parameter vector.
        dt: The time step for integration.
        n_steps: Number of integration steps per interval (N). Default: 1.

    Returns:
        The updated state vector after integration.
    """
    assert n_steps >= 1, f"n_steps must be at least 1, got {n_steps}."
    n_stages = 4

    # Butcher tableau for RK4
    b = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
    A = [
        [1 / 2],
        [0, 1 / 2],
        [0, 0, 1],
    ]

    ode = ca.Function("ode", [x, u, p], [f_expl])
    h = dt / n_steps

    xf = x
    for _ in range(n_steps):
        k = []
        for j in range(n_stages):
            x_aug = xf + sum(k[jj] * A[j - 1][jj] for jj in range(j)) if j > 0 else xf
            k.append(h * ode(x_aug, u, p))

        xf = xf + sum(b[j] * k[j] for j in range(n_stages))

    return xf
