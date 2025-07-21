from dataclasses import dataclass

import numpy as np

from leap_c.ocp.acados.parameters import Parameter


@dataclass(kw_only=True)
class ChainParams:
    L: Parameter  # rest length of spring
    D: Parameter  # spring constant
    C: Parameter  # damping constant
    m: Parameter  # mass of the balls
    w: Parameter  # disturbance on intermediate balls
    q_sqrt_diag: Parameter  # weight on state
    r_sqrt_diag: Parameter  # weight on control inputs
    fix_point: Parameter  # fixed point for the chain (the anchor point)
    phi_range: Parameter  # range for phi angle initialization (min, max)
    theta_range: Parameter  # range for theta angle initialization (min, max)


def make_default_chain_params(n_mass: int = 3, stagewise: bool = False) -> ChainParams:
    """Returns a ChainParams instance with default parameter values."""
    q_sqrt_diag = np.ones(3 * (n_mass - 1) + 3 * (n_mass - 2))
    r_sqrt_diag = 1e-1 * np.ones(3)

    return ChainParams(
        # Dynamics parameters (fixed)
        L=Parameter("L", np.repeat([0.033, 0.033, 0.033], n_mass - 1)),
        D=Parameter("D", np.repeat([1.0, 1.0, 1.0], n_mass - 1)),
        C=Parameter("C", np.repeat([0.1, 0.1, 0.1], n_mass - 1)),
        m=Parameter("m", np.repeat([0.033], n_mass - 1)),
        w=Parameter("w", np.repeat([0.0, 0.0, 0.0], n_mass - 2)),
        # Cost parameters (learnable)
        q_sqrt_diag=Parameter(
            "q_sqrt_diag",
            q_sqrt_diag,
            lower_bound=0.5 * q_sqrt_diag,
            upper_bound=1.5 * q_sqrt_diag,
            differentiable=True,
            stagewise=stagewise,
            fix=False,
        ),
        r_sqrt_diag=Parameter(
            "r_sqrt_diag",
            r_sqrt_diag,
            lower_bound=0.5 * r_sqrt_diag,
            upper_bound=1.5 * r_sqrt_diag,
            differentiable=True,
            stagewise=stagewise,
            fix=False,
        ),
        # General parameters
        fix_point=Parameter("fix_point", np.zeros(3)),
        phi_range=Parameter("phi_range", np.array([np.pi / 6, np.pi / 3])),
        theta_range=Parameter("theta_range", np.array([-np.pi / 4, np.pi / 4])),
    )
