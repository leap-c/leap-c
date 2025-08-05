from dataclasses import dataclass

import numpy as np

from leap_c.ocp.acados.parameters import Parameter


@dataclass(kw_only=True)
class PointMassParams:
    # Mass and friction parameters
    m: Parameter  # mass
    cx: Parameter  # x friction coefficient
    cy: Parameter  # y friction coefficient

    # Cost function parameters
    q_sqrt_diag: Parameter  # state cost diagonal (sqrt of the diagonal)
    r_sqrt_diag: Parameter  # control cost diagonal (sqrt of the diagonal)

    # Reference parameters
    xref: Parameter  # state reference
    uref: Parameter  # control reference


def make_default_pointmass_params(stagewise: bool = False) -> PointMassParams:
    """Returns a PointMassParams instance with default parameter values."""
    q_sqrt_diag = np.array([1.0, 1.0, 1.0, 1.0])
    r_sqrt_diag = np.array([0.1, 0.1])

    return PointMassParams(
        # Mass and friction parameters
        m=Parameter("m", np.array([1.0])),
        cx=Parameter("cx", np.array([0.1])),
        cy=Parameter("cy", np.array([0.1])),
        # Cost function parameters
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
            # lower_bound=0.5 * r_sqrt_diag,
            # upper_bound=1.5 * r_sqrt_diag,
            # differentiable=True,
            # stagewise=stagewise,
            # fix=False,
        ),
        # Reference parameters
        xref=Parameter(
            "xref",
            np.array([0.0, 0.0, 0.0, 0.0]),
            lower_bound=np.array([0.0, 0.0, -20, -20]),
            upper_bound=np.array([4.0, 1.0, 20, 20]),
            fix=False,
            differentiable=True,
            stagewise=stagewise,
        ),
        uref=Parameter(
            "uref",
            np.array([0.0, 0.0]),
            lower_bound=np.array([-10.0, -10.0]),
            upper_bound=np.array([10.0, 10.0]),
            fix=False,
            differentiable=True,
            stagewise=stagewise,
        ),
    )
