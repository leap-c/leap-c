from dataclasses import dataclass, asdict
import numpy as np
from typing import List
from leap_c.ocp.acados.parameters import Parameter


@dataclass(kw_only=True)
class CartPoleParams:
    # Dynamics parameters
    M: np.ndarray  # mass of the cart [kg]
    m: np.ndarray  # mass of the ball [kg]
    g: np.ndarray  # gravity constant [m/s^2]
    l: np.ndarray  # length of the rod [m]

    # Cost matrix factorization parameters
    q_diag: Parameter
    r_diag: Parameter

    # Reference parameters (for NONLINEAR_LS cost)
    xref1: np.ndarray  # reference position
    xref2: np.ndarray  # reference theta
    xref3: np.ndarray  # reference v
    xref4: np.ndarray  # reference thetadot
    uref: np.ndarray  # reference u


def make_default_cartpole_params() -> CartPoleParams:
    """Returns a CartPoleParams instance with default parameter values."""
    return CartPoleParams(
        # Dynamics parameters
        M=np.array([1.0]),
        m=np.array([0.1]),
        g=np.array([9.81]),
        l=np.array([0.8]),
        # Cost matrix factorization parameters
        q_diag=Parameter("q_diag", np.sqrt(np.array([2e3, 2e3, 1e-2, 1e-2]))),
        r_diag=Parameter("r_diag", np.sqrt(np.array([2e-1]))),
        # Reference parameters (for NONLINEAR_LS cost)
        xref1=np.array([0.0]),
        xref2=np.array([0.0]),
        xref3=np.array([0.0]),
        xref4=np.array([0.0]),
        uref=np.array([0.0]),
    )
