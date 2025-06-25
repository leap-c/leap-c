from dataclasses import dataclass, asdict
import numpy as np
from typing import List


@dataclass(kw_only=True)
class CartPoleParams:
    # Dynamics parameters
    M: np.ndarray         # mass of the cart [kg]
    m: np.ndarray         # mass of the ball [kg]
    g: np.ndarray         # gravity constant [m/s^2]
    l: np.ndarray         # length of the rod [m]

    # Cost matrix factorization parameters
    L11: np.ndarray
    L22: np.ndarray
    L33: np.ndarray
    L44: np.ndarray
    L55: np.ndarray
    Lloweroffdiag: np.ndarray

    # Linear cost parameters (for EXTERNAL cost)
    c1: np.ndarray        # position linear cost
    c2: np.ndarray        # theta linear cost
    c3: np.ndarray        # v linear cost
    c4: np.ndarray        # thetadot linear cost
    c5: np.ndarray        # u linear cost

    # Reference parameters (for NONLINEAR_LS cost)
    xref1: np.ndarray     # reference position
    xref2: np.ndarray     # reference theta
    xref3: np.ndarray     # reference v
    xref4: np.ndarray     # reference thetadot
    uref: np.ndarray      # reference u


def make_default_cartpole_params() -> CartPoleParams:
    """Returns a CartPoleParams instance with default parameter values."""
    return CartPoleParams(
        # Dynamics parameters
        M=np.array([1.0]),
        m=np.array([0.1]),
        g=np.array([9.81]),
        l=np.array([0.8]),

        # Cost matrix factorization parameters
        L11=np.array([np.sqrt(2e3)]),
        L22=np.array([np.sqrt(2e3)]),
        L33=np.array([np.sqrt(1e-2)]),
        L44=np.array([np.sqrt(1e-2)]),
        L55=np.array([np.sqrt(2e-1)]),
        Lloweroffdiag=np.array([0.0] * (4 + 3 + 2 + 1)),

        # Linear cost parameters (for EXTERNAL cost)
        c1=np.array([0.0]),
        c2=np.array([0.0]),
        c3=np.array([0.0]),
        c4=np.array([0.0]),
        c5=np.array([0.0]),

        # Reference parameters (for NONLINEAR_LS cost)
        xref1=np.array([0.0]),
        xref2=np.array([0.0]),
        xref3=np.array([0.0]),
        xref4=np.array([0.0]),
        uref=np.array([0.0]),
    )
