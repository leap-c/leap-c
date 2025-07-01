from dataclasses import dataclass
from leap_c.ocp.acados.parameters import Parameter
import numpy as np


@dataclass(kw_only=True)
class CartPoleParams:
    # Dynamics parameters
    M: Parameter  # mass of the cart [kg]
    m: Parameter  # mass of the ball [kg]
    g: Parameter  # gravity constant [m/s^2]
    l: Parameter  # length of the rod [m]

    # Cost matrix factorization parameters
    L11: Parameter
    L22: Parameter
    L33: Parameter
    L44: Parameter
    L55: Parameter
    Lloweroffdiag: Parameter

    # Linear cost parameters (for EXTERNAL cost)
    c1: Parameter  # position linear cost
    c2: Parameter  # theta linear cost
    c3: Parameter  # v linear cost
    c4: Parameter  # thetadot linear cost
    c5: Parameter  # u linear cost

    # Reference parameters (for NONLINEAR_LS cost)
    xref1: Parameter  # reference position
    xref2: Parameter  # reference theta
    xref3: Parameter  # reference v
    xref4: Parameter  # reference thetadot
    uref: Parameter  # reference u


def make_default_cartpole_params(stage_wise: bool = False) -> CartPoleParams:
    """Returns a CartPoleParams instance with default parameter values."""
    return CartPoleParams(
        # Dynamics parameters
        M=Parameter("M", np.array([1.0])),
        m=Parameter("m", np.array([0.1])),
        g=Parameter("g", np.array([9.81])),
        l=Parameter("l", np.array([0.8])),
        # Cost matrix factorization parameters
        L11=Parameter("L11", np.array([np.sqrt(2e3)])),
        L22=Parameter("L22", np.array([np.sqrt(2e3)])),
        L33=Parameter("L33", np.array([np.sqrt(1e-2)])),
        L44=Parameter("L44", np.array([np.sqrt(1e-2)])),
        L55=Parameter("L55", np.array([np.sqrt(2e-1)])),
        Lloweroffdiag=Parameter("Lloweroffdiag", np.array([0.0] * (4 + 3 + 2 + 1))),
        # Linear cost parameters (for EXTERNAL cost)
        c1=Parameter("c1", np.array([0.0])),
        c2=Parameter("c2", np.array([0.0])),
        c3=Parameter("c3", np.array([0.0])),
        c4=Parameter("c4", np.array([0.0])),
        c5=Parameter("c5", np.array([0.0])),
        # Reference parameters (for NONLINEAR_LS cost)
        xref1=Parameter("xref1", np.array([0.0])),
        xref2=Parameter(
            "xref2",
            np.array([0.0]),
            lower_bound=-2.0 * np.pi,
            upper_bound=2.0 * np.pi,
            differentiable=True,
            stage_wise=stage_wise,
        ),
        xref3=Parameter("xref3", np.array([0.0])),
        xref4=Parameter("xref4", np.array([0.0])),
        uref=Parameter("uref", np.array([0.0])),
    )
