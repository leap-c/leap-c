from dataclasses import dataclass, asdict
import numpy as np


@dataclass(kw_only=True)
class PointMassParams:
    # Mass and friction parameters
    m: float  # mass
    cx: float  # x friction coefficient
    cy: float  # y friction coefficient

    # Cost function parameters
    q_diag: np.ndarray  # state cost diagonal
    r_diag: np.ndarray  # control cost diagonal
    q_diag_e: np.ndarray  # terminal state cost diagonal

    # Reference parameters
    xref: np.ndarray  # state reference
    uref: np.ndarray  # control reference
    xref_e: np.ndarray  # terminal state reference


def make_default_pointmass_params() -> PointMassParams:
    """Returns a PointMassParams instance with default parameter values."""
    return PointMassParams(
        # Mass and friction parameters
        m=1.0,
        cx=0.1,
        cy=0.1,

        # Cost function parameters
        q_diag=np.array([1.0, 1.0, 1.0, 1.0]),
        r_diag=np.array([0.1, 0.1]),
        q_diag_e=np.array([1.0, 1.0, 1.0, 1.0]),

        # Reference parameters
        xref=np.array([0.0, 0.0, 0.0, 0.0]),
        uref=np.array([0.0, 0.0]),
        xref_e=np.array([0.0, 0.0, 0.0, 0.0]),
    )
