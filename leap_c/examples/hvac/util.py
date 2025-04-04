import casadi as ca
import numpy as np
import scipy
import scipy.linalg


def nominal_ode(x: ca.SX, u: ca.SX, d: ca.SX, p: dict[str, float]) -> np.ndarray:
    """Calculate the state derivatives for the thermal model.

    Args:
        x: Current state vector [Ti, Th, Te]
        u: Heat input to radiator [W]
        d: Disturbance inputs [Ta, Phi_s]
        p: Parameters

    Returns:
        State derivatives [dTi/dt, dTh/dt, dTe/dt]

    """
    Ti, Th, Te = x
    qh = u
    Ta, Phi_s = d

    # State derivatives according to equations (1a)-(1c)
    dTi_dt = (
        (1 / (p["Ci"] * p["Rhi"])) * (Th - Ti)
        + (1 / (p["Ci"] * p["Rie"])) * (Te - Ti)
        + (1 / p["Ci"]) * p["gAw"] * Phi_s
    )
    dTh_dt = (1 / (p["Ch"] * p["Rhi"])) * (Ti - Th) + (qh / p["Ch"])
    dTe_dt = (1 / (p["Ce"] * p["Rie"])) * (Ti - Te) + (1 / (p["Ce"] * p["Rea"])) * (
        Ta - Te
    )

    return np.array([dTi_dt, dTh_dt, dTe_dt])


def transcribe_continuous_state_space(
    Ac: ca.SX | np.ndarray,
    Bc: ca.SX | np.ndarray,
    Ec: ca.SX | np.ndarray,
    params: dict[str, float],
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Create continuous-time state-space matrices Ac, Bc, Ec as per equation (6).

    Args:
        Ac: State-space matrix (system dynamics)
        Bc: State-space matrix (control input)
        Ec: State-space matrix (disturbances)
        params: Dictionary with thermal parameters

    Returns:
        Ac, Bc, Ec: State-space matrices

    """
    # Extract parameters
    Ch = params["Ch"]  # Radiator thermal capacitance
    Ci = params["Ci"]  # Indoor air thermal capacitance
    Ce = params["Ce"]  # Envelope thermal capacitance
    Rhi = params["Rhi"]  # Radiator to indoor air resistance
    Rie = params["Rie"]  # Indoor air to envelope resistance
    Rea = params["Rea"]  # Envelope to outdoor resistance
    gAw = params["gAw"]  # Effective window area

    # Create Ac matrix (system dynamics)
    # Indoor air temperature equation coefficients [Ti, Th, Te]
    Ac[0, 0] = -(1 / (Ci * Rhi) + 1 / (Ci * Rie))  # Ti coefficient
    Ac[0, 1] = 1 / (Ci * Rhi)  # Th coefficient
    Ac[0, 2] = 1 / (Ci * Rie)  # Te coefficient

    # Radiator temperature equation coefficients [Ti, Th, Te]
    Ac[1, 0] = 1 / (Ch * Rhi)  # Ti coefficient
    Ac[1, 1] = -1 / (Ch * Rhi)  # Th coefficient
    Ac[1, 2] = 0  # Te coefficient

    # Envelope temperature equation coefficients [Ti, Th, Te]
    Ac[2, 0] = 1 / (Ce * Rie)  # Ti coefficient
    Ac[2, 1] = 0  # Th coefficient
    Ac[2, 2] = -(1 / (Ce * Rie) + 1 / (Ce * Rea))  # Te coefficient

    # Create Bc matrix (control input)
    Bc[0, 0] = 0  # No direct effect on indoor temperature
    Bc[1, 0] = 1 / Ch  # Effect on radiator temperature
    Bc[2, 0] = 0  # No direct effect on envelope temperature

    # Create Ec matrix (disturbances: outdoor temperature and solar radiation)
    Ec[0, 0] = 0  # No direct effect of outdoor temperature on indoor temp
    Ec[0, 1] = gAw / Ci  # Effect of solar radiation on indoor temperature

    Ec[1, 0] = 0  # No effect of outdoor temp or solar on radiator
    Ec[1, 1] = 0

    Ec[2, 0] = 1 / (Ce * Rea)  # Effect of outdoor temperature on envelope
    Ec[2, 1] = 0  # No direct effect of solar radiation on envelope

    return Ac, Bc, Ec


def transcribe_discrete_state_space(
    Ad: ca.SX | np.ndarray,
    Bd: ca.SX | np.ndarray,
    Ed: ca.SX | np.ndarray,
    dt: float,
    params: dict[str, float],
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Create discrete-time state-space matrices Ad, Bd, Ed as per equation (7).

    Args:
        Ad: State-space matrix (system dynamics)
        Bd: State-space matrix (control input)
        Ed: State-space matrix (disturbances)
        dt: Sampling time
        params: Dictionary with thermal parameters

    Returns:
        Ad, Bd, Ed: Discrete-time state-space matrices

    """
    # Extract type of Ad
    if isinstance(Ad, np.ndarray):
        # Create continuous-time state-space matrices
        Ac, Bc, Ec = transcribe_continuous_state_space(
            Ac=np.zeros((3, 3)),
            Bc=np.zeros((3, 1)),
            Ec=np.zeros((3, 2)),
            params=params,
        )

        # Discretize the continuous-time state-space representation
        Ad = scipy.linalg.expm(Ac * dt)  # Discrete-time state matrix
        Bd = np.linalg.solve(Ac, (Ad - np.eye(3))) @ Bc
        Ed = np.linalg.solve(Ac, (Ad - np.eye(3))) @ Ec

    elif isinstance(Ad, ca.SX):
        # Create continuous-time state-space matrices
        Ac, Bc, Ec = transcribe_continuous_state_space(
            Ac=ca.SX.zeros(3, 3),
            Bc=ca.SX.zeros(3, 1),
            Ec=ca.SX.zeros(3, 2),
            params=params,
        )

        # Discretize the continuous-time state-space representation
        Ad = ca.expm(Ac * dt)  # Discrete-time state matrix
        Bd = ca.mtimes(
            ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Bc
        )  # Discrete-time input matrix
        Ed = ca.mtimes(
            ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Ec
        )  # Discrete-time disturbance matrix

    return Ad, Bd, Ed
