from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import casadi as ca
import numpy as np
import scipy
import scipy.linalg


@dataclass
class BestestParameters:
    """Base class for hydronic system parameters."""

    # Effective window area [m²]
    gAw: float  # noqa: N815

    # Thermal capacitances [J/K]
    Ch: float  # Heating system thermal capacity
    Ci: float  # Indoor thermal capacity
    Ce: float  # External thermal capacity

    # Noise parameters
    e11: float  # Measurement noise
    sigmai: float
    sigmah: float
    sigmae: float

    # Thermal resistances [K/W]
    Rea: float  # Resistance external-ambient
    Rhi: float  # Resistance heating-indoor
    Rie: float  # Resistance indoor-external

    # Heater parameters
    eta: float  # Efficiency for electric heater

    def to_dict(self) -> dict[str, float]:
        """Convert parameters to a dictionary with string keys and float values."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, params_dict: dict[str, float]) -> "BestestParameters":
        """Create an instance from a dictionary."""
        return cls(**params_dict)


@dataclass
class BestestHydronicParameters(BestestParameters):
    """Standard hydronic system parameters."""

    gAw: float = 10.1265729225269  # noqa: N815
    Ch: float = 4015.39425109821
    Ci: float = 1914908.30860716
    Ce: float = 15545663.6743828
    e11: float = -9.49409438095981
    sigmai: float = -37.8538482163307
    sigmah: float = -50.4867241844347
    sigmae: float = -5.57887704511886
    Rea: float = 0.00751396226986365
    Rhi: float = 0.0761996125919563
    Rie: float = 0.00135151763922409
    eta: float = 0.98


@dataclass
class BestestHydronicHeatpumpParameters(BestestParameters):
    """Heat pump system parameters for a hydronic heating system."""

    gAw: float = 40.344131392192  # noqa: N815
    Ch: float = 10447262.2318648
    Ci: float = 14827137.0377258
    Ce: float = 50508258.9032192
    e11: float = -30.0936560706053
    sigmai: float = -23.3175423490014
    sigmah: float = -19.5274067368137
    sigmae: float = -5.07591222090641
    Rea: float = 0.00163027389197229
    Rhi: float = 0.000437603769897038
    Rie: float = 0.000855786902577802
    eta: float = 0.98


def nominal_ode(x: ca.SX, u: ca.SX, d: ca.SX, p: dict[str, float]) -> np.ndarray:
    """
    Calculate the state derivatives for the thermal model.

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


def get_f_expl_expr(
    x: ca.SX,
    u: ca.SX,
    d: ca.SX,
    params: dict[str, float],
) -> ca.SX:
    """
    Get the state derivatives as a CasADi expression.

    Args:
        x: Current state vector [Ti, Th, Te]
        u: Heat input to radiator [W]
        d: Disturbance inputs [Ta, Phi_s]
        params: Parameters
    Returns:
        State derivatives as a CasADi expression [dTi/dt, dTh/dt, dTe/dt]
    """
    # Unpack state variables
    Ti, Th, Te = x[0], x[1], x[2]
    qh = u
    Ta, Phi_s = d[0], d[1]

    # State derivatives according to equations (1a)-(1c)
    dTi_dt = (
        (1 / (params["Ci"] * params["Rhi"])) * (Th - Ti)
        + (1 / (params["Ci"] * params["Rie"])) * (Te - Ti)
        + (1 / params["Ci"]) * params["gAw"] * Phi_s
    )
    dTh_dt = (1 / (params["Ch"] * params["Rhi"])) * (Ti - Th) + (qh / params["Ch"])
    dTe_dt = (1 / (params["Ce"] * params["Rie"])) * (Ti - Te) + (
        1 / (params["Ce"] * params["Rea"])
    ) * (Ta - Te)

    return ca.vertcat(dTi_dt, dTh_dt, dTe_dt)


def transcribe_continuous_state_space(
    Ac: ca.SX | np.ndarray,
    Bc: ca.SX | np.ndarray,
    Ec: ca.SX | np.ndarray,
    params: dict[str, float],
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """
    Create continuous-time state-space matrices Ac, Bc, Ec as per equation (6).

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
    """
    Create discrete-time state-space matrices Ad, Bd, Ed as per equation (7).

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
