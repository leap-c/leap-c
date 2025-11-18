"""Thermal dynamics and state-space utilities for HVAC environment."""

from dataclasses import dataclass, fields

import casadi as ca
import numpy as np
import scipy.linalg


@dataclass(kw_only=True)
class HydronicDynamicsParameters:
    """Deterministic HVAC dynamics parameters.

    The parameters are from a CSTMR hydronic system model to be used as a control model
    in the BOPTEST BestestHydronic test case
    (https://ibpsa.github.io/project1-boptest/testcases/ibpsa/testcases_ibpsa_bestest_hydronic/).

    Attributes:
        Ch: Heating system thermal capacity [J/K]
        Ci: Indoor thermal capacity [J/K]
        Ce: External thermal capacity [J/K]
        Rhi: Resistance heating-indoor [K/W]
        Rie: Resistance indoor-external [K/W]
        Rea: Resistance external-ambient [K/W]
        gAw: Effective window area [m²]
    """

    gAw: float | ca.SX = 10.1265729225269
    Ch: float | ca.SX = 4015.39425109821
    Ci: float | ca.SX = 1914908.30860716
    Ce: float | ca.SX = 15545663.6743828
    Rea: float | ca.SX = 0.00751396226986365
    Rhi: float | ca.SX = 0.0761996125919563
    Rie: float | ca.SX = 0.00135151763922409


@dataclass(kw_only=True)
class HydronicNoiseParameters:
    """Noise parameters for the hydronic system.

    The parameters are from a CSTMR hydronic system model to be used as a control model
    in the BOPTEST BestestHydronic test case
    (https://ibpsa.github.io/project1-boptest/testcases/ibpsa/testcases_ibpsa_bestest_hydronic/).

    Attributes:
        e11: Measurement noise
        sigmai: Indoor temperature process noise
        sigmah: Heating system process noise
        sigmae: External temperature process noise
    """

    e11: float | ca.SX = -9.49409438095981
    sigmai: float | ca.SX = -37.8538482163307
    sigmah: float | ca.SX = -50.4867241844347
    sigmae: float | ca.SX = -5.57887704511886


@dataclass(kw_only=True)
class HydronicParameters:
    """Hydronic system thermal parameters.

    Default values are from the BESTEST hydronic system standard configuration.

    Attributes:
        deterministic: Deterministic thermal parameters
        noise: Noise parameters
        eta: Efficiency for electric heater
    """

    dynamics: HydronicDynamicsParameters = None
    noise: HydronicNoiseParameters = None

    # Heater parameters
    def __post_init__(self):
        """Initialize default values for nested dataclasses if None."""
        if self.dynamics is None:
            self.dynamics = HydronicDynamicsParameters()
        if self.noise is None:
            self.noise = HydronicNoiseParameters()

    def randomize(self, rng: np.random.Generator, noise_scale: float = 0.3) -> "HydronicParameters":
        """Generate a new HydronicParameters instance with randomized values.

        Args:
            rng: NumPy random generator for reproducibility.
            noise_scale: Scale for parameter randomization (default: 0.3).

        Returns:
            New HydronicParameters instance with randomized values.
        """
        # Randomize deterministic parameters
        randomized_det = {}
        for field in fields(self.dynamics):
            value = getattr(self.dynamics, field.name)
            randomized_det[field.name] = rng.normal(
                loc=value, scale=noise_scale * np.sqrt(value**2)
            )

        # Randomize noise parameters
        randomized_noise = {}
        for field in fields(self.noise):
            value = getattr(self.noise, field.name)
            randomized_noise[field.name] = rng.normal(
                loc=value, scale=noise_scale * np.sqrt(value**2)
            )

        return HydronicParameters(
            dynamics=HydronicDynamicsParameters(**randomized_det),
            noise=HydronicNoiseParameters(**randomized_noise),
        )


def transcribe_continuous_state_space(
    Ac: ca.SX | np.ndarray,
    Bc: ca.SX | np.ndarray,
    Ec: ca.SX | np.ndarray,
    params: HydronicDynamicsParameters,
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Create continuous-time state-space matrices Ac, Bc, Ec as per equation (6).

    Args:
        Ac: State-space matrix (system dynamics)
        Bc: State-space matrix (control input)
        Ec: State-space matrix (disturbances)
        params: Hydronic thermal parameters

    Returns:
        Ac, Bc, Ec: State-space matrices

    """
    # Extract parameters
    Ch = params.Ch  # Radiator thermal capacitance
    Ci = params.Ci  # Indoor air thermal capacitance
    Ce = params.Ce  # Envelope thermal capacitance
    Rhi = params.Rhi  # Radiator to indoor air resistance
    Rie = params.Rie  # Indoor air to envelope resistance
    Rea = params.Rea  # Envelope to outdoor resistance
    gAw = params.gAw  # Effective window area

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
    params: HydronicDynamicsParameters,
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Create discrete-time state-space matrices Ad, Bd, Ed as per equation (7).

    Args:
        Ad: State-space matrix (system dynamics)
        Bd: State-space matrix (control input)
        Ed: State-space matrix (disturbances)
        dt: Sampling time
        params: Hydronic thermal parameters

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
        Bd = ca.mtimes(ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Bc)  # Discrete-time input matrix
        Ed = ca.mtimes(
            ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Ec
        )  # Discrete-time disturbance matrix

    return Ad, Bd, Ed


def compute_noise_covariance(Ac: np.ndarray, Sigma: np.ndarray, dt: float) -> np.ndarray:
    """Compute the exact discrete-time noise covariance matrix using matrix exponential.

    The discrete-time noise covariance is computed as:
    Q_d = ∫₀^Δt e^(Aτ) Σ Σᵀ e^(Aᵀτ) dτ.

    Args:
        Ac: Continuous-time system matrix.
        Sigma: Noise intensity matrix (diagonal).
        dt: Sampling time in seconds.

    Returns:
        Discrete-time noise covariance matrix.
    """
    n = Ac.shape[0]  # State dimension (3)

    # Create the augmented matrix for computing the noise covariance integral
    # [ A    Σ Σᵀ ]
    # [ 0      -Aᵀ ]
    SigmaSigmaT = Sigma @ Sigma.T

    # Augmented matrix (6x6)
    M = np.block([[Ac, SigmaSigmaT], [np.zeros((n, n)), -Ac.T]])

    # Matrix exponential of augmented system
    exp_M = scipy.linalg.expm(M * dt)

    # Extract the noise covariance from the upper-right block
    # Qd = e^(A*dt) * (upper-right block of exp_M)
    Ad = exp_M[:n, :n]
    Phi = exp_M[:n, n:]

    # The discrete-time covariance is Qd = Ad @ Phi
    matrix = Ad @ Phi

    # Make symmetric by averaging with transpose
    symmetric_matrix = 0.5 * (matrix + matrix.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)

    # Clip negative eigenvalues to zero (or small positive value)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Reconstruct the matrix
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def compute_discrete_matrices(
    params: HydronicParameters,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute discrete-time matrices using exact discretization via matrix exponential.

    This includes both deterministic dynamics and noise covariance.

    Args:
        params: Hydronic thermal parameters
            (Ch, Ci, Ce, Rhi, Rie, Rea, gAw, sigmai, sigmah, sigmae).
        dt: Sampling time in seconds.

    Returns:
        Tuple containing:
            - Ad: Discrete-time state matrix.
            - Bd: Discrete-time input matrix.
            - Ed: Discrete-time disturbance matrix.
            - Qd: Discrete-time noise covariance matrix.
    """
    # Create noise intensity matrix Σ from parameters
    # The stochastic terms are σᵢω̇ᵢ, σₕω̇ₕ, σₑω̇ₑ
    sigma_i = np.exp(params.noise.sigmai)
    sigma_h = np.exp(params.noise.sigmah)
    sigma_e = np.exp(params.noise.sigmae)

    # Compute continuous-time Ac
    Ac, _, _ = transcribe_continuous_state_space(
        Ac=np.zeros((3, 3)),
        Bc=np.zeros((3, 1)),
        Ec=np.zeros((3, 2)),
        params=params.dynamics,
    )

    Qd = compute_noise_covariance(
        Ac=Ac,
        Sigma=np.diag([sigma_i, sigma_h, sigma_e]),
        dt=dt,
    )

    # Compute discrete-time state-space matrices
    Ad, Bd, Ed = transcribe_discrete_state_space(
        Ad=np.zeros((3, 3)),
        Bd=np.zeros((3, 1)),
        Ed=np.zeros((3, 2)),
        dt=dt,
        params=params.dynamics,
    )

    return Ad, Bd, Ed, Qd


def compute_steady_state(
    Ti_ss: float,
    Ta_ss: float,
    Phi_ss: float,
    params: HydronicDynamicsParameters,
) -> tuple[float, float, float]:
    """Compute steady-state values for HVAC system.

    Args:
        Ti_ss: Indoor temperature steady-state value in Kelvin.
        Ta_ss: Outdoor temperature steady-state value in Kelvin.
        Phi_ss: Solar radiation steady-state value in W/m^2.
        params: Hydronic system parameters.

    Returns:
        Tuple of (qh_ss, Th_ss, Te_ss) where:
            - qh_ss: Steady-state heating power in W, clipped to be non-negative
            - Th_ss: Steady-state radiator temperature in Kelvin
            - Te_ss: Steady-state envelope temperature in Kelvin
    """
    Ac, Bc, Ec = transcribe_continuous_state_space(
        Ac=np.zeros((3, 3)),
        Bc=np.zeros((3, 1)),
        Ec=np.zeros((3, 2)),
        params=params.dynamics,
    )

    A = np.hstack([Bc, Ac[:, 1:]])
    b = -(Ac[:, 0] * Ti_ss + Ec[:, 0] * Ta_ss + Ec[:, 1] * Phi_ss).reshape(-1, 1)

    ss = np.linalg.solve(A, b).flatten()

    qh_ss = max(0, ss[0])
    Th_ss = ss[1]  # Temperature in Kelvin
    Te_ss = ss[2]  # Temperature in Kelvin

    return qh_ss, Th_ss, Te_ss
