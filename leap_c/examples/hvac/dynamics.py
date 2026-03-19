"""Thermal dynamics and state-space utilities for HVAC environment."""

from dataclasses import dataclass, field, fields

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
        gAw: Effective window area [m²]
        Ch: Heating system thermal capacity [J/K]
        Ci: Indoor thermal capacity [J/K]
        Ce: External thermal capacity [J/K]
        Rea: Resistance external-ambient [K/W]
        Rie: Resistance indoor-external [K/W]
        Rhi: Resistance heating-indoor [K/W]
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

    dynamics: HydronicDynamicsParameters = field(default_factory=HydronicDynamicsParameters)
    noise: HydronicNoiseParameters = field(default_factory=HydronicNoiseParameters)

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
        for field_name in fields(self.dynamics):
            value = getattr(self.dynamics, field_name.name)
            randomized_det[field_name.name] = rng.normal(
                loc=value, scale=noise_scale * np.sqrt(value**2)
            )

        # Randomize noise parameters
        randomized_noise = {}
        for field_name in fields(self.noise):
            value = getattr(self.noise, field_name.name)
            randomized_noise[field_name.name] = rng.normal(
                loc=value, scale=noise_scale * np.sqrt(value**2)
            )

        return HydronicParameters(
            dynamics=HydronicDynamicsParameters(**randomized_det),
            noise=HydronicNoiseParameters(**randomized_noise),
        )


def transcribe_continuous_state_space(
    params: HydronicDynamicsParameters,
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Create continuous-time state-space matrices Ac, Bc, Ec as per equation (6).

    Args:
        params: Hydronic thermal parameters

    Returns:
        Ac, Bc, Ec: State-space matrices

    """
    # Extract parameters
    # Convert to scalar for numpy arrays to avoid scalar conversion deprecation
    if any(isinstance(p, ca.SX) for p in vars(params).values()):
        Ac = ca.SX.zeros(3, 3)
        Bc = ca.SX.zeros(3, 1)
        Ec = ca.SX.zeros(3, 2)

        # Keep as-is for CasADi symbolic expressions
        Ch = params.Ch  # Radiator thermal capacitance
        Ci = params.Ci  # Indoor air thermal capacitance
        Ce = params.Ce  # Envelope thermal capacitance
        Rhi = params.Rhi  # Radiator to indoor air resistance
        Rie = params.Rie  # Indoor air to envelope resistance
        Rea = params.Rea  # Envelope to outdoor resistance
        gAw = params.gAw  # Effective window area
    else:
        Ac = np.zeros((3, 3))
        Bc = np.zeros((3, 1))
        Ec = np.zeros((3, 2))

        # Use .item() for numpy arrays/scalars, otherwise use value directly
        Ch = params.Ch.item() if hasattr(params.Ch, "item") else params.Ch
        Ci = params.Ci.item() if hasattr(params.Ci, "item") else params.Ci
        Ce = params.Ce.item() if hasattr(params.Ce, "item") else params.Ce
        Rhi = params.Rhi.item() if hasattr(params.Rhi, "item") else params.Rhi
        Rie = params.Rie.item() if hasattr(params.Rie, "item") else params.Rie
        Rea = params.Rea.item() if hasattr(params.Rea, "item") else params.Rea
        gAw = params.gAw.item() if hasattr(params.gAw, "item") else params.gAw

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


def expm_pade66_robust(At, s=1):
    # Scale the matrix
    A_scaled = At / (2**s)

    # 2. Padé [6, 6] on the scaled matrix with U, V split
    I = np.eye(At.shape[0]) if isinstance(At, np.ndarray) else ca.SX.eye(At.size1())
    A2 = A_scaled @ A_scaled
    A4 = A2 @ A2
    A6 = A4 @ A2

    V = 1.0 * I + (1 / 10) * A2 + (1 / 2100) * A4 + (1 / 1995840) * A6
    U = A_scaled @ (0.5 * I + (1 / 120) * A2 + (1 / 50400) * A4)

    # Solve (V-U)X = (V+U)
    X = ca.solve(V - U, V + U) if not isinstance(At, np.ndarray) else np.linalg.solve(V - U, V + U)

    # e^(At) = (e^(At/2^s))^(2^s)
    for _ in range(s):
        X = X @ X

    return X


def transcribe_discrete_state_space(
    dt: float,
    params: HydronicDynamicsParameters,
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Create discrete-time state-space matrices Ad, Bd, Ed as per equation (7).

    Args:
        dt: Sampling time
        params: Hydronic thermal parameters

    Returns:
        Ad, Bd, Ed: Discrete-time state-space matrices

    """
    # Extract type of Ad
    # Create continuous-time state-space matrices
    Ac, Bc, Ec = transcribe_continuous_state_space(
        params=params,
    )
    if all(isinstance(Mc, np.ndarray) for Mc in [Ac, Bc, Ec]):
        # Discretize the continuous-time state-space representation
        Ad = expm_pade66_robust(Ac * dt)  # Discrete-time state matrix
        Bd = np.linalg.solve(Ac, (Ad - np.eye(3))) @ Bc
        Ed = np.linalg.solve(Ac, (Ad - np.eye(3))) @ Ec
    else:
        # Create continuous-time state-space matrices
        Ac, Bc, Ec = transcribe_continuous_state_space(
            params=params,
        )

        # Discretize the continuous-time state-space representation
        Ad = expm_pade66_robust(Ac * dt)  # Discrete-time state matrix
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
        params=params.dynamics,
    )

    Qd = compute_noise_covariance(
        Ac=Ac,
        Sigma=np.diag([sigma_i, sigma_h, sigma_e]),
        dt=dt,
    )

    # Compute discrete-time state-space matrices
    Ad, Bd, Ed = transcribe_discrete_state_space(
        dt=dt,
        params=params.dynamics,
    )

    return Ad, Bd, Ed, Qd


def compute_steady_state(
    Ti_ss: float,
    temperature_ss: float,
    solar_ss: float,
    params: HydronicDynamicsParameters,
) -> tuple[float, float, float]:
    """Compute steady-state values for HVAC system.

    Args:
        Ti_ss: Indoor temperature steady-state value in Kelvin.
        temperature_ss: Outdoor temperature steady-state value in Kelvin.
        solar_ss: Solar radiation steady-state value in W/m^2.
        params: Hydronic system parameters.

    Returns:
        Tuple of (qh_ss, Th_ss, Te_ss) where:
            - qh_ss: Steady-state heating power in W, clipped to be non-negative
            - Th_ss: Steady-state radiator temperature in Kelvin
            - Te_ss: Steady-state envelope temperature in Kelvin
    """
    Ac, Bc, Ec = transcribe_continuous_state_space(
        params=params,
    )

    A = np.hstack([Bc, Ac[:, 1:]])
    b = -(Ac[:, 0] * Ti_ss + Ec[:, 0] * temperature_ss + Ec[:, 1] * solar_ss).reshape(-1, 1)

    ss = np.linalg.solve(A, b).flatten()

    qh_ss = max(0, ss[0])
    Th_ss = ss[1]  # Temperature in Kelvin
    Te_ss = ss[2]  # Temperature in Kelvin

    return qh_ss, Th_ss, Te_ss


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.linalg import expm as scipy_expm

    # --- Part 1: Numerical Error Analysis ---
    params = HydronicDynamicsParameters()
    Ac, _, _ = transcribe_continuous_state_space(params)

    t_values = np.linspace(1, 1800, 300)

    errors = []
    norms = []

    for t in t_values:
        true_val = scipy_expm(Ac * t)
        # We test with s=11 (scaling of 2048)
        pade_val = expm_pade66_robust(Ac * t, s=11)

        err = np.linalg.norm(true_val - pade_val)
        errors.append(err)
        norms.append(np.linalg.norm(Ac * t, np.inf))

    # Create Two-Column Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error over time
    ax1.semilogy(t_values, errors, label="[6,6] Padé + S&S (s=11)", color="royalblue", lw=2)
    ax1.axvline(900, color="green", linestyle=":", label="Your target (900s)")
    ax1.axhline(1e-15, color="red", linestyle="--", alpha=0.4, label="Machine Epsilon")
    ax1.set_title("Approximation Error for Hydronic System")
    ax1.set_xlabel("Time Step (dt) [seconds]")
    ax1.set_ylabel("Frobenius Norm Error")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # Plot 2: Matrix Norm vs Error (The true stability metric)
    ax2.scatter(norms, errors, c=t_values, cmap="viridis", s=10)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_title("Error vs. Matrix Infinity Norm")
    ax2.set_xlabel("||Ac * dt||_inf")
    ax2.set_ylabel("Error")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Error at dt=900: {errors[np.argmin(np.abs(t_values - 900))]:.2e}")

    # --- Part 2: Sensitivity Analysis w.r.t Rhi ---
    # We replace Rhi with a symbolic variable
    Rhi_sx = ca.SX.sym("Rhi")

    # Use existing dataclass but swap one member for the symbolic variable
    params_sx = HydronicDynamicsParameters(Rhi=Rhi_sx)

    # Transcribe continuous system symbolically
    Ac_sx, _, _ = transcribe_continuous_state_space(params_sx)

    # Discretize at target dt=900
    dt_target = 900
    Ad_sx = expm_pade66_robust(Ac_sx * dt_target)

    # Compute Jacobian of the first state (T_indoor) w.r.t Rhi
    # We look at the top-left element of the discrete state matrix Ad
    jac_Ad_Rhi = ca.jacobian(Ad_sx[0, 0], Rhi_sx)

    f_Ad = ca.Function("f_Ad", [Rhi_sx], [Ad_sx[0, 0]])
    f_jac = ca.Function("f_jac", [Rhi_sx], [jac_Ad_Rhi])

    # Sweep Rhi around nominal value 0.076
    r_sweep = np.linspace(0.01, 0.2, 100)
    ad_vals = [float(f_Ad(r)) for r in r_sweep]
    jac_vals = [float(f_jac(r)) for r in r_sweep]

    # --- NEW: Finite Difference Calculation ---
    eps = 1e-6  # Perturbation step
    fd_vals = []
    for r in r_sweep:
        v_plus = float(f_Ad(r + eps))
        v_minus = float(f_Ad(r - eps))
        fd_vals.append((v_plus - v_minus) / (2 * eps))

    # --- Part 3: Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Left: Numerical Error (Proves value is correct)

    ax1.semilogy(t_values, errors, color="royalblue", lw=2)

    ax1.axvline(dt_target, color="green", linestyle=":", label="Target dt=900s")

    ax1.set_title("Numerical Error vs. dt")

    ax1.set_ylabel("Frobenius Norm Error")

    ax1.legend()

    # Right: Sensitivity Plot (Proves gradient is correct)
    ax2_twin = ax2.twinx()
    p1 = ax2.plot(r_sweep, ad_vals, color="tab:blue", label="Ad[0,0] Value", lw=2)

    # CasADi Analytical Gradient (Dashed Red)
    p2 = ax2_twin.plot(
        r_sweep, jac_vals, color="tab:red", ls="--", label="CasADi Jacobian (AD)", lw=2
    )

    # Finite Difference Gradient (Dotted Black - should overlap p2)
    p3 = ax2_twin.plot(
        r_sweep, fd_vals, color="black", ls=":", label="Finite Difference (FD)", lw=1.5
    )

    ax2.set_title(f"Sensitivity Analysis @ dt={dt_target}s")
    ax2.set_xlabel("Resistance Rhi [K/W]")
    ax2.set_ylabel("Value", color="tab:blue")
    ax2_twin.set_ylabel("Jacobian / Slope", color="tab:red")

    # Combine legends
    lns = p1 + p2 + p3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="best")

    plt.tight_layout()
    plt.show()
