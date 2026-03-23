from dataclasses import asdict
from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp
from scipy.constants import convert_temperature

from leap_c.examples.hvac.dynamics import (
    HydronicDynamicsParameters,
    HydronicParameters,
    expm_pade66_robust,
    transcribe_discrete_state_space,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

HvacAcadosParamInterface = Literal["reference", "reference_dynamics"]
"""Determines the exposed parameter interface of the planner.
- "reference": Only reference parameters are exposed (e.g., ambient temperature,
  solar radiation, electricity price, comfort bounds, etc.).
- "reference_dynamics": Both reference parameters and dynamics parameters are exposed.
"""

HvacAcadosParamGranularity = Literal["global", "stagewise"] | int

# Mapping from resistance parameter names to conductance parameter names.
_R_TO_G: dict[str, str] = {"Rhi": "ghi", "Rie": "gie", "Rea": "gea"}
_G_TO_R: dict[str, str] = {v: k for k, v in _R_TO_G.items()}


def make_default_hvac_params(
    interface: HvacAcadosParamInterface,
    granularity: HvacAcadosParamGranularity,
    N_horizon: int = 96,
    hydronic_params: HydronicParameters | None = None,
) -> tuple[AcadosParameter, ...]:
    """Return a tuple of default parameters for the hvac planner.

    Args:
        interface: The parameter interface type.
        granularity: The granularity of the parameters.
        N_horizon: The number of steps in the MPC horizon
            (default: 96, i.e., 24 hours in 15-minute steps).
        hydronic_params: Optional HydronicParameters object.

    Returns:
        Tuple of AcadosParameter objects for the HVAC system.
    """
    hydronic_params = HydronicParameters() if hydronic_params is None else hydronic_params

    # TODO: The default parameter shifts are currently hardcoded. This will be
    # improved once the environment interface is refactored.
    scaling_shifts = {
        "gAw": 0.8,
        "Rea": 0.8,
        "Rhi": 0.8,
        "Rie": 0.8,
        "Ch": 0.8,
        "Ci": 0.8,
        "Ce": 0.8,
    }
    params = []

    for k, v in asdict(hydronic_params.dynamics).items():
        default_value = scaling_shifts[k] * np.array([v])
        if k in scaling_shifts and "dynamics" in interface:
            param_interface = "learnable"
        else:
            param_interface = "fix"

        params.append(
            AcadosParameter(
                name=k,
                default=default_value,
                space=gym.spaces.Box(
                    low=0.7 * np.array([v]), high=1.3 * np.array([v]), dtype=np.float64
                ),
                interface=param_interface,
            )
        )

    if isinstance(granularity, int):
        assert 1 <= granularity <= N_horizon, (
            "Granularity must be between 0 and N_horizon (inclusive) when specified as an integer."
        )
        step = max(1, N_horizon // granularity)
        end_stages = list(range(0, N_horizon + 1, step))
        if end_stages[-1] != N_horizon:
            end_stages.append(N_horizon)
    else:
        end_stages = list(range(N_horizon + 1)) if granularity == "stagewise" else []

    params.extend(
        [
            AcadosParameter(
                name="temperature",  # Ambient temperature in Kelvin
                default=np.array([convert_temperature(20.0, "celsius", "kelvin")]),
                space=gym.spaces.Box(
                    low=np.array([convert_temperature(-20.0, "celsius", "kelvin")]),
                    high=np.array([convert_temperature(40.0, "celsius", "kelvin")]),
                    dtype=np.float64,
                ),
                interface="non-learnable",
                end_stages=list(range(N_horizon + 1)),
            ),
            AcadosParameter(
                name="solar",
                default=np.array([200.0]),  # Solar radiation in W/m²
                space=gym.spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([2000.0]),
                    dtype=np.float64,
                ),
                interface="non-learnable",
                end_stages=list(range(N_horizon + 1)),
            ),
            AcadosParameter(
                name="price",
                default=np.array([0.15]),  # Electricity price in €/kWh
                space=gym.spaces.Box(
                    low=np.array([0.00]),
                    high=np.array([10.0]),
                    dtype=np.float64,
                ),
                interface="non-learnable",
                end_stages=list(range(N_horizon + 1)),
            ),
        ]
    )

    # Comfort constraints for indoor temperature
    # Note: lb_Ti and ub_Ti are set at runtime via constraints_set() in the planner,
    # NOT as OCP parameters — they have no effect as parameters since the OCP model
    # never calls param_manager.get("lb_Ti"/"ub_Ti").
    params.extend(
        [
            AcadosParameter(
                name="ref_Ti",  # Reference indoor temperature in Kelvin
                default=np.array([convert_temperature(21.0, "celsius", "kelvin")]),
                space=gym.spaces.Box(
                    low=np.array([convert_temperature(10.0, "celsius", "kelvin")]),
                    high=np.array([convert_temperature(30.0, "celsius", "kelvin")]),
                    dtype=np.float64,
                ),
                interface="learnable",
                end_stages=end_stages,
            ),
            AcadosParameter(
                name="backoff_Ti",  # Backoff temperature for indoor temperature in Kelvin
                default=np.array([0.0]),
                space=gym.spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([2.0]),
                    dtype=np.float64,
                ),
                interface="fix",
                end_stages=end_stages,
            ),
        ]
    )

    params.extend(
        [
            AcadosParameter(
                name="log_q_Ti",
                default=np.array([-1.9]),  # log10 of the weight for indoor temperature residuals
                space=gym.spaces.Box(low=np.array([-2.0]), high=np.array([1.0]), dtype=np.float64),
                interface="learnable",
                end_stages=end_stages,
            ),
            AcadosParameter(
                name="q_dqh",
                default=np.array([0.001]),  # weight for residuals of rate of change of heater power
                space=gym.spaces.Box(low=np.array([0.0]), high=np.array([0.01]), dtype=np.float64),
                interface="fix",
                end_stages=end_stages,
            ),
        ]
    )

    return tuple(params)


def _build_ocp_with_matrices(
    Ad: ca.SX,
    Bd_kW: ca.SX,
    Ed: ca.SX,
    param_manager: AcadosParameterManager,
    N_horizon: int,
    name: str,
    x0: np.ndarray | None,
) -> AcadosOcp:
    """Construct an AcadosOcp from pre-computed discrete state-space matrices.

    Shared by all parameterisations; the only difference between them is how
    Ad, Bd_kW, Ed are built before this function is called.

    State:  x = [Ti, Th, Te, qh_prev]   (K, K, K, kW)
    Input:  u = qh                        (kW)
    """
    dt: float = 900.0

    ocp = AcadosOcp()
    param_manager.assign_to_ocp(ocp)
    ocp.model.name = name

    # ── State ─────────────────────────────────────────────────────────────────
    Ti = ca.SX.sym("Ti")
    Th = ca.SX.sym("Th")
    Te = ca.SX.sym("Te")
    qh_prev = ca.SX.sym("qh_prev")
    ocp.model.x = ca.vertcat(Ti, Th, Te, qh_prev)

    # ── Input ─────────────────────────────────────────────────────────────────
    qh = ca.SX.sym("qh")
    ocp.model.u = qh

    # ── Discrete dynamics ─────────────────────────────────────────────────────
    x_phys = ca.vertcat(Ti, Th, Te)
    d_vec = ca.vertcat(param_manager.get("temperature"), param_manager.get("solar"))
    x_next_phys = Ad @ x_phys + Bd_kW * qh + Ed @ d_vec
    ocp.model.disc_dyn_expr = ca.vertcat(x_next_phys, qh)

    # abs_qh = ca.fabs(qh)

    # ── Cost ──────────────────────────────────────────────────────────────────
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (
        0.25 * param_manager.get("price") * qh + param_manager.get("q_dqh") * (qh - qh_prev) ** 2
    )
    ocp.model.cost_expr_ext_cost_e = 1e-3 * (Ti - convert_temperature(20.0, "C", "K")) ** 2

    # ── Constraints ───────────────────────────────────────────────────────────
    Ti_ic = convert_temperature(20.0, "C", "K")
    ocp.constraints.x0 = np.array([Ti_ic, Ti_ic, Ti_ic, 0.0]) if x0 is None else x0

    ocp.constraints.lbx = np.array([convert_temperature(12.0, "C", "K")])
    ocp.constraints.ubx = np.array([convert_temperature(30.0, "C", "K")])
    ocp.constraints.idxbx = np.array([0])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zl = 1e3 * np.ones((1,))
    ocp.cost.Zl = 1e3 * np.ones((1,))
    ocp.cost.zu = 1e3 * np.ones((1,))
    ocp.cost.Zu = 1e3 * np.ones((1,))

    ocp.constraints.lbx_e = np.array([convert_temperature(12.0, "C", "K")])
    ocp.constraints.ubx_e = np.array([convert_temperature(30.0, "C", "K")])
    ocp.constraints.idxbx_e = np.array([0])

    ocp.constraints.lbu = np.array([-5.0])
    ocp.constraints.ubu = np.array([5.0])
    ocp.constraints.idxbu = np.array([0])

    # ── Solver options ────────────────────────────────────────────────────────
    ocp.solver_options.tf = N_horizon * dt
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp


def _build_ocp(
    dynamics_params: HydronicDynamicsParameters,
    param_manager: AcadosParameterManager,
    N_horizon: int,
    name: str,
    x0: np.ndarray | None,
) -> AcadosOcp:
    """Construct an AcadosOcp from HydronicDynamicsParameters."""
    dt: float = 900.0
    Ad, Bd, Ed = transcribe_discrete_state_space(dt=dt, params=dynamics_params)
    return _build_ocp_with_matrices(Ad, Bd * 1e3, Ed, param_manager, N_horizon, name, x0)


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    N_horizon: int,
    name: str = "hvac",
    x0: np.ndarray | None = None,
) -> AcadosOcp:
    """Export the HVAC OCP with resistance-space learnable dynamics parameters.

    Learnable dynamics parameters: gAw, Ch, Ci, Ce, Rea, Rhi, Rie (physical units).

    State:   x = [Ti, Th, Te, qh_prev]  (K, K, K, kW)
    Input:   u = qh                      (kW)

    Args:
        param_manager: Parameter manager built from ``make_default_hvac_params``.
        N_horizon: Number of time steps in the horizon.
        name: Acados model name.
        x0: Initial state.  Uses a 20 °C default when ``None``.

    Returns:
        Configured AcadosOcp ready for solver creation.
    """
    return _build_ocp(
        dynamics_params=param_manager.recreate_dataclass(HydronicDynamicsParameters),
        param_manager=param_manager,
        N_horizon=N_horizon,
        name=name,
        x0=x0,
    )


def make_default_hvac_params_conductance(
    interface: HvacAcadosParamInterface,
    granularity: HvacAcadosParamGranularity,
    N_horizon: int = 96,
    hydronic_params: HydronicParameters | None = None,
) -> tuple[AcadosParameter, ...]:
    """Return HVAC parameters with thermal conductances instead of resistances.

    Replaces the learnable resistance parameters Rhi, Rie, Rea with their
    reciprocals ghi = 1/Rhi, gie = 1/Rie, gea = 1/Rea  [W/K].  All other
    parameters are identical to ``make_default_hvac_params``.

    Conductance bounds are the reciprocals of the resistance bounds (inverted):
        R ∈ [0.7·R_nom, 1.3·R_nom]  ⟺  g ∈ [1/(1.3·R_nom), 1/(0.7·R_nom)]

    Args:
        interface: Parameter interface type.
        granularity: Parameter granularity.
        N_horizon: MPC horizon length.
        hydronic_params: Optional nominal hydronic parameters.

    Returns:
        Tuple of AcadosParameter objects with conductance dynamics parameters.
    """
    hydronic_params = HydronicParameters() if hydronic_params is None else hydronic_params
    dyn = hydronic_params.dynamics

    scaling = 0.8
    params: list[AcadosParameter] = []

    for k, v in asdict(dyn).items():
        is_dyn_learnable = "dynamics" in interface
        if k in _R_TO_G:
            # Conductance parameter: g = 1/R
            g_name = _R_TO_G[k]
            r_nom = float(v)
            g_default = 1.0 / (scaling * r_nom)
            g_lb = 1.0 / (1.3 * r_nom)
            g_ub = 1.0 / (0.7 * r_nom)
            params.append(
                AcadosParameter(
                    name=g_name,
                    default=np.array([g_default]),
                    space=gym.spaces.Box(
                        low=np.array([g_lb]), high=np.array([g_ub]), dtype=np.float64
                    ),
                    interface="learnable" if is_dyn_learnable else "fix",
                )
            )
        else:
            # Non-resistance parameter: keep in physical units
            params.append(
                AcadosParameter(
                    name=k,
                    default=scaling * np.array([v]),
                    space=gym.spaces.Box(
                        low=0.7 * np.array([v]), high=1.3 * np.array([v]), dtype=np.float64
                    ),
                    interface="learnable" if is_dyn_learnable else "fix",
                )
            )

    # Non-learnable and cost/reference parameters are identical to the original.
    # Reuse make_default_hvac_params to avoid duplication.
    original = make_default_hvac_params(interface, granularity, N_horizon, hydronic_params)
    non_dyn = [p for p in original if p.name not in asdict(dyn)]
    params.extend(non_dyn)

    return tuple(params)


def export_parametric_ocp_conductance(
    param_manager: AcadosParameterManager,
    N_horizon: int,
    name: str = "hvac_conductance",
    x0: np.ndarray | None = None,
) -> AcadosOcp:
    """Export the HVAC OCP with conductance-space learnable dynamics parameters.

    Learnable dynamics parameters: gAw, Ch, Ci, Ce, ghi, gie, gea  [W/K for conductances].
    The OCP internally converts ghi → 1/ghi before evaluating the dynamics, so the
    physics are identical to ``export_parametric_ocp`` for equivalent parameter values.

    Args:
        param_manager: Parameter manager built from ``make_default_hvac_params_conductance``.
        N_horizon: Number of time steps in the horizon.
        name: Acados model name.
        x0: Initial state.  Uses a 20 °C default when ``None``.

    Returns:
        Configured AcadosOcp ready for solver creation.
    """
    dynamics_params = HydronicDynamicsParameters(
        gAw=param_manager.get("gAw"),
        Ch=param_manager.get("Ch"),
        Ci=param_manager.get("Ci"),
        Ce=param_manager.get("Ce"),
        Rhi=1.0 / param_manager.get("ghi"),
        Rie=1.0 / param_manager.get("gie"),
        Rea=1.0 / param_manager.get("gea"),
    )
    return _build_ocp(dynamics_params, param_manager, N_horizon, name, x0)


def make_default_hvac_params_normalized(
    interface: HvacAcadosParamInterface,
    granularity: HvacAcadosParamGranularity,
    N_horizon: int = 96,
    hydronic_params: HydronicParameters | None = None,
) -> tuple[AcadosParameter, ...]:
    """Return HVAC parameters with all learnable parameters normalised to [−1, 1].

    Each learnable parameter p with physical bounds [lb, ub] is replaced by
        p_norm = −1 + 2·(p − lb)/(ub − lb)  ∈ [−1, 1].

    The default in normalised space is derived from the physical default:
        p_norm_default = −1 + 2·(p_default − lb)/(ub − lb).

    Non-learnable and fixed parameters are unchanged (they are always passed in
    physical units by the planner).

    Args:
        interface: Parameter interface type.
        granularity: Parameter granularity.
        N_horizon: MPC horizon length.
        hydronic_params: Optional nominal hydronic parameters.

    Returns:
        Tuple of AcadosParameter objects with normalised learnable parameters.
    """
    original = make_default_hvac_params(interface, granularity, N_horizon, hydronic_params)
    normalised: list[AcadosParameter] = []

    for p in original:
        if p.interface != "learnable" or p.space is None:
            normalised.append(p)
            continue

        lb = p.space.low.astype(np.float64)
        ub = p.space.high.astype(np.float64)
        norm_default = -1.0 + 2.0 * (p.default.astype(np.float64) - lb) / (ub - lb)

        normalised.append(
            AcadosParameter(
                name=p.name,
                default=norm_default,
                space=gym.spaces.Box(
                    low=-np.ones_like(lb), high=np.ones_like(ub), dtype=np.float64
                ),
                interface="learnable",
                end_stages=p.end_stages,
            )
        )

    return tuple(normalised)


def make_default_hvac_params_normalized_conductance(
    interface: HvacAcadosParamInterface,
    granularity: HvacAcadosParamGranularity,
    N_horizon: int = 96,
    hydronic_params: HydronicParameters | None = None,
) -> tuple[AcadosParameter, ...]:
    """Return HVAC parameters in conductance space, then normalised to [−1, 1].

    This combines the conductance reparametrisation (Rhi→ghi=1/Rhi, etc.) with
    the normalisation to [−1, 1].  The result has better gradient conditioning
    than either transformation alone:
    - conductance removes the 1/R² amplification,
    - normalisation makes all parameter ranges commensurate.

    Args:
        interface: Parameter interface type.
        granularity: Parameter granularity.
        N_horizon: MPC horizon length.
        hydronic_params: Optional nominal hydronic parameters.

    Returns:
        Tuple of AcadosParameter objects with normalised conductance parameters.
    """
    conductance = make_default_hvac_params_conductance(
        interface, granularity, N_horizon, hydronic_params
    )
    normalised: list[AcadosParameter] = []

    for p in conductance:
        if p.interface != "learnable" or p.space is None:
            normalised.append(p)
            continue

        lb = p.space.low.astype(np.float64)
        ub = p.space.high.astype(np.float64)
        norm_default = -1.0 + 2.0 * (p.default.astype(np.float64) - lb) / (ub - lb)

        normalised.append(
            AcadosParameter(
                name=p.name,
                default=norm_default,
                space=gym.spaces.Box(
                    low=-np.ones_like(lb), high=np.ones_like(ub), dtype=np.float64
                ),
                interface="learnable",
                end_stages=p.end_stages,
            )
        )

    return tuple(normalised)


def export_parametric_ocp_normalized_conductance(
    param_manager: AcadosParameterManager,
    N_horizon: int,
    name: str = "hvac_norm_cond",
    x0: np.ndarray | None = None,
    hydronic_params: HydronicParameters | None = None,
) -> AcadosOcp:
    """Export the HVAC OCP with conductance parameters normalised to [−1, 1].

    Combines conductance reparametrisation and normalisation:
    1. Each normalised conductance/parameter is mapped back to its physical range:
           p_phys = lb + (p_norm + 1)/2 · (ub - lb)
    2. For the thermal resistances (ghi, gie, gea), the physical resistance is
       recovered as  R = 1 / g_phys.

    Physical bounds for conductances are:
        g ∈ [1/(1.3·R_nom), 1/(0.7·R_nom)]

    Args:
        param_manager: Parameter manager built from
            ``make_default_hvac_params_normalized_conductance``.
        N_horizon: Number of time steps in the horizon.
        name: Acados model name.
        x0: Initial state.  Uses a 20 °C default when ``None``.
        hydronic_params: Nominal hydronic parameters used to recover physical bounds.

    Returns:
        Configured AcadosOcp ready for solver creation.
    """
    hydronic_params = HydronicParameters() if hydronic_params is None else hydronic_params
    dyn = hydronic_params.dynamics

    def _denorm(sym: ca.SX, lb: float, ub: float) -> ca.SX:
        return lb + (sym + 1.0) / 2.0 * (ub - lb)

    # Conductance bounds (same as in make_default_hvac_params_conductance)
    g_lb = {r: 1.0 / (1.3 * float(getattr(dyn, r))) for r in _R_TO_G}
    g_ub = {r: 1.0 / (0.7 * float(getattr(dyn, r))) for r in _R_TO_G}

    dynamics_params = HydronicDynamicsParameters(
        gAw=_denorm(param_manager.get("gAw"), 0.7 * dyn.gAw, 1.3 * dyn.gAw),
        Ch=_denorm(param_manager.get("Ch"), 0.7 * dyn.Ch, 1.3 * dyn.Ch),
        Ci=_denorm(param_manager.get("Ci"), 0.7 * dyn.Ci, 1.3 * dyn.Ci),
        Ce=_denorm(param_manager.get("Ce"), 0.7 * dyn.Ce, 1.3 * dyn.Ce),
        Rhi=1.0 / _denorm(param_manager.get("ghi"), g_lb["Rhi"], g_ub["Rhi"]),
        Rie=1.0 / _denorm(param_manager.get("gie"), g_lb["Rie"], g_ub["Rie"]),
        Rea=1.0 / _denorm(param_manager.get("gea"), g_lb["Rea"], g_ub["Rea"]),
    )
    return _build_ocp(dynamics_params, param_manager, N_horizon, name, x0)


def export_parametric_ocp_normalized(
    param_manager: AcadosParameterManager,
    N_horizon: int,
    name: str = "hvac_normalized",
    x0: np.ndarray | None = None,
    hydronic_params: HydronicParameters | None = None,
) -> AcadosOcp:
    """Export the HVAC OCP with all learnable parameters normalised to [−1, 1].

    The OCP internally maps each normalised parameter back to its physical value
        p_phys = lb + (p_norm + 1)/2 · (ub − lb)
    before evaluating the dynamics.  The physical bounds are re-derived from
    ``hydronic_params`` using the same scaling fractions as ``make_default_hvac_params``.

    Args:
        param_manager: Parameter manager built from ``make_default_hvac_params_normalized``.
        N_horizon: Number of time steps in the horizon.
        name: Acados model name.
        x0: Initial state.  Uses a 20 °C default when ``None``.
        hydronic_params: Nominal hydronic parameters used to recover physical bounds.
            Must match the parameters used when building ``param_manager``.

    Returns:
        Configured AcadosOcp ready for solver creation.
    """
    hydronic_params = HydronicParameters() if hydronic_params is None else hydronic_params
    dyn = hydronic_params.dynamics

    def _denorm(sym: ca.SX, lb: float, ub: float) -> ca.SX:
        """Map a normalised symbol from [-1, 1] to the physical interval [lb, ub]."""
        return lb + (sym + 1.0) / 2.0 * (ub - lb)

    dynamics_params = HydronicDynamicsParameters(
        gAw=_denorm(param_manager.get("gAw"), 0.7 * dyn.gAw, 1.3 * dyn.gAw),
        Ch=_denorm(param_manager.get("Ch"), 0.7 * dyn.Ch, 1.3 * dyn.Ch),
        Ci=_denorm(param_manager.get("Ci"), 0.7 * dyn.Ci, 1.3 * dyn.Ci),
        Ce=_denorm(param_manager.get("Ce"), 0.7 * dyn.Ce, 1.3 * dyn.Ce),
        Rea=_denorm(param_manager.get("Rea"), 0.7 * dyn.Rea, 1.3 * dyn.Rea),
        Rhi=_denorm(param_manager.get("Rhi"), 0.7 * dyn.Rhi, 1.3 * dyn.Rhi),
        Rie=_denorm(param_manager.get("Rie"), 0.7 * dyn.Rie, 1.3 * dyn.Rie),
    )
    return _build_ocp(dynamics_params, param_manager, N_horizon, name, x0)


def make_default_hvac_params_parameter_linear(
    interface: HvacAcadosParamInterface,
    granularity: HvacAcadosParamGranularity,
    N_horizon: int = 96,
    hydronic_params: HydronicParameters | None = None,
) -> tuple[AcadosParameter, ...]:
    """Return HVAC parameters that appear linearly in the continuous-time dynamics.

    Replaces the seven original physical parameters with seven compound parameters
    that are the literal coefficients of the continuous-time Ac / Bc / Ec matrices:

      ahi    = 1/(Ci·Rhi)   Ac[0,1], −Ac[0,0] contribution
      aie    = 1/(Ci·Rie)   Ac[0,2], −Ac[0,0] contribution
      bhi    = 1/(Ch·Rhi)   Ac[1,0], −Ac[1,1]
      cie    = 1/(Ce·Rie)   Ac[2,0], −Ac[2,2] contribution
      cea    = 1/(Ce·Rea)   Ac[2,2] contribution, Ec[2,0]
      ch_inv = 1/Ch          Bc[1,0]
      gaw_ci = gAw/Ci        Ec[0,1]

    Each compound parameter appears linearly in the dynamics, so gradients
    ∂(dynamics)/∂θ are constant (no 1/R² amplification).

    Defaults are computed from the physical defaults (0.8 × nominal for each
    underlying physical parameter).  Bounds span the full range of compound
    values reachable when each physical parameter independently varies in
    [0.7, 1.3] × nominal.

    Args:
        interface: Parameter interface type.
        granularity: Parameter granularity.
        N_horizon: MPC horizon length.
        hydronic_params: Optional nominal hydronic parameters.

    Returns:
        Tuple of AcadosParameter objects with parameter-linear dynamics parameters.
    """
    hydronic_params = HydronicParameters() if hydronic_params is None else hydronic_params
    dyn = hydronic_params.dynamics

    s = 0.8  # same default scaling used by all other variants

    def _rc(C_nom: float, R_nom: float) -> tuple[float, float, float]:
        """(default, lb, ub) for a 1/(C·R) compound parameter."""
        return (
            1.0 / (s * C_nom * s * R_nom),  # default: both C, R at 0.8·nom
            1.0 / (1.3 * C_nom * 1.3 * R_nom),  # lb:      both C, R at upper bound
            1.0 / (0.7 * C_nom * 0.7 * R_nom),  # ub:      both C, R at lower bound
        )

    def _inv(C_nom: float) -> tuple[float, float, float]:
        """(default, lb, ub) for a 1/C compound parameter."""
        return (
            1.0 / (s * C_nom),
            1.0 / (1.3 * C_nom),
            1.0 / (0.7 * C_nom),
        )

    def _ratio(A_nom: float, B_nom: float) -> tuple[float, float, float]:
        """(default, lb, ub) for an A/B compound parameter."""
        return (
            (s * A_nom) / (s * B_nom),  # = A_nom/B_nom  (s cancels)
            (0.7 * A_nom) / (1.3 * B_nom),  # min A / max B
            (1.3 * A_nom) / (0.7 * B_nom),  # max A / min B
        )

    compound_specs: dict[str, tuple[float, float, float]] = {
        "ahi": _rc(dyn.Ci, dyn.Rhi),
        "aie": _rc(dyn.Ci, dyn.Rie),
        "bhi": _rc(dyn.Ch, dyn.Rhi),
        "cie": _rc(dyn.Ce, dyn.Rie),
        "cea": _rc(dyn.Ce, dyn.Rea),
        "ch_inv": _inv(dyn.Ch),
        "gaw_ci": _ratio(dyn.gAw, dyn.Ci),
    }

    is_dyn_learnable = "dynamics" in interface
    params: list[AcadosParameter] = []
    for pname, (default, lb, ub) in compound_specs.items():
        params.append(
            AcadosParameter(
                name=pname,
                default=np.array([default]),
                space=gym.spaces.Box(low=np.array([lb]), high=np.array([ub]), dtype=np.float64),
                interface="learnable" if is_dyn_learnable else "fix",
            )
        )

    # Append all non-dynamics params from the standard parameterisation unchanged.
    original = make_default_hvac_params(interface, granularity, N_horizon, hydronic_params)
    from dataclasses import asdict

    dyn_names = set(asdict(dyn).keys())
    params.extend(p for p in original if p.name not in dyn_names)

    return tuple(params)


def export_parametric_ocp_parameter_linear(
    param_manager: AcadosParameterManager,
    N_horizon: int,
    name: str = "hvac_pl",
    x0: np.ndarray | None = None,
) -> AcadosOcp:
    """Export the HVAC OCP with parameter-linear dynamics.

    The 7 compound learnable parameters (ahi, aie, bhi, cie, cea, ch_inv, gaw_ci)
    are the literal coefficients of the continuous-time state-space matrices, so
    ∂(dynamics)/∂θ is constant — no 1/R² gradient amplification.

    The continuous-time model is::

        dTi/dt = -(ahi+aie)·Ti + ahi·Th + aie·Te             + gaw_ci·Is
        dTh/dt =  bhi·Ti - bhi·Th                + ch_inv·qh
        dTe/dt =  cie·Ti         -(cie+cea)·Te   + cea·Ta

    which is discretised with a zero-order hold at dt = 900 s.

    Args:
        param_manager: Parameter manager from make_default_hvac_params_parameter_linear.
        N_horizon: Number of time steps in the horizon.
        name: Acados model name.
        x0: Initial state.  Uses a 20 °C default when ``None``.

    Returns:
        Configured AcadosOcp ready for solver creation.
    """
    dt: float = 900.0

    ahi = param_manager.get("ahi")
    aie = param_manager.get("aie")
    bhi = param_manager.get("bhi")
    cie = param_manager.get("cie")
    cea = param_manager.get("cea")
    ch_inv = param_manager.get("ch_inv")
    gaw_ci = param_manager.get("gaw_ci")

    # Continuous-time state-space  x = [Ti, Th, Te]
    Ac = ca.SX.zeros(3, 3)
    Ac[0, 0] = -(ahi + aie)
    Ac[0, 1] = ahi
    Ac[0, 2] = aie
    Ac[1, 0] = bhi
    Ac[1, 1] = -bhi
    Ac[2, 0] = cie
    Ac[2, 2] = -(cie + cea)

    Bc = ca.SX.zeros(3, 1)
    Bc[1, 0] = ch_inv  # qh input in W

    Ec = ca.SX.zeros(3, 2)
    Ec[0, 1] = gaw_ci  # solar radiation
    Ec[2, 0] = cea  # outdoor temperature

    # ZOH discretisation
    Ad = expm_pade66_robust(Ac * dt)
    Bd = ca.mtimes(ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Bc)
    Ed = ca.mtimes(ca.mtimes(ca.inv(Ac), (Ad - ca.SX.eye(3))), Ec)

    return _build_ocp_with_matrices(Ad, Bd * 1e3, Ed, param_manager, N_horizon, name, x0)


def make_default_hvac_params_discrete_matrix(
    interface: HvacAcadosParamInterface,
    granularity: HvacAcadosParamGranularity,
    N_horizon: int = 96,
    hydronic_params: HydronicParameters | None = None,
    dt: float = 900.0,
) -> tuple[AcadosParameter, ...]:
    """Return HVAC parameters that directly represent the normalised discrete matrices.

    Instead of learning physical parameters, the 18 elements of the ZOH-discretised
    matrices Ad (3×3), Bd (3×1), Ed (3×2) are treated as learnable parameters,
    each normalised to [−1, 1] with ±30% relative bounds around a reference value.

    **Reference point**: discrete matrices computed from 0.8 × nominal physical
    parameters, matching the default operating point of all other variants.

    **Normalisation**::

        θ_norm = (M_phys[i,j] / M_ref[i,j] − 1) / 0.3   ∈ [−1, 1]
        M_phys[i,j] = M_ref[i,j] · (1 + 0.3 · θ_norm)

    The default θ_norm = 0 for every element, corresponding exactly to M_ref.

    Parameter names: ``ad{i}{j}`` (Ad), ``bd{i}`` (Bd), ``ed{i}{j}`` (Ed).
    18 learnable parameters in total.

    Args:
        interface: Parameter interface type.
        granularity: Parameter granularity.
        N_horizon: MPC horizon length.
        hydronic_params: Optional nominal hydronic parameters.
        dt: Sampling period in seconds (default 900 s = 15 min).

    Returns:
        Tuple of AcadosParameter objects.
    """
    hydronic_params = HydronicParameters() if hydronic_params is None else hydronic_params
    dyn = hydronic_params.dynamics

    # Reference: 0.8× scaled physical params, same default as all other variants
    s = 0.8
    ref_dyn = HydronicDynamicsParameters(
        gAw=s * dyn.gAw,
        Ch=s * dyn.Ch,
        Ci=s * dyn.Ci,
        Ce=s * dyn.Ce,
        Rhi=s * dyn.Rhi,
        Rie=s * dyn.Rie,
        Rea=s * dyn.Rea,
    )
    Ad_ref, Bd_ref, Ed_ref = transcribe_discrete_state_space(dt=dt, params=ref_dyn)

    is_dyn_learnable = "dynamics" in interface
    params: list[AcadosParameter] = []

    for i in range(3):
        for j in range(3):
            params.append(
                AcadosParameter(
                    name=f"ad{i}{j}",
                    default=np.array([0.0]),
                    space=gym.spaces.Box(
                        low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float64
                    ),
                    interface="learnable" if is_dyn_learnable else "fix",
                )
            )
    for i in range(3):
        params.append(
            AcadosParameter(
                name=f"bd{i}",
                default=np.array([0.0]),
                space=gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float64),
                interface="learnable" if is_dyn_learnable else "fix",
            )
        )
    for i in range(3):
        for j in range(2):
            params.append(
                AcadosParameter(
                    name=f"ed{i}{j}",
                    default=np.array([0.0]),
                    space=gym.spaces.Box(
                        low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float64
                    ),
                    interface="learnable" if is_dyn_learnable else "fix",
                )
            )

    # Non-dynamics params identical to the standard parameterisation.
    original = make_default_hvac_params(interface, granularity, N_horizon, hydronic_params)
    from dataclasses import asdict

    dyn_names = set(asdict(dyn).keys())
    params.extend(p for p in original if p.name not in dyn_names)

    return tuple(params)


def export_parametric_ocp_discrete_matrix(
    param_manager: AcadosParameterManager,
    N_horizon: int,
    name: str = "hvac_dm",
    x0: np.ndarray | None = None,
    hydronic_params: HydronicParameters | None = None,
    dt: float = 900.0,
) -> AcadosOcp:
    """Export the HVAC OCP where the learnable parameters ARE the normalised discrete matrices.

    Each element of Ad, Bd, Ed is recovered from its normalised parameter via::

        M_phys[i,j] = M_ref[i,j] · (1 + 0.3 · θ_norm[i,j])

    where M_ref is the discrete matrix evaluated at 0.8 × nominal physical parameters.
    This makes the dynamics **affine** in the normalised parameters and gives uniform
    gradient magnitudes across all 18 learnable parameters.

    Args:
        param_manager: Parameter manager from make_default_hvac_params_discrete_matrix.
        N_horizon: Number of time steps in the horizon.
        name: Acados model name.
        x0: Initial state.  Uses a 20 °C default when ``None``.
        hydronic_params: Nominal hydronic parameters used to compute M_ref.
        dt: Sampling period (must match make_default_hvac_params_discrete_matrix).

    Returns:
        Configured AcadosOcp ready for solver creation.
    """
    hydronic_params = HydronicParameters() if hydronic_params is None else hydronic_params
    dyn = hydronic_params.dynamics

    s = 0.8
    ref_dyn = HydronicDynamicsParameters(
        gAw=s * dyn.gAw,
        Ch=s * dyn.Ch,
        Ci=s * dyn.Ci,
        Ce=s * dyn.Ce,
        Rhi=s * dyn.Rhi,
        Rie=s * dyn.Rie,
        Rea=s * dyn.Rea,
    )
    Ad_ref, Bd_ref, Ed_ref = transcribe_discrete_state_space(dt=dt, params=ref_dyn)

    def _denorm(sym: ca.SX, ref_val: float) -> ca.SX:
        return ref_val * (1.0 + 0.3 * sym)

    Ad = ca.SX.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            Ad[i, j] = _denorm(param_manager.get(f"ad{i}{j}"), float(Ad_ref[i, j]))

    Bd = ca.SX.zeros(3, 1)
    for i in range(3):
        Bd[i, 0] = _denorm(param_manager.get(f"bd{i}"), float(Bd_ref[i, 0]))

    Ed = ca.SX.zeros(3, 2)
    for i in range(3):
        for j in range(2):
            Ed[i, j] = _denorm(param_manager.get(f"ed{i}{j}"), float(Ed_ref[i, j]))

    return _build_ocp_with_matrices(Ad, Bd * 1e3, Ed, param_manager, N_horizon, name, x0)
