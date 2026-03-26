"""Acados OCP equivalent to the i4b MPC optimization problem.

OCP formulation (matching external/i4b/src/controller/mpc/optimization_problem.py):

  State (4R3C example):  x = [T_room, T_wall, T_hp_ret]      [degC]
  Control:               u = [T_HP]                           [degC]
  Parameters per stage:  p = [T_amb, Qdot_gains, T_set_lower, grid_signal]

  Stage cost:  Qth / (COP(T_HP, T_amb) * 100) * grid_signal
               where Qth = mdot_HP * c_water * (T_HP - T_hp_ret) / 1000  [kW]

  Soft constraints on T_room (lower and upper), with quadratic slack penalty ws*s^2.
  Hard constraint on HP thermal power: 0 <= Qth <= 26 kW.
  Hard box constraint on T_HP: 0 <= T_HP <= 65 degC.

Building models (from external/i4b/src/models/model_buildings.py::Building):
  "2R2C"  nx=2  x = [T_room, T_hp_ret]
  "4R3C"  nx=3  x = [T_room, T_wall, T_hp_ret]
  "5R4C"  nx=4  x = [T_room, T_int, T_wall, T_hp_ret]

Heat pump models (from external/i4b/src/models/model_hvac.py):
  Heatpump_AW      - Dimplex LA 6TU air-water HP
  Heatpump_Vitocal - Vitocal ground-water HP
"""

import casadi as ca
import gymnasium as gym
import numpy as np
import scipy.linalg
from acados_template import ACADOS_INFTY, AcadosOcp
from i4b.constants import C_WATER_SPEC
from i4b.models.model_buildings import Building
from i4b.models.model_hvac import Heatpump, Heatpump_AW, Heatpump_Vitocal

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager


def make_i4b_params(N_horizon: int) -> tuple[AcadosParameter, ...]:
    """Return the four stage-wise non-learnable parameters for the i4b OCP.

    All parameters use ``interface="non-learnable"``: they can be updated at
    runtime via ``solver.set(k, "p", values)`` but are not exposed to a
    learning interface.  Default values are representative winter conditions.

    Parameter order (= flat ``p`` vector per stage):
        0  T_amb        Ambient temperature               [degC]   default  5.0
        1  Qdot_gains   Total heat gains (solar+internal) [W]      default  500.0
        2  T_set_lower  Lower comfort setpoint            [degC]   default  20.0
        3  grid_signal  Grid support signal               [-]      default  1.0

    Returns:
        Tuple of four AcadosParameter objects (preserving the order above).
    """
    return (
        AcadosParameter(
            name="T_amb",
            default=np.array([5.0]),
            space=gym.spaces.Box(low=np.array([-25.0]), high=np.array([45.0]), dtype=np.float64),
            interface="non-learnable",
        ),
        AcadosParameter(
            name="Qdot_gains",
            default=np.array([500.0]),
            space=gym.spaces.Box(low=np.array([0.0]), high=np.array([8000.0]), dtype=np.float64),
            interface="non-learnable",
        ),
        AcadosParameter(
            name="T_set_lower",
            default=np.array([20.0]),
            space=gym.spaces.Box(low=np.array([15.0]), high=np.array([30.0]), dtype=np.float64),
            interface="non-learnable",
        ),
        AcadosParameter(
            name="T_set_upper",
            default=np.array([26.0]),
            space=gym.spaces.Box(low=np.array([20.0]), high=np.array([30.0]), dtype=np.float64),
            interface="non-learnable",
        ),
        AcadosParameter(
            name="grid_signal",
            default=np.array([1.0]),
            space=gym.spaces.Box(low=np.array([0.0]), high=np.array([5.0]), dtype=np.float64),
            interface="non-learnable",
        ),
        AcadosParameter(
            name="int_gains",
            default=np.array([0.0]),
            # space=gym.spaces.Box(low=np.array([-25.0]), high=np.array([45.0]), dtype=np.float64),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)),
        ),
    )


def _cop_casadi(hp_model: Heatpump, T_HP: ca.SX, T_amb: ca.SX) -> ca.SX:
    """COP as a CasADi expression, matching the HP model's .COP() method."""
    if isinstance(hp_model, Heatpump_AW):
        a = np.array((8.2553, -0.17068, 0.16176, 0.00108, 0.00022, -0.00186))
        return (
            a[0]
            + a[1] * T_HP
            + a[2] * T_amb
            + a[3] * T_HP**2
            + a[4] * T_amb**2
            + a[5] * T_HP * T_amb
        )
    elif isinstance(hp_model, Heatpump_Vitocal):
        # Ground temperature from ambient: 6.645*tanh(0.188*(T_amb - 9.177)) + 7.872
        T_source = 6.645 * ca.tanh(0.188 * (T_amb - 9.177)) + 7.872
        z0, a, b, c, d, f = 10.893436, -0.228602, 0.266006, 0.001461, 0.000501, -0.003546
        return z0 + a * T_HP + b * T_source + c * T_HP**2 + d * T_source**2 + f * T_HP * T_source
    else:
        raise ValueError(f"Unsupported HP model type: {type(hp_model).__name__}")


def transcribe_i4b_continuous_state_space(
    f_expl: ca.SX,
    x: ca.SX,
    u: ca.SX,
    disturbances: list[ca.SX],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract continuous-time LTI matrices by evaluating the (constant) Jacobians of f_expl.

    Requires that f_expl is affine in x, u, and disturbances (i.e. LTI up to parameters).

        dx/dt = Ac @ x + Bc @ u + Ec @ disturbances

    Each entry of ``disturbances`` must be a purely symbolic scalar or vector.
    ``ca.jacobian`` requires purely symbolic differentiation variables; passing
    ``ca.vertcat(...)`` of indexed struct elements would fail, so a list is used
    and the column Jacobians are concatenated afterwards.

    Args:
        f_expl: Continuous-time right-hand side, shape (nx,).
        x: State symbolic vector, shape (nx,).
        u: Control symbolic vector, shape (nu,).
        disturbances: List of purely symbolic disturbance variables, e.g.
            [T_amb, Qdot_gains, int_gains]. One column per entry in Ec.

    Returns:
        Ac: State matrix,        shape (nx, nx).
        Bc: Control matrix,      shape (nx, nu).
        Ec: Disturbance matrix,  shape (nx, nd).
    """
    Ac = np.array(ca.evalf(ca.jacobian(f_expl, x)))
    Bc = np.array(ca.evalf(ca.jacobian(f_expl, u)))
    Ec = np.array(ca.evalf(ca.jacobian(f_expl, disturbances)))
    return Ac, Bc, Ec


def transcribe_i4b_discrete_state_space(
    dt: float,
    Ac: np.ndarray,
    Bc: np.ndarray,
    Ec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize a continuous LTI system via zero-order hold (ZOH).

    Uses the matrix-exponential formula:

        Ad = expm(Ac * dt)
        Bd = inv(Ac) @ (Ad - I) @ Bc
        Ed = inv(Ac) @ (Ad - I) @ Ec

    Args:
        dt: Sampling time [s].
        Ac: Continuous state matrix,        shape (nx, nx).
        Bc: Continuous control matrix,      shape (nx, nu).
        Ec: Continuous disturbance matrix,  shape (nx, nd).

    Returns:
        Ad: Discrete state matrix,        shape (nx, nx).
        Bd: Discrete control matrix,      shape (nx, nu).
        Ed: Discrete disturbance matrix,  shape (nx, nd).
    """
    nx = Ac.shape[0]
    Ad = scipy.linalg.expm(Ac * dt)
    M = np.linalg.solve(Ac, Ad - np.eye(nx))  # inv(Ac) @ (Ad - I)
    return Ad, M @ Bc, M @ Ec


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    N_horizon: int,
    building_model: Building,
    hp_model: Heatpump,
    name: str = "i4b",
    x0: np.ndarray | None = None,
    ws: float = 0.1,
    delta_t: float = 900.0,
) -> AcadosOcp:
    """Export an AcadosOcp equivalent to the i4b MPC optimization problem.

    Args:
        param_manager: Parameter manager built from make_i4b_params().
            Provides symbolic variables for T_amb, Qdot_gains, T_set_lower,
            grid_signal and assigns them to the OCP's p vector.
        N_horizon: Number of shooting intervals.
        building_model: Instantiated Building model (sets method, params, mdot_hp).
        hp_model: Instantiated Heatpump model (Heatpump_AW or Heatpump_Vitocal).
        name: Acados model name (used for generated C code).
        x0: Initial state. Defaults to 20 degC for all states.
        ws: Quadratic weight on soft-constraint slack variables.
        delta_t: Sampling time in seconds (default 900 = 15 min).

    Returns:
        Configured AcadosOcp (not yet compiled into a solver).
    """
    nx = len(building_model.state_keys)

    ocp = AcadosOcp()
    ocp.model.name = name

    # Assign p vector from param manager (sets ocp.model.p and ocp.parameter_values)
    param_manager.assign_to_ocp(ocp)

    # ── Symbolic state / control ──────────────────────────────────────────────
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", 1)  # T_HP [degC]

    T_HP = u[0]
    T_room = x[0]
    T_hp_ret = x[-1]

    ocp.model.x = x
    ocp.model.u = u

    # ── Parameters (via manager) ──────────────────────────────────────────────
    T_amb = param_manager.get("T_amb")
    Qdot_gains = param_manager.get("Qdot_gains")
    T_set_lower = param_manager.get("T_set_lower")
    grid_signal = param_manager.get("grid_signal")

    # ── Continuous dynamics ───────────────────────────────────────────────────
    # building_model.calc_casadi reads p[0]=T_amb and p[1]=Qdot_gains
    p_bldg = ca.vertcat(T_amb, Qdot_gains)
    f_expl_expr = building_model.calc_casadi(x, T_HP, p_bldg)

    # int_gains [W] is a learnable offset to Qdot_gains, entering T_room via 1/C_room [J/K].
    # The capacity key depends on the building method (matches calc_*_casadi).
    # _C_room_key = {"2R2C": "C_bldg", "4R3C": "C_zone", "5R4C": "C_air"}
    # C_room = building_model.params[_C_room_key[building_model.method]]
    # ocp.model.f_expl_expr[0] += param_manager.get("int_gains") / C_room

    # ── Verify dynamics are affine in x and u (LTI up to parameters) ────────────
    d = ca.vertcat(T_amb, Qdot_gains)
    # d = ca.vertcat(T_amb)
    J_x = ca.jacobian(f_expl_expr, x)
    J_u = ca.jacobian(f_expl_expr, u)
    J_d = ca.jacobian(f_expl_expr, d)
    assert not ca.depends_on(J_x, x), "f_expl_expr is not affine in x"
    assert not ca.depends_on(J_x, u), "A matrix depends on u (not LTI)"
    assert not ca.depends_on(J_u, u), "f_expl_expr is not affine in u"
    assert not ca.depends_on(J_u, x), "B matrix depends on x (not LTI)"
    assert not ca.depends_on(J_d, x), "f_expl_expr depends nonlinearly on d and x"
    assert not ca.depends_on(J_d, u), "f_expl_expr depends nonlinearly on d and u"

    # ── Discrete state-space (ZOH) ───────────────────────────────────────────
    Ac_np, Bc_np, Ec_np = transcribe_i4b_continuous_state_space(f_expl_expr, x, u, d)
    Ad, Bd, Ed = transcribe_i4b_discrete_state_space(delta_t, Ac_np, Bc_np, Ec_np)

    ocp.model.disc_dyn_expr = Ad @ x + Bd @ u + Ed @ d

    # ── Stage cost: electrical energy ─────────────────────────────────────────
    COP = _cop_casadi(hp_model, T_HP, T_amb)
    Qth = hp_model.mdot_HP * C_WATER_SPEC * (T_HP - T_hp_ret) / 1000  # [kW]
    stage_cost = Qth / (COP * 100) * grid_signal

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = stage_cost
    ocp.model.cost_expr_ext_cost_e = ca.SX(0)

    # ── Nonlinear path constraints ────────────────────────────────────────────
    # h[0]: soft lower bound on T_room  (T_room >= T_set_lower)
    # h[1]: soft upper bound on T_room  (T_room <= T_set_upper)
    # h[2]: hard HP thermal power       (0 <= Qth <= 26 kW)
    T_set_upper = param_manager.get("T_set_upper")
    h_expr = ca.vertcat(
        T_room - T_set_lower,
        T_set_upper - T_room,
        Qth,
    )
    ocp.model.con_h_expr = h_expr

    ocp.constraints.lh = np.array([0.0, 0.0, 0.0])
    ocp.constraints.uh = np.array([ACADOS_INFTY, ACADOS_INFTY, 26.0])

    # Terminal: room temperature constraints only (no control at terminal stage)
    ocp.model.con_h_expr_e = h_expr[:2]
    ocp.constraints.lh_e = np.array([0.0, 0.0])
    ocp.constraints.uh_e = np.array([ACADOS_INFTY, ACADOS_INFTY])

    # ── Soft constraints (h[0] and h[1] softened) ────────────────────────────
    # Slack penalty: ws * sl^2  →  Zl = ws, zl = 0
    ocp.constraints.idxsh = np.array([0, 1])
    ocp.constraints.idxsh_e = np.array([0, 1])

    ocp.cost.zl = np.zeros(2)
    ocp.cost.zu = np.zeros(2)
    ocp.cost.Zl = ws * np.ones(2)
    ocp.cost.Zu = ws * np.ones(2)
    ocp.cost.zl_e = np.zeros(2)
    ocp.cost.zu_e = np.zeros(2)
    ocp.cost.Zl_e = ws * np.ones(2)
    ocp.cost.Zu_e = ws * np.ones(2)

    # ── Control bounds ────────────────────────────────────────────────────────
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([65.0])
    ocp.constraints.idxbu = np.array([0])

    # ── Initial state ─────────────────────────────────────────────────────────
    ocp.constraints.x0 = x0 if x0 is not None else 20.0 * np.ones(nx)

    # ── Solver options ────────────────────────────────────────────────────────
    ocp.solver_options.tf = N_horizon * delta_t
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.print_level = 0

    return ocp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from acados_template import AcadosOcpSolver
    from i4b_data.buildings.sfh_2016_now import sfh_2016_now_0_soc

    METHOD = "4R3C"
    N = 96  # 96 x 900 s = 24 h
    DELTA_T = 900.0

    building = Building(
        params=sfh_2016_now_0_soc,
        mdot_hp=0.25,
        method=METHOD,
        T_room_set_lower=20.0,
        T_room_set_upper=26.0,
    )
    hp = Heatpump_AW(mdot_HP=0.25)

    x0 = 15.0 * np.ones(len(building.state_keys))

    # ── Build OCP via param manager ───────────────────────────────────────────
    param_manager = AcadosParameterManager(make_i4b_params(N_horizon=N), N_horizon=N)
    ocp = export_parametric_ocp(param_manager, N, building, hp, x0=x0)
    solver = AcadosOcpSolver(ocp, json_file="i4b_ocp.json")

    # ── Set stagewise parameters: cold winter day ─────────────────────────────
    p_stagewise = param_manager.combine_non_learnable_parameter_values(
        batch_size=1,
        T_amb=np.full((1, N + 1, 1), -5.0),
        Qdot_gains=np.full((1, N + 1, 1), 300.0),
        T_set_lower=np.full((1, N + 1, 1), 20.0),
        T_set_upper=np.full((1, N + 1, 1), 26.0),
        grid_signal=np.full((1, N + 1, 1), 1.0),
    )  # shape (1, N+1, p_dim)
    for k in range(N + 1):
        solver.set(k, "p", p_stagewise[0, k])

    # ── Solve ─────────────────────────────────────────────────────────────────
    status = solver.solve()
    if status != 0:
        print(f"WARNING: solver returned status {status}")

    # ── Extract and plot solution ─────────────────────────────────────────────
    X = np.array([solver.get(k, "x") for k in range(N + 1)])
    U = np.array([solver.get(k, "u") for k in range(N)])
    t = np.arange(N + 1) * DELTA_T / 3600

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax0 = axes[0]
    for i, label in enumerate(building.state_keys):
        ax0.plot(t, X[:, i], label=label)
    ax0.axhline(20.0, color="k", linestyle="--", linewidth=0.8, label="T_set_lower")
    ax0.axhline(
        building.T_room_set_upper, color="r", linestyle="--", linewidth=0.8, label="T_set_upper"
    )
    ax0.set_ylabel("Temperature [degC]")
    ax0.legend(fontsize=8)
    ax0.set_title(f"i4b MPC OCP  |  {METHOD}  |  N={N}  |  T_amb={p_stagewise[0, 0, 0]:.1f} degC")
    ax0.grid(True)

    ax1 = axes[1]
    ax1.step(t[:-1], U[:, 0], where="post", label="T_HP (supply)")
    ax1.set_ylabel("T_HP [degC]")
    ax1.set_xlabel("Time [h]")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig("i4b_ocp_solution.png", dpi=120)
    print("Saved i4b_ocp_solution.png")
    plt.show()
