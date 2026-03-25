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

import os
import sys
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameter

# i4b root - needed for Building/Heatpump model imports
_I4B_ROOT = Path(__file__).resolve().parents[3] / "external" / "i4b"
if str(_I4B_ROOT) not in sys.path:
    sys.path.insert(0, str(_I4B_ROOT))

from src.constants import C_WATER_SPEC  # noqa: E402
from src.models.model_buildings import Building  # noqa: E402
from src.models.model_hvac import Heatpump, Heatpump_AW, Heatpump_Vitocal  # noqa: E402


def make_i4b_params(
    param_interface: str,
    N_horizon: int,
) -> list[AcadosParameter]:
    pass


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


def export_parametric_ocp(
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

    # ── Symbolic variables ────────────────────────────────────────────────────
    x = ca.SX.sym("x", nx)  # states
    u = ca.SX.sym("u", 1)  # control: T_HP [degC]
    # p = [T_amb, Qdot_gains, T_set_lower, grid_signal]
    p = ca.SX.sym("p", 4)

    T_amb = p[0]
    Qdot_gains = p[1]  # noqa: F841  (accessed inside building_model via p[1])
    T_set_lower = p[2]
    grid_signal = p[3]

    T_HP = u[0]
    T_room = x[0]
    T_hp_ret = x[-1]

    ocp.model.x = x
    ocp.model.u = u
    ocp.model.p = p

    # ── Continuous dynamics (building_model uses p[0]=T_amb, p[1]=Qdot_gains) ─
    # p has 4 entries; the building model only reads p[0] and p[1], so passing
    # the full vector is safe for all three methods (2R2C, 4R3C, 5R4C).
    ocp.model.f_expl_expr = building_model.calc_casadi(x, T_HP, p)

    # ── Stage cost: electrical energy ─────────────────────────────────────────
    COP = _cop_casadi(hp_model, T_HP, T_amb)
    Qth = hp_model.mdot_HP * C_WATER_SPEC * (T_HP - T_hp_ret) / 1000  # [kW]
    stage_cost = Qth / (COP * 100) * grid_signal

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = stage_cost
    ocp.model.cost_expr_ext_cost_e = ca.SX(0)  # no explicit terminal cost

    # ── Nonlinear path constraints ────────────────────────────────────────────
    # h[0]: soft lower bound on T_room  (T_room >= T_set_lower)
    # h[1]: soft upper bound on T_room  (T_room <= T_set_upper)
    # h[2]: hard HP thermal power       (0 <= Qth <= 26 kW)
    T_set_upper = building_model.T_room_set_upper
    h_expr = ca.vertcat(
        T_room - T_set_lower,
        T_set_upper - T_room,
        Qth,
    )
    ocp.model.con_h_expr = h_expr

    ocp.constraints.lh = np.array([0.0, 0.0, 0.0])
    ocp.constraints.uh = np.array([1e9, 1e9, 26.0])

    # Terminal: room temperature constraints only (no control at terminal stage)
    ocp.model.con_h_expr_e = h_expr[:2]
    ocp.constraints.lh_e = np.array([0.0, 0.0])
    ocp.constraints.uh_e = np.array([1e9, 1e9])

    # ── Soft constraints (h[0] and h[1] are softened) ─────────────────────────
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

    # ── Control bounds (hard box) ─────────────────────────────────────────────
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([65.0])
    ocp.constraints.idxbu = np.array([0])

    # ── Initial state ─────────────────────────────────────────────────────────
    ocp.constraints.x0 = x0 if x0 is not None else 20.0 * np.ones(nx)

    # ── Default parameter values ──────────────────────────────────────────────
    # [T_amb=5°C, Qdot_gains=500W, T_set_lower=20°C, grid_signal=1]
    ocp.parameter_values = np.array([5.0, 500.0, 20.0, 1.0])

    # ── Solver options ────────────────────────────────────────────────────────
    ocp.solver_options.tf = N_horizon * delta_t
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4  # RK4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.print_level = 0

    return ocp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from acados_template import AcadosOcpSolver

    # Must chdir to i4b root so its internal data imports resolve
    os.chdir(_I4B_ROOT)

    # ── Select building and HP model ──────────────────────────────────────────
    from data.buildings.sfh_2016_now import sfh_2016_now_0_soc

    METHOD = "4R3C"
    N = 96  # 96 steps x 900s = 24 h
    DELTA_T = 900.0  # 15 min

    building = Building(
        params=sfh_2016_now_0_soc,
        mdot_hp=0.25,
        method=METHOD,
        T_room_set_lower=20.0,
        T_room_set_upper=26.0,
    )
    hp = Heatpump_AW(mdot_HP=0.25)

    # Cold initial condition: house at 15 degC
    nx = len(building.state_keys)
    x0 = 15.0 * np.ones(nx)

    # ── Build and compile OCP ─────────────────────────────────────────────────
    ocp = export_parametric_ocp(N, building, hp, x0=x0)
    solver = AcadosOcpSolver(ocp, json_file="i4b_ocp.json")

    # ── Set stagewise parameters: cold winter day ─────────────────────────────
    # p = [T_amb, Qdot_gains, T_set_lower, grid_signal]
    p_stage = np.array([-5.0, 300.0, 20.0, 1.0])
    for k in range(N + 1):
        solver.set(k, "p", p_stage)

    # ── Solve ─────────────────────────────────────────────────────────────────
    status = solver.solve()
    if status != 0:
        print(f"WARNING: solver returned status {status}")

    # ── Extract solution ──────────────────────────────────────────────────────
    X = np.array([solver.get(k, "x") for k in range(N + 1)])
    U = np.array([solver.get(k, "u") for k in range(N)])
    t = np.arange(N + 1) * DELTA_T / 3600  # hours

    # ── Plot ──────────────────────────────────────────────────────────────────
    state_labels = list(building.state_keys)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax0 = axes[0]
    for i, label in enumerate(state_labels):
        ax0.plot(t, X[:, i], label=label)
    ax0.axhline(20.0, color="k", linestyle="--", linewidth=0.8, label="T_set_lower")
    ax0.axhline(26.0, color="r", linestyle="--", linewidth=0.8, label="T_set_upper")
    ax0.set_ylabel("Temperature [degC]")
    ax0.legend(fontsize=8)
    ax0.set_title(f"i4b MPC OCP solution  |  {METHOD}  |  N={N}  |  T_amb={p_stage[0]} degC")
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
