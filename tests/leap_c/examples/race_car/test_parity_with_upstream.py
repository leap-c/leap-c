"""Parity test: leap-c race_car vs upstream acados race_car (acados_settings_dev.py).

Drives both solvers through the same closed-loop update rule (identical to upstream
`external/acados/examples/acados_python/race_cars/main.py`) and asserts that the first
stage ``x`` and ``u`` trajectories agree element-wise to within 1e-6 over 100 steps.

The leap-c OCP is instantiated via ``export_parametric_ocp`` (production code path), then
its symbolic NONLINEAR_LS cost is materialized into numerical LINEAR_LS. This bypasses
``AcadosDiffMpcTorch`` (which applies a ``1/N`` discount) and matches upstream's code path
exactly; the rest of the OCP (integrator with substeps, soft-constraint layout, solver
options) is left as leap-c production code produces it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from acados_template import AcadosOcpSolver

from leap_c.examples.race_car.acados_ocp import (
    create_race_car_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.parameters import AcadosParameterManager

REPO_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_DIR = REPO_ROOT / "external/acados/examples/acados_python/race_cars"
TRACK = "LMS_Track.txt"

N_HORIZON = 50
T_HORIZON = 1.0
NSIM = 100
SREF_N = 3.0
TOL = 1e-6


def _load_upstream_module(name: str, filename: str):
    """Import a module file from UPSTREAM_DIR.

    Without polluting the global sys.path
    (upstream's ``bicycle_model.py`` does ``from casadi import *``).
    """
    sys_path_before = sys.path[:]
    try:
        sys.path.insert(0, str(UPSTREAM_DIR))
        spec = importlib.util.spec_from_file_location(name, UPSTREAM_DIR / filename)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = sys_path_before


def _materialize_linear_ls_cost(ocp, pm: AcadosParameterManager) -> None:
    """Replace leap-c's symbolic NONLINEAR_LS cost with numerical LINEAR_LS matching upstream.

    Uses the default parameter values from ``pm`` to compute numerical ``W`` and ``W_e``.
    After this call, the OCP can be passed to ``AcadosOcpSolver`` directly (no
    ``AcadosDiffMpcTorch`` indirection). ``yref`` / ``yref_e`` become numerical placeholders
    that will be overwritten per-stage in the closed-loop via ``solver.set(j, "yref", ...)``.
    """
    q_sqrt = pm.parameters["q_diag_sqrt"].default.flatten()
    r_sqrt = pm.parameters["r_diag_sqrt"].default.flatten()
    q_e_sqrt = pm.parameters["q_e_diag_sqrt"].default.flatten()

    nx = 6
    nu = 2
    ny = nx + nu
    ny_e = nx

    W = np.diag(np.concatenate([q_sqrt, r_sqrt]) ** 2)
    W_e = np.diag(q_e_sqrt**2)

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:ny, :nu] = np.eye(nu)
    Vx_e = np.eye(ny_e)

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = W
    ocp.cost.W_e = W_e
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = Vx_e
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)
    ocp.model.cost_y_expr = None
    ocp.model.cost_y_expr_e = None


def _build_leap_solver(workdir: Path) -> AcadosOcpSolver:
    pm = AcadosParameterManager(
        parameters=create_race_car_params(
            param_interface="global",
            N_horizon=N_HORIZON,
            T_horizon=T_HORIZON,
        ),
        N_horizon=N_HORIZON,
    )
    ocp = export_parametric_ocp(
        param_manager=pm,
        N_horizon=N_HORIZON,
        T_horizon=T_HORIZON,
    )
    _materialize_linear_ls_cost(ocp, pm)
    ocp.code_export_directory = str(workdir / "c_generated_code_leap")
    return AcadosOcpSolver(ocp, json_file=str(workdir / "leap_parity.json"))


def _build_upstream_solver(workdir: Path) -> AcadosOcpSolver:
    """Upstream ``acados_settings`` hardcodes ``json_file='acados_ocp.json'``.

    chdir into ``workdir`` so the generated artefacts land there and do not collide with
    the leap-c build.
    """
    cwd_before = os.getcwd()
    try:
        os.chdir(workdir)
        mod = _load_upstream_module("upstream_acados_settings_dev", "acados_settings_dev.py")
        _, _, solver = mod.acados_settings(T_HORIZON, N_HORIZON, TRACK)
        return solver
    finally:
        os.chdir(cwd_before)


@pytest.fixture(scope="module")
def solvers(tmp_path_factory: pytest.TempPathFactory):
    wd_up = tmp_path_factory.mktemp("upstream")
    wd_lc = tmp_path_factory.mktemp("leap_c")
    up = _build_upstream_solver(wd_up)
    lc = _build_leap_solver(wd_lc)
    return lc, up


def test_race_car_parity_with_upstream(solvers):
    lc, up = solvers
    nx = 6
    nu = 2

    simX_lc = np.zeros((NSIM, nx))
    simU_lc = np.zeros((NSIM, nu))
    simX_up = np.zeros((NSIM, nx))
    simU_up = np.zeros((NSIM, nu))

    # At step 0 both solvers start from ocp.constraints.x0 = [-2, 0, 0, 0, 0, 0].
    # Upstream main.py does not explicitly set lbx_0/ubx_0 before the first solve — we
    # follow the same pattern so acados' internal ``idxbx_0 / x0`` plumbing is the
    # authority at step 0.
    x0_init = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s0 = x0_init[0]
    statuses_up: list[int] = []
    statuses_lc: list[int] = []

    for i in range(NSIM):
        sref = s0 + SREF_N
        for j in range(N_HORIZON):
            yref = np.array([s0 + (sref - s0) * j / N_HORIZON, 0, 0, 0, 0, 0, 0, 0])
            up.set(j, "yref", yref)
            lc.set(j, "yref", yref)
        yref_e = np.array([sref, 0, 0, 0, 0, 0])
        up.set(N_HORIZON, "yref", yref_e)
        lc.set(N_HORIZON, "yref", yref_e)

        # Upstream's SQP hits MAXITER (status=2) on some early steps; the returned iterate
        # is still usable (upstream main.py just prints and continues). Parity is asserted
        # on the x/u values, not on convergence — but we require both solvers to agree on
        # whether the step converged (any divergence in iteration count would itself be
        # evidence of OCP mismatch).
        statuses_up.append(up.solve())
        statuses_lc.append(lc.solve())

        simX_up[i] = up.get(0, "x")
        simU_up[i] = up.get(0, "u")
        simX_lc[i] = lc.get(0, "x")
        simU_lc[i] = lc.get(0, "u")

        # Feed both solvers the same next x0 (upstream's x(1)) so any per-step divergence
        # does not compound through the closed loop — we measure only OCP-level parity.
        x1 = up.get(1, "x")
        for solver in (lc, up):
            solver.set(0, "lbx", x1)
            solver.set(0, "ubx", x1)
        s0 = x1[0]

    max_x = float(np.max(np.abs(simX_up - simX_lc)))
    max_u = float(np.max(np.abs(simU_up - simU_lc)))
    assert statuses_up == statuses_lc, (
        f"solver status histories differ: upstream={statuses_up}, leap-c={statuses_lc}"
    )
    assert max_x < TOL, f"max |Δx| = {max_x:g} exceeds tolerance {TOL:g}"
    assert max_u < TOL, f"max |Δu| = {max_u:g} exceeds tolerance {TOL:g}"
