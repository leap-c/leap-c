"""Smoke tests for the MPCC race-car planner.

Covers (per frame):
- Building the parametric OCP via ``export_mpcc_ocp``.
- Instantiating ``MpccPlanner`` and solving for one observation.
- Driving ``RaceCarEnv`` closed-loop for 20 steps with finite controls.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from leap_c.examples.race_car.env import RaceCarEnv, RaceCarEnvConfig
from leap_c.examples.race_car.mpcc_acados_ocp import create_mpcc_params, export_mpcc_ocp
from leap_c.examples.race_car.mpcc_model import NU_MPCC, NX_MPCC
from leap_c.examples.race_car.mpcc_planner import MpccPlanner, MpccPlannerConfig
from leap_c.ocp.acados.parameters import AcadosParameterManager


@pytest.mark.parametrize("frame", ["cartesian", "frenet"])
def test_export_mpcc_ocp_builds(frame: str) -> None:
    N = 50
    pm = AcadosParameterManager(
        parameters=create_mpcc_params(param_interface="global", N_horizon=N),
        N_horizon=N,
    )
    ocp = export_mpcc_ocp(pm, frame=frame, N_horizon=N, T_horizon=1.0)
    assert ocp.dims.nx == NX_MPCC
    assert ocp.dims.nu == NU_MPCC
    assert ocp.cost.cost_type == "NONLINEAR_LS"
    assert ocp.cost.cost_type_e == "NONLINEAR_LS"
    assert ocp.model.disc_dyn_expr.shape[0] == NX_MPCC
    assert ocp.model.con_h_expr.shape[0] == 5  # [a_long, a_lat, e_c, D, delta]


@pytest.mark.parametrize("frame", ["cartesian", "frenet"])
def test_planner_forward_single_step(frame: str) -> None:
    planner = MpccPlanner(cfg=MpccPlannerConfig(frame=frame))
    env = RaceCarEnv(cfg=RaceCarEnvConfig())
    obs_np, _ = env.reset(seed=0)
    obs_t = torch.tensor(obs_np, dtype=torch.float64).unsqueeze(0)

    with torch.no_grad():
        ctx, u0, x_plan, u_plan, value = planner(obs_t)

    assert u_plan is not None
    action = u_plan[0, 0, :].cpu().numpy()
    assert action.shape == (2,), f"expected env-shape action, got {action.shape}"
    assert np.all(np.isfinite(action)), "action contains non-finite values"
    assert env.action_space.contains(action.astype(np.float64))


@pytest.mark.parametrize("frame", ["cartesian", "frenet"])
def test_closed_loop_short(frame: str) -> None:
    planner = MpccPlanner(cfg=MpccPlannerConfig(frame=frame))
    env = RaceCarEnv(cfg=RaceCarEnvConfig())
    obs_np, _ = env.reset(seed=0)

    ctx = None
    for _ in range(20):
        obs_t = torch.tensor(obs_np, dtype=torch.float64).unsqueeze(0)
        with torch.no_grad():
            ctx, _u0, _x_plan, u_plan, _value = planner(obs_t, ctx=ctx)
        action = u_plan[0, 0, :].cpu().numpy().astype(np.float64)
        assert np.all(np.isfinite(action))
        obs_np, _r, term, trunc, _info = env.step(action)
        if term or trunc:
            break

    assert np.all(np.isfinite(obs_np))
