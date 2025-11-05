import numpy as np
import pytest
import torch

from leap_c.examples.chain.env import ChainEnv, ChainEnvConfig
from leap_c.examples.chain.planner import ChainControllerConfig, ChainPlanner
from leap_c.planner import ControllerFromPlanner


@pytest.fixture(scope="module")
def chain_controller():
    cfg = ChainControllerConfig(n_mass=3)
    planner = ChainPlanner(cfg)
    return ControllerFromPlanner(planner)


def test_chain_policy_evaluation_works(chain_controller):
    x0 = chain_controller.planner.diff_mpc.diff_mpc_fun.ocp.constraints.x0

    # Move the second mass a bit in x direction
    x0[3] += 0.1

    obs = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
    default_param = chain_controller.default_param(obs)
    default_param = torch.as_tensor(default_param, dtype=torch.float32).unsqueeze(0)

    ctx, _ = chain_controller(obs, default_param)

    assert ctx.status[0] == 0, "Policy evaluation failed"


def test_chain_env_mpc_closed_loop(chain_controller):
    cfg = ChainEnvConfig(n_mass=3)
    env = ChainEnv(cfg=cfg)

    obs, _ = env.reset()

    x_ref = env.x_ref

    default_param = chain_controller.default_param(obs)
    default_param = torch.as_tensor(default_param, dtype=torch.float32).unsqueeze(0)

    ctx = None

    for _ in range(100):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        ctx, a = chain_controller(obs, default_param, ctx=ctx)
        a = a.squeeze(0).numpy()
        obs, *_ = env.step(a)

    error_norm = np.linalg.norm(x_ref - obs)

    assert error_norm < 1e-2, "Error norm is too high"
