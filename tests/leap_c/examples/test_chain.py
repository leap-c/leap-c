import numpy as np
import pytest
import torch

import leap_c.examples  # noqa: F401
import leap_c.torch.rl  # noqa: F401
from leap_c.examples.chain.env import ChainEnv
from leap_c.examples.chain.controller import ChainController


@pytest.fixture(scope="module")
def chain_controller():
    return ChainController(n_mass=3)


def test_chain_policy_evaluation_works(chain_controller: ChainController):
    x0 = chain_controller.diff_mpc.ocp.constraints.x0

    # Move the second mass a bit in x direction
    x0[3] += 0.1

    obs = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
    default_param = chain_controller.default_param(obs)
    default_param = torch.as_tensor(default_param, dtype=torch.float32).unsqueeze(0)

    ctx, _ = chain_controller(obs, default_param)

    assert ctx.status[0] == 0, "Policy evaluation failed"


def test_chain_env_mpc_closed_loop(chain_controller: ChainController):
    env = ChainEnv(n_mass=3)

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
