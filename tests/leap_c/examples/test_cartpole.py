import os
import shutil
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from acados_template import AcadosOcpSolver
from gymnasium.utils.save_video import save_video

from leap_c.examples.cartpole.controller import CartPoleController
from leap_c.examples.cartpole.env import CartPoleEnv


@pytest.fixture(scope="module", params=["EXTERNAL", "NONLINEAR_LS"])
def cartpole_controller(request):
    return CartPoleController(cost_type=request.param)


def plot_cart_pole_solution(
    ocp_solver: AcadosOcpSolver,
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    k = np.arange(0, ocp_solver.N + 1)
    u = np.array([ocp_solver.get(stage, "u") for stage in range(ocp_solver.N)])
    x = np.array([ocp_solver.get(stage, "x") for stage in range(ocp_solver.N + 1)])

    fig, axs = plt.subplots(5, 1)
    labels = ["x", "theta", "dx", "dtheta"]

    for i in range(4):
        axs[i].step(k, x[:, i])
        axs[i].set_ylabel(labels[i])
        axs[i].grid()

    axs[4].step(k[:-1], u)
    axs[4].set_ylabel("F")
    axs[4].set_xlabel("k")
    axs[4].grid()

    return fig, axs


def test_solution(cartpole_controller: CartPoleController):
    ocp_solver = (
        cartpole_controller.diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[0]
    )
    ocp_solver.solve_for_x0(np.array([0.0, np.pi, 0.0, 0.0]))

    if ocp_solver.status != 0:
        raise ValueError(f"Solver failed with status {ocp_solver.status}")

    fig, axs = plot_cart_pole_solution(ocp_solver)


def test_env_terminates():
    """Test if the environment terminates correctly when applying minimum and maximum control inputs.

    This test ensures that the environment terminates properly when applying either minimum or maximum control
    inputs continuously. It checks both termination conditions and verifies that the episode ends with a termination
    rather than a truncation.
    """

    env = CartPoleEnv()

    for action in [env.action_space.low, env.action_space.high]:  # type:ignore
        env.reset(seed=0)
        for _ in range(1000):
            state, _, term, trunc, _ = env.step(action)
            if term:
                break
        assert term
        assert not trunc
        assert state[0] < -env.x_threshold or state[0] > env.x_threshold


def test_env_truncates():
    """Test if the environment truncates correctly when applying minimum and maximum control inputs.

    This test ensures that the environment truncates properly when doing nothing (i.e. it cannot come from termination).
    It checks both termination conditions and verifies that the episode ends with a truncation
    rather than a truncation.
    """

    env = CartPoleEnv()
    env.reset(seed=0)

    action = np.array([0])
    for _ in range(1000):
        _, _, term, trunc, _ = env.step(action)
        if trunc:
            break
    assert not term
    assert trunc


def test_env_types():
    """Test whether the type of the state is and stays np.float32
    for an action from the action space (note that the action space has type np.float32).
    """

    env = CartPoleEnv()

    x, _ = env.reset(seed=0)
    assert x.dtype == np.float32
    action = np.zeros(env.action_space.shape, dtype=np.float32)  # type:ignore
    x, _, _, _, _ = env.step(action)
    assert x.dtype == np.float32


def test_closed_loop_rendering(
    cartpole_controller: CartPoleController,
):
    env = CartPoleEnv()

    obs, _ = env.reset(seed=1337)

    count = 0
    terminated = False
    truncated = False
    frames = []
    cwd = os.getcwd()
    savefile_dir_path = os.path.join(cwd, "test_closed_loop_pendulum_on_cart")

    default_param = cartpole_controller.default_param(obs)
    default_param = torch.as_tensor(default_param, dtype=torch.float32).unsqueeze(0)

    ctx = None

    if not os.path.exists(savefile_dir_path):
        os.mkdir(savefile_dir_path)
    while count < 300 and not terminated and not truncated:
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        ctx, a = cartpole_controller(obs, default_param, ctx=ctx)
        a = a.squeeze(0).numpy()
        obs_prime, r, terminated, truncated, info = env.step(a)
        frames.append(env.render())
        obs = obs_prime
        count += 1
    assert count <= 200, (
        "max_time and dt dictate that no more than 200 steps should be possible until termination."
    )
    save_video(
        frames,  # type:ignore
        video_folder=savefile_dir_path,
        name_prefix="pendulum_on_cart",
        fps=env.metadata["render_fps"],
    )

    shutil.rmtree(savefile_dir_path)
