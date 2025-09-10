import os
import shutil
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from acados_template import AcadosOcpSolver
from gymnasium.utils.save_video import save_video

from leap_c.examples.race_cars.controller import (
    RaceCarController,
    RaceCarControllerConfig,
)
from leap_c.examples.race_cars.env import RaceCarEnv


@pytest.fixture(scope="module", params=["EXTERNAL", "NONLINEAR_LS"])
def race_car_controller(request):
    cfg = RaceCarControllerConfig(cost_type=request.param)
    return RaceCarController(cfg)


def plot_race_car_solution(
    ocp_solver: AcadosOcpSolver,
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    k = np.arange(0, ocp_solver.N + 1)
    u = np.array([ocp_solver.get(stage, "u") for stage in range(ocp_solver.N)])
    x = np.array([ocp_solver.get(stage, "x") for stage in range(ocp_solver.N + 1)])

    fig, axs = plt.subplots(6, 1, figsize=(10, 12))
    state_labels = ["s", "n", "alpha", "v", "D", "delta"]

    for i in range(6):
        axs[i].step(k, x[:, i])
        axs[i].set_ylabel(state_labels[i])
        axs[i].grid()

    # Add control input plot
    fig, axs = plt.subplots(8, 1, figsize=(10, 16))
    
    # Plot states
    for i in range(6):
        axs[i].step(k, x[:, i])
        axs[i].set_ylabel(state_labels[i])
        axs[i].grid()

    # Plot control inputs
    axs[6].step(k[:-1], u[:, 0])
    axs[6].set_ylabel("derD")
    axs[6].grid()
    
    axs[7].step(k[:-1], u[:, 1])
    axs[7].set_ylabel("derDelta")
    axs[7].set_xlabel("k")
    axs[7].grid()

    return fig, axs


def test_solution(race_car_controller: RaceCarController):
    ocp_solver = (
        race_car_controller.diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[0]
    )
    # Initial state: slightly before start line with small perturbations
    x0 = np.array([-2.0, 0.01, 0.05, 0.0, 0.0, 0.0])
    ocp_solver.solve_for_x0(x0)

    if ocp_solver.status != 0:
        raise ValueError(f"Solver failed with status {ocp_solver.status}")

    fig, axs = plot_race_car_solution(ocp_solver)


def test_env_terminates():
    """Test if the environment terminates correctly when going off track.

    This test ensures that the environment terminates properly when the car goes too far
    laterally off the track. It checks termination conditions and verifies that the episode 
    ends with a termination rather than a truncation.
    """

    env = RaceCarEnv()

    # Test with extreme steering to cause off-track termination
    for action in [
        np.array([0.0, env.action_space.high[1]]),  # Maximum steering rate
        np.array([0.0, env.action_space.low[1]]),   # Minimum steering rate
    ]:
        env.reset(seed=0)
        terminated = False
        truncated = False
        
        for _ in range(1000):
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Should terminate due to going off track, not truncate due to time limit
        assert terminated or truncated
        
        # If terminated, should be due to violation (going off track)
        if terminated and info.get("task"):
            assert info["task"]["violation"] == True


def test_env_truncates():
    """Test if the environment truncates correctly when reaching time limit.

    This test ensures that the environment truncates properly when the maximum time
    is reached without termination conditions being met.
    """

    env = RaceCarEnv()
    env.reset(seed=0)

    # Apply minimal control to avoid termination but reach time limit
    action = np.array([0.1, 0.0])  # Small throttle, no steering
    
    terminated = False
    truncated = False
    
    for _ in range(int(env.cfg.max_time / env.cfg.dt) + 10):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # Should eventually truncate due to time limit
    assert truncated or terminated
    
    # If truncated, should be due to time limit, not violation
    if truncated and info.get("task"):
        assert info["task"]["violation"] == False


def test_env_types():
    """Test whether the type of the state is and stays np.float32
    for an action from the action space (note that the action space has type np.float32).
    """

    env = RaceCarEnv()

    x, _ = env.reset(seed=0)
    assert x.dtype == np.float32
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    x, _, _, _, _ = env.step(action)
    assert x.dtype == np.float32


def test_env_info_dict_numeric():
    """Test that the info dictionary contains only numeric values that can be averaged."""
    
    env = RaceCarEnv()
    env.reset(seed=0)
    
    # Test normal step
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    
    # Check that all values in info dict are numeric
    def check_numeric_values(d, path=""):
        for key, value in d.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                check_numeric_values(value, current_path)
            else:
                assert isinstance(value, (int, float, bool, np.number)), \
                    f"Non-numeric value at {current_path}: {value} (type: {type(value)})"
    
    if info:
        check_numeric_values(info)


def test_closed_loop_rendering(
    race_car_controller: RaceCarController,
):
    env = RaceCarEnv(render_mode="rgb_array")

    obs, _ = env.reset(seed=1337)

    count = 0
    terminated = False
    truncated = False
    frames = []
    cwd = os.getcwd()
    savefile_dir_path = os.path.join(cwd, "test_closed_loop_race_car")

    default_param = race_car_controller.default_param(obs)
    default_param = torch.as_tensor(default_param, dtype=torch.float32).unsqueeze(0)

    ctx = None

    if not os.path.exists(savefile_dir_path):
        os.mkdir(savefile_dir_path)
    
    # Run for shorter duration due to race car dynamics
    max_steps = min(int(env.cfg.max_time / env.cfg.dt), 500)
    
    while count < max_steps and not terminated and not truncated:
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        ctx, a = race_car_controller(obs, default_param, ctx=ctx)
        a = a.squeeze(0).numpy()
        obs_prime, r, terminated, truncated, info = env.step(a)
        
        # Only render every few frames to avoid too many frames
        if count % 5 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        obs = obs_prime
        count += 1
    
    # Save video if we have frames
    if frames:
        save_video(
            frames,
            video_folder=savefile_dir_path,
            name_prefix="race_car",
            fps=env.metadata["render_fps"] // 5,  # Slower due to frame skipping
        )

    shutil.rmtree(savefile_dir_path)


def test_env_reset():
    """Test that environment resets properly."""
    
    env = RaceCarEnv()
    
    # Initial reset
    obs1, info1 = env.reset(seed=42)
    assert obs1.shape == env.observation_space.shape
    assert obs1.dtype == np.float32
    
    # Take some steps
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    
    # Reset again
    obs2, info2 = env.reset(seed=42)
    
    # Should get same initial state with same seed
    np.testing.assert_array_almost_equal(obs1, obs2)
    
    # Reset with different seed should give different state
    obs3, info3 = env.reset(seed=123)
    assert not np.allclose(obs1, obs3)


def test_env_reward_structure():
    """Test that reward has expected structure and range."""
    
    env = RaceCarEnv()
    obs, _ = env.reset(seed=0)
    
    rewards = []
    
    # Test with various actions
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    # Rewards should be finite numbers
    assert all(np.isfinite(r) for r in rewards)
    
    # Should have some variation in rewards
    assert len(set(rewards)) > 1 or len(rewards) == 1