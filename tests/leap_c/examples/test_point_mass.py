import numpy as np
import torch

from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.examples.pointmass.planner import PointMassControllerConfig, PointMassPlanner
from leap_c.planner import ControllerFromPlanner


def test_run_closed_loop(n_iter: int = 200) -> None:
    """Test the closed-loop performance of a differentiable point mass MPC.

    Asserts:
    - The final position of the point mass is close to the origin.
    - The final velocity of the point mass is close to zero.

    """
    env = PointMassEnv()
    obs, _ = env.reset()

    # overwrite initial state closely over the goal
    goal_pos = env.goal.pos
    goal_x_ref = np.array([goal_pos[0], goal_pos[1], 0.0, 0.0])
    start_pos = goal_pos + np.array([0.0, 0.7])
    env.state[:2] = start_pos

    # replace the default reference with the goal position
    cfg = PointMassControllerConfig(x_ref_value=goal_x_ref)
    planner = PointMassPlanner(cfg=cfg)
    controller = ControllerFromPlanner(planner=planner)

    ctx = None

    for _ in range(n_iter - 1):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        ctx, a = controller(obs, ctx=ctx)
        a = a.squeeze(0).numpy()
        obs, r, terminated, truncated, info = env.step(a)

        if terminated or truncated:
            break

    assert np.linalg.norm(obs[:2] - goal_pos) < 0.2, (
        "Final position is not close to the goal"
    )  # Check that the final position is close to the goal
    assert np.linalg.norm(obs[2:4]) < 0.1  # Check that the final velocity is close to zero
