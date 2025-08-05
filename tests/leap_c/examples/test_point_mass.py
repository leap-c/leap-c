import numpy as np
import torch

from leap_c.ocp.acados.parameters import Parameter
from leap_c.examples.pointmass.controller import PointMassController
from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.examples.pointmass.config import make_default_pointmass_params


def test_run_closed_loop(
    n_iter: int = 200,
) -> None:
    """
    Test the closed-loop performance of a learnable point mass MPC.


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
    param = make_default_pointmass_params()
    old_xref_param = param.xref
    kw = {**old_xref_param._asdict(), "value": goal_x_ref}
    param.xref = Parameter(**kw)
    controller = PointMassController(params=param)

    default_param = controller.default_param(obs)
    default_param = torch.as_tensor(default_param, dtype=torch.float32).unsqueeze(0)
    ctx = None

    for _ in range(n_iter - 1):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        ctx, a = controller(obs, default_param, ctx=ctx)
        a = a.squeeze(0).numpy()
        obs, r, terminated, truncated, info = env.step(a)

        if terminated or truncated:
            break

    assert np.linalg.norm(obs[:2] - goal_pos) < 0.2, (
        "Final position is not close to the goal"
    )  # Check that the final position is close to the goal
    assert (
        np.linalg.norm(obs[2:4]) < 0.1
    )  # Check that the final velocity is close to zero
