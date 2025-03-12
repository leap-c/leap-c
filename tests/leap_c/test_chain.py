import matplotlib.pyplot as plt
import numpy as np

import leap_c.examples
import leap_c.rl  # noqa: F401
from leap_c.examples.chain.env import ChainEnv
from leap_c.examples.chain.mpc import ChainMpc
from leap_c.examples.chain.utils import (
    animate_chain_position_3D,
    Ellipsoid,
)
from leap_c.registry import create_default_cfg, create_task, create_trainer


def test_chain_policy_evaluation_works():
    learnable_params = ["m", "D", "L", "C", "w"]
    mpc = ChainMpc(learnable_params=learnable_params, n_mass=4)

    x0 = mpc.ocp_solver.acados_ocp.constraints.x0

    # Move the second mass a bit in x direction
    x0[3] += 0.1

    u0, du0_dp_global, status = mpc.policy(state=x0, sens=False, p_global=None)

    assert status == 0, "Policy evaluation failed"


def test_chain_env_mpc_closed_loop(plot: bool = False, animate: bool = False):
    learnable_params = ["m", "D", "L", "C", "w"]
    n_mass = 3

    params = {}

    # rest length of spring
    params["L"] = np.repeat([0.033, 0.033, 0.033], n_mass - 1)

    # spring constant
    params["D"] = np.repeat([1.0, 1.0, 1.0], n_mass - 1)

    # damping constant
    params["C"] = np.repeat([0.1, 0.1, 0.1], n_mass - 1)

    # mass of the balls
    params["m"] = np.repeat([0.033], n_mass - 1)

    # disturbance on intermediate balls
    params["w"] = np.repeat([0.0, 0.0, 0.0], n_mass - 2)

    # Weight on state
    params["q_sqrt_diag"] = np.ones(3 * (n_mass - 1) + 3 * (n_mass - 2))

    # Weight on control inputs
    params["r_sqrt_diag"] = 1e-1 * np.ones(3)

    fix_point = np.zeros(3)
    pos_last_mass_ref = fix_point + 10 * np.array([0.033 * (n_mass - 1), 0, 0])

    ellipsoid = Ellipsoid(center=fix_point, radii=np.array(params["L"]).reshape(-1, 3).sum(axis=0))
    # ellipsoid.plot_surface()

    phi_start = 0.0
    theta_start = np.pi / 2
    phi_end = np.pi
    theta_end = np.pi / 2

    xstart = ellipsoid.spherical_to_cartesian(phi=phi_start, theta=theta_start)
    xend = ellipsoid.spherical_to_cartesian(phi=phi_end, theta=theta_end)

    # points = np.vstack([xstart, xend])
    # ellipsoid.plot_points(points)
    # plt.show()

    pos_last_mass_ref = ellipsoid.spherical_to_cartesian(phi=phi_end, theta=theta_end)

    env = ChainEnv(n_mass=n_mass, fix_point=fix_point, pos_last_ref=pos_last_mass_ref, param=params)
    env.reset()

    mpc = ChainMpc(learnable_params=learnable_params, n_mass=n_mass, pos_last_mass_ref=pos_last_mass_ref)

    x_ref = env.x_ref
    u_ref = env.u_ref

    sim_x = [env.state]
    sim_u = []

    for _ in range(100):
        u0, _, status = mpc.policy(state=sim_x[-1], sens=False, p_global=None)

        sim_u.append(u0)
        o, r, term, trunc, info = env.step(u0)
        sim_x.append(o)

    sim_x = np.array(sim_x)
    sim_u = np.array(sim_u)

    error_norm = np.linalg.norm(x_ref - sim_x, axis=1)

    # assert error_norm[-1] < 1e-2, "Error norm is too high"

    if plot:
        plt.figure()
        plt.plot(error_norm, label="x_ref - x")

    # if animate:
    #     animate_chain_position_3D(sim_x, fix_point)

    plt.show()


if __name__ == "__main__":
    # test_chain_policy_evaluation_works()
    test_chain_env_mpc_closed_loop(animate=True, plot=True)

    # trainer = create_trainer(
    #     name="sac_fop",
    #     task=create_task("chain"),
    #     output_path="output/videos",
    #     device="cpu",
    #     cfg=create_default_cfg("sac_fop"),
    # )

    # trainer.validate()
