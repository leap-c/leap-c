import numpy as np
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.linear_mpc import LinearMPC
from test_mpc_p_global import (
    run_test_mpc_solve_and_batch_solve_on_batch_p_global,
)
from conftest import generate_batch_variation


# def test_parametric_sensitivities(
#     learnable_point_mass_mpc: LinearMPC, point_mass_mpc_p_global: np.ndarray
# ):
#     run_test_mpc_solve_and_batch_solve_on_batch_p_global(
#         learnable_point_mass_mpc, point_mass_mpc_p_global, plot=False
#     )


def main():
    if False:
        mpc = PointMassMPC(learnable_params=["m", "c", "q_diag"])

        s = np.array([1.0, 0.0, 0.0, 0.0])
        a = np.array([0.0, 0.0])

        policy = mpc.policy(state=s, p_global=None)[0]
        print(policy)

        state_value = mpc.state_value(state=s, p_global=None)[0]
        print(state_value)

        state_action_value = mpc.state_action_value(state=s, action=a, p_global=None)[0]
        print(state_action_value)

    # env = PointMassOcpEnv(mpc)

    # (x, p), _ = env.reset(seed=0)

    # x0 = np.array([1.0, 1.0, 0.0, 0.0])

    # _ = env.step(mpc.policy(state=x0, p_global=None)[0])

    # n_batch = 4

    # learnable_point_mass_mpc = PointMassMPC(
    #     learnable_params=["m", "c"], n_batch=n_batch
    # )

    # p_global = generate_batch_variation(
    #     learnable_point_mass_mpc.ocp_solver.acados_ocp.p_global_values, n_batch
    # )

    # test_parametric_sensitivities(learnable_point_mass_mpc, p_global)


if __name__ == "__main__":
    main()
