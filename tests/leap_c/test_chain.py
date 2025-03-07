from leap_c.examples.chain.mpc import ChainMpc


def test_chain_policy_evaluation_works():
    learnable_params = ["m", "D", "L", "C", "w"]
    mpc = ChainMpc(learnable_params=learnable_params, n_mass=3)

    x0 = mpc.ocp_solver.acados_ocp.constraints.x0

    # Move the second mass a bit in x direction
    x0[3] += 0.1

    u0, du0_dp_global, status = mpc.policy(state=x0, sens=True, p_global=None)

    assert status == 0, "Policy evaluation failed"


if __name__ == "__main__":
    test_chain_policy_evaluation_works()
