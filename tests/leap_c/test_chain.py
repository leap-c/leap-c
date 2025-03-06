from leap_c.examples.chain.mpc import ChainMpc

if __name__ == "__main__":
    learnable_params = ["m", "D", "L", "C", "w"]
    mpc = ChainMpc(learnable_params=learnable_params, n_mass=3)

    ocp_solver = mpc.ocp_solver

    x_init = [ocp_solver.get(stage, "x") for stage in range(ocp_solver.N + 1)]
    u_init = [ocp_solver.get(stage, "u") for stage in range(ocp_solver.N)]

    x0 = ocp_solver.acados_ocp.constraints.x0

    # Move the second mass a bit in x direction
    x0[3] += 0.1

    u0, du0_dp_global, status = mpc.policy(state=x0, sens=True, p_global=None)

    print("u0: ", u0)
    print("du0_dp_global: ", du0_dp_global)
    print("status: ", status)
    print("sqp_iter", mpc.ocp_solver.get_stats("sqp_iter"))
