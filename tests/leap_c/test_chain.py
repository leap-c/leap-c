from leap_c.examples.chain.mpc import ChainMpc

if __name__ == "__main__":
    learnable_params = ["m", "D", "L", "C", "w"]
    chain_mpc = ChainMpc(learnable_params=learnable_params)

    ocp_solver = chain_mpc.ocp_solver
