import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from leap_c.examples.cartpole.task import CartPoleSwingup
from leap_c.examples.cartpole.controller import CartPoleController
from leap_c.ocp.acados.torch import AcadosDiffMpc

def plot_solution(params):
    # Prepare figure: 5 rows (states+force), 2 columns per param value
    n_params = len(params)
    fig, axs = plt.subplots(
        5, n_params * 2, figsize=(5 * n_params * 2, 10),
        sharex=True, sharey='row'
    )
    labels = ["x", "theta", "dx", "dtheta", "F"]

    fig.suptitle("CartPole Trajectories for Different Params", fontsize=18)

    # Create solvers outside the loop
    task = CartPoleSwingup()
    ocp_solver = task.mpc.mpc.ocp_solver
    controller = CartPoleController()
    diff_mpc = controller.diff_mpc
    diff_mpc.ocp.p_global_values = np.array([params[0]]).astype(np.float64)

    for idx, param in enumerate(params):
        # get old trajectories
        ocp_solver.set_p_global_and_precompute_dependencies(
            np.array([param]).astype(np.float64)
        )
        ocp_solver.solve_for_x0(np.array([0.0, np.pi, 0.0, 0.0]))

        if ocp_solver.status != 0:
            raise RuntimeError(f"Solver failed with status {ocp_solver.status}")

        u_old = np.array([ocp_solver.get(stage, "u") for stage in range(ocp_solver.N)])
        x_old = np.array([ocp_solver.get(stage, "x") for stage in range(ocp_solver.N + 1)])

        # get new trajectories
        state = np.array([[0.0, np.pi, 0.0, 0.0]])
        param_arr = np.array([[param]])

        ctx, _, x_new, u_new, _ = diff_mpc(x0=state, p_global=param_arr, ctx=None)
        x_new = x_new.detach().cpu().numpy().reshape(-1, 4)
        u_new = u_new.detach().cpu().numpy().reshape(-1, 1)

        # Plot for this param
        col_old = idx * 2
        col_new = idx * 2 + 1

        # Add column labels
        axs[0, col_old].set_title(f"Old\nparam={param:.2f}")
        axs[0, col_new].set_title(f"New\nparam={param:.2f}")

        for i in range(4):
            # Old (left)
            axs[i, col_old].step(np.arange(x_old.shape[0]), x_old[:, i])
            if i == 0 and idx == 0:
                axs[i, col_old].set_ylabel(labels[i], fontsize=12)
            axs[i, col_old].grid()
            # New (right)
            axs[i, col_new].step(np.arange(x_new.shape[0]), x_new[:, i], color="tab:orange")
            # axs[i, col_new].set_ylabel(labels[i])
            axs[i, col_new].grid()

        # Add force plots as the 5th row
        axs[4, col_old].step(np.arange(u_old.shape[0]), u_old)
        if idx == 0:
            axs[4, col_old].set_ylabel(labels[4], fontsize=12)
        axs[4, col_old].set_xlabel("k")
        axs[4, col_old].grid()

        axs[4, col_new].step(np.arange(u_new.shape[0]), u_new, color="tab:orange")
        # axs[4, col_new].set_ylabel(labels[4])
        axs[4, col_new].set_xlabel("k")
        axs[4, col_new].grid()

    # print all p globals
    print("Global parameters in DiffMpc:")
    print(diff_mpc.ocp.model.p_global)
    print(diff_mpc.ocp.p_global_values)
    print("Stagewise parameters in DiffMpc:")
    print(diff_mpc.ocp.model.p)
    print(diff_mpc.ocp.parameter_values)
    print("yref in DiffMpc:")
    print(diff_mpc.ocp.model.cost_y_expr)
    print(diff_mpc.ocp.cost.yref)

    print("Global parameters in OCP Solver:")
    print(ocp_solver.acados_ocp.model.p_global)
    print(ocp_solver.acados_ocp.p_global_values)
    print("Stagewise parameters in OCP Solver:")
    print(ocp_solver.acados_ocp.model.p)
    print(ocp_solver.acados_ocp.parameter_values)
    print("yref in OCP Solver:")
    print(ocp_solver.acados_ocp.model.cost_y_expr)
    print(ocp_solver.acados_ocp.cost.yref)


    plt.tight_layout(rect=[0.05, 0, 1, 1])  # leave space on the left for y-labels
    plt.show()

if __name__ == "__main__":
    plot_solution(params=[-np.pi/2, 0, np.pi/2])
