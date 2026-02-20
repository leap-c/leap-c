import casadi as ca
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager


def make_default_msd_params(N_horizon: int = 100) -> tuple[AcadosParameter, ...]:
    """Return a tuple of default parameters for the mass-spring-damper planner.

    Args:
        N_horizon: The number of steps in the MPC horizon

    Returns:
        Tuple of AcadosParameter objects for the mass-spring-damper system.

    Note: The default parameters do not match the true parameter values used in the environment.
    """
    params = []
    params.extend(
        [
            AcadosParameter(
                name="q_diag_sqrt",
                default=np.sqrt(np.array([5.0, 0.2])),
                space=gym.spaces.Box(
                    low=np.sqrt(np.array([0.1, 0.01])),
                    high=np.sqrt(np.array([10.0, 1.0])),
                    dtype=np.float64,
                ),
                interface="learnable",
            ),
            AcadosParameter(
                name="r_diag_sqrt",
                default=np.sqrt(np.array([0.08])),
                space=gym.spaces.Box(
                    low=np.sqrt(np.array([0.001])),
                    high=np.sqrt(np.array([0.1])),
                    dtype=np.float64,
                ),
                interface="learnable",
            ),
            AcadosParameter(
                name="p_diag_sqrt",
                default=np.sqrt(np.array([5.0, 0.5])),
                space=gym.spaces.Box(
                    low=np.sqrt(np.array([1.0, 0.1])),
                    high=np.sqrt(np.array([10.0, 1.0])),
                    dtype=np.float64,
                ),
                interface="learnable",
            ),
            AcadosParameter(
                name="mass",
                default=np.array([1.5]),
                space=gym.spaces.Box(
                    low=np.array([0.1]),
                    high=np.array([10.0]),
                    dtype=np.float64,
                ),
                interface="learnable",
            ),
            AcadosParameter(
                name="damping",
                default=np.array([0.7]),
                space=gym.spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([2.0]),
                    dtype=np.float64,
                ),
                interface="learnable",
            ),
            AcadosParameter(
                name="stiffness",
                default=np.array([2.0]),
                space=gym.spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([5.0]),
                    dtype=np.float64,
                ),
                interface="learnable",
            ),
        ]
    )

    return tuple(params)


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    N_horizon: int,
    name: str = "mass_spring_damper",
    x0: np.ndarray | None = None,
) -> AcadosOcp:
    """Export the mass-spring-damper OCP.

    Args:
        param_manager: The parameter manager containing the parameters for the OCP.
        N_horizon: Number of time steps in the horizon.
        name: Name of the OCP model.
        x0: Initial state. If None, a default value is used.

    Returns:
        AcadosOcp: The configured OCP object.
    """
    ocp = AcadosOcp()

    param_manager.assign_to_ocp(ocp)

    # Model
    ocp.model.name = name

    ocp.model.x = ca.vertcat(
        ca.SX.sym("x"),
        ca.SX.sym("v"),
    )

    ocp.model.u = ca.SX.sym("F")

    dt: float = 0.1  # Time step in seconds

    # Get physical parameters
    mass = param_manager.get("mass")
    damping = param_manager.get("damping")
    stiffness = param_manager.get("stiffness")

    # Parametric dynamics matrices for mass-spring-damper system
    # Continuous dynamics: x_dot = v, v_dot = F/m - (b/m)*v - (k/m)*x
    # Discretized using Euler method
    A = ca.vertcat(
        ca.horzcat(1.0, dt),
        ca.horzcat(-dt * stiffness / mass, 1.0 - dt * damping / mass),
    )
    B = ca.vertcat(0.0, dt / mass)

    ocp.model.disc_dyn_expr = A @ ocp.model.x + B @ ocp.model.u

    # Cost function
    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    p_diag_sqrt = param_manager.get("p_diag_sqrt")

    # Construct stage cost weight matrix W from sqrt diagonal
    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    W = W_sqrt @ ca.transpose(W_sqrt)

    # Construct terminal cost weight matrix W_e from sqrt diagonal
    W_e_sqrt = ca.diag(p_diag_sqrt)
    W_e = W_e_sqrt @ ca.transpose(W_e_sqrt)

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = W
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.cost.yref = np.zeros((ocp.cost.W.shape[0],))

    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = W_e
    ocp.model.cost_y_expr_e = ocp.model.x
    ocp.cost.yref_e = np.zeros((ocp.cost.W_e.shape[0],))

    # Initial condition
    ocp.constraints.x0 = x0

    # State constraints
    ocp.constraints.lbx = np.array([-2.0, -2.0])
    ocp.constraints.ubx = np.array([+2.0, +2.0])
    ocp.constraints.idxbx = np.array([0, 1])  # x, v

    # Slack variables for state constraints
    ocp.constraints.idxsbx = np.array([0, 1])
    ocp.cost.Zu = ocp.cost.Zl = np.array([1e3, 1e3])
    ocp.cost.zu = ocp.cost.zl = np.array([0.0, 0.0])

    # Control constraints
    ocp.constraints.lbu = np.array([-0.5])
    ocp.constraints.ubu = np.array([+0.5])
    ocp.constraints.idxbu = np.array([0])  # F

    # Solver options
    ocp.solver_options.tf = N_horizon * dt
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp


if __name__ == "__main__":
    N_horizon = 100
    param_manager = AcadosParameterManager(
        make_default_msd_params(N_horizon=N_horizon),
        N_horizon=N_horizon,
    )

    ocp = export_parametric_ocp(
        param_manager=param_manager,
        N_horizon=N_horizon,
        x0=np.array([1.5, -1.0]),
    )

    ocp.translate_initial_cost_term_to_external(cost_hessian=ocp.solver_options.hessian_approx)
    ocp.translate_intermediate_cost_term_to_external(cost_hessian=ocp.solver_options.hessian_approx)
    ocp.translate_terminal_cost_term_to_external(cost_hessian=ocp.solver_options.hessian_approx)

    ocp_solver = AcadosOcpSolver(ocp)

    status = ocp_solver.solve()

    X = np.array([ocp_solver.get(i, "x") for i in range(N_horizon + 1)])
    U = np.array([ocp_solver.get(i, "u") for i in range(N_horizon)])

    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(X[:, 0], label="Position x1")
    plt.ylabel("Position x1")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(X[:, 1], label="Velocity x2", color="orange")
    plt.ylabel("Velocity x2")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.step(np.arange(N_horizon), U[:, 0], where="post", label="Control u", color="green")
    plt.ylabel("Control u")
    plt.xlabel("Time step")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(status == 0 and "SUCCESS" or "FAILURE")
