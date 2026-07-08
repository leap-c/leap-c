"""The fully parametric mass-spring-damper OCP (six differentiable parameters).

Sole home of this builder — used by ``custom_examples/advanced_sensitivities.py``.
The getting-started intro notebook teaches a smaller single-parameter MSD
variant inline instead.
"""

import casadi as ca
import numpy as np
from acados_template import AcadosOcp

from leap_c.parameters import AcadosParameterManager


def build_msd_ocp(N_horizon: int, dt: float) -> tuple[AcadosOcp, AcadosParameterManager]:
    """Build the parametric mass-spring-damper OCP and its parameter manager.

    Always builds the OCP and the manager together, fresh: a manager is
    finalized by ``AcadosDiffMpcTorch`` (via ``assign_to_ocp``) and must not
    be reused for a second OCP.
    """
    manager = AcadosParameterManager(N_horizon=N_horizon)

    def register(name, default):
        # differentiable=True -> part of p_global, gradients flow.
        return manager.register_parameter(name=name, default=default, differentiable=True)

    # Cost parameters
    q_diag_sqrt = register("q_diag_sqrt", np.sqrt(np.array([5.0, 0.2])))
    r_diag_sqrt = register("r_diag_sqrt", np.sqrt(np.array([0.08])))
    p_diag_sqrt = register("p_diag_sqrt", np.sqrt(np.array([5.0, 0.5])))

    # Model parameters
    mass = register("mass", np.array([1.5]))
    damping = register("damping", np.array([0.7]))
    stiffness = register("stiffness", np.array([2.0]))

    ocp = AcadosOcp()
    ocp.model.name = "mass_spring_damper"

    # State [position, velocity] and control [force].
    ocp.model.x = ca.vertcat(ca.SX.sym("p"), ca.SX.sym("v"))
    ocp.model.u = ca.SX.sym("F")

    # Parametric discrete-time dynamics x_{t+1} = A x + B F.
    A = ca.vertcat(
        ca.horzcat(1.0, dt),
        ca.horzcat(-dt * stiffness / mass, 1.0 - dt * damping / mass),
    )
    B = ca.vertcat(0.0, dt / mass)
    ocp.model.disc_dyn_expr = A @ ocp.model.x + B @ ocp.model.u

    # Quadratic regulator cost, weights built from the sqrt-diagonal parameters.
    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = W_sqrt @ W_sqrt.T
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.cost.yref = np.zeros((3,))

    W_e_sqrt = ca.diag(p_diag_sqrt)
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = W_e_sqrt @ W_e_sqrt.T
    ocp.model.cost_y_expr_e = ocp.model.x
    ocp.cost.yref_e = np.zeros((2,))

    # Initial state — a nominal value, overwritten on every solve.
    ocp.constraints.x0 = np.array([1.0, 0.0])

    # Soft box constraints on the state.
    ocp.constraints.idxbx = np.array([0, 1])
    ocp.constraints.lbx = np.array([-2.0, -2.0])
    ocp.constraints.ubx = np.array([2.0, 2.0])
    ocp.constraints.idxsbx = np.array([0, 1])
    ocp.cost.Zl = ocp.cost.Zu = np.array([1e3, 1e3])
    ocp.cost.zl = ocp.cost.zu = np.array([0.0, 0.0])

    # Hard box constraint on the force.
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([-0.5])
    ocp.constraints.ubu = np.array([0.5])

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon * dt
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    # NOTE: we never set ocp.model.p / ocp.model.p_global ourselves — the
    # AcadosDiffMpcTorch constructor calls manager.assign_to_ocp(ocp) for us
    # and sets them from the registered parameters.
    return ocp, manager
