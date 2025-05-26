import casadi as ca

from acados_template import AcadosOcp


def set_standard_sensitivity_options(ocp_sensitivity: AcadosOcp):
    ocp_sensitivity.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp_sensitivity.solver_options.qp_solver_ric_alg = 1
    ocp_sensitivity.solver_options.qp_solver_cond_N = (
        ocp_sensitivity.solver_options.N_horizon
    )
    ocp_sensitivity.solver_options.hessian_approx = "EXACT"
    ocp_sensitivity.solver_options.exact_hess_dyn = True
    ocp_sensitivity.solver_options.exact_hess_cost = True
    ocp_sensitivity.solver_options.exact_hess_constr = True
    ocp_sensitivity.solver_options.with_solution_sens_wrt_params = True
    ocp_sensitivity.solver_options.with_value_sens_wrt_params = True
    ocp_sensitivity.solver_options.with_batch_functionality = True
    ocp_sensitivity.model.name += "_sensitivity"  # type:ignore


def SX_to_labels(SX: ca.SX) -> list[str]:
    return SX.str().strip("[]").split(", ")


def find_idx_for_labels(sub_vars: ca.SX, sub_label: str) -> list[int]:
    """Return a list of indices where sub_label is part of the variable label."""
    return [
        idx
        for idx, label in enumerate(sub_vars.str().strip("[]").split(", "))
        if sub_label in label
    ]
