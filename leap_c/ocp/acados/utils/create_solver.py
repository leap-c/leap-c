"""Utilities for creating an AcadosOcpBatchSolver from an AcadosOcp object."""

from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

from acados_template import (
    AcadosOcp,
    AcadosOcpBatchSolver,
    AcadosOcpSolver,
)

from leap_c.ocp.acados.utils.delete_directory_hook import DeleteDirectoryHook


def create_batch_solver(
    ocp: AcadosOcp,
    export_directory: str | Path | None = None,
    discount_factor: float | None = None,
    n_batch_max: int = 256,
    num_threads: int = 4,
) -> AcadosOcpBatchSolver:
    """
    Create an AcadosOcpBatchSolver from an AcadosOcp object.

    Args:
        ocp: Acados optimal control problem formulation.
        export_directory: Directory to export the generated code. If None, a
            temporary directory is created and the directory is cleaned afterwards.
        discount_factor: Discount factor. If None, acados default cost
            scaling is used, i.e. dt for intermediate stages, 1 for
            terminal stage.
        n_batch_max: Maximum batch size.
        num_threads: Number of threads used in the batch solver.
    """
    ocp.solver_options.with_batch_functionality = True

    # translate cost terms to external to allow
    # implicit differentiation for a p_global parameter.
    if ocp.cost.cost_type_0 not in ["EXTERNAL", None]:
        ocp.translate_initial_cost_term_to_external(
            cost_hessian=ocp.solver_options.hessian_approx
        )
    if ocp.cost.cost_type not in ["EXTERNAL"]:
        ocp.translate_intermediate_cost_term_to_external(
            cost_hessian=ocp.solver_options.hessian_approx
        )
    if ocp.cost.cost_type_e not in ["EXTERNAL"]:
        ocp.translate_terminal_cost_term_to_external(
            cost_hessian=ocp.solver_options.hessian_approx
        )

    # TODO (Leonard): Check whether we still need this.
    _turn_on_warmstart(ocp)

    if export_directory is None:
        export_directory = Path(mkdtemp())
        add_delete_hook = True
    else:
        export_directory = Path(export_directory)
        add_delete_hook = False

    ocp.code_export_directory = str(export_directory / "c_generated_code")
    json_file = str(export_directory / "acados_ocp.json")

    try:
        batch_solver = AcadosOcpBatchSolver(
            ocp,
            json_file=json_file,
            N_batch_max=n_batch_max,
            num_threads_in_batch_solve=num_threads,
            build=False,
            generate=False,
        )
    except FileNotFoundError:
        batch_solver = AcadosOcpBatchSolver(
            ocp,
            json_file=json_file,
            N_batch_max=n_batch_max,
            num_threads_in_batch_solve=num_threads,
            build=True,
        )

    if discount_factor is not None:
        _set_discount_factor(batch_solver, discount_factor)

    if add_delete_hook:
        DeleteDirectoryHook(batch_solver, export_directory)

    return batch_solver


def create_forward_backward_batch_solvers(
    ocp: AcadosOcp,
    sensitivity_ocp: AcadosOcp | None = None,  # type:ignore
    export_directory: str | Path | None = None,
    discount_factor: float | None = None,
    n_batch_max: int = 256,
    num_threads: int = 4,
):
    # check if we can use the forward solver for the backward pass.
    need_backward_solver = _check_need_sensitivity_solver(ocp)

    if need_backward_solver:
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.with_value_sens_wrt_params = True

    forward_batch_solver = create_batch_solver(
        ocp,
        export_directory=export_directory,
        discount_factor=discount_factor,
        n_batch_max=n_batch_max,
        num_threads=num_threads,
    )

    if not need_backward_solver:
        return forward_batch_solver, forward_batch_solver

    if sensitivity_ocp is None:
        sensitivity_ocp: AcadosOcp = deepcopy(
            forward_batch_solver.ocp_solvers[0].acados_ocp  # type:ignore
        )
        sensitivity_ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        sensitivity_ocp.solver_options.qp_solver_ric_alg = 1
        sensitivity_ocp.solver_options.qp_solver_cond_N = (
            sensitivity_ocp.solver_options.N_horizon
        )
        sensitivity_ocp.solver_options.hessian_approx = "EXACT"
        sensitivity_ocp.solver_options.regularize_method = "NO_REGULARIZE"
        sensitivity_ocp.solver_options.exact_hess_constr = True
        sensitivity_ocp.solver_options.exact_hess_cost = True
        sensitivity_ocp.solver_options.exact_hess_dyn = True
        sensitivity_ocp.solver_options.fixed_hess = 0
        sensitivity_ocp.solver_options.levenberg_marquardt = 0.0
        sensitivity_ocp.solver_options.with_solution_sens_wrt_params = True
        sensitivity_ocp.solver_options.with_value_sens_wrt_params = True

        sensitivity_ocp.model.cost_expr_ext_cost_custom_hess_0 = None
        sensitivity_ocp.model.cost_expr_ext_cost_custom_hess = None
        sensitivity_ocp.model.cost_expr_ext_cost_custom_hess_e = None

    sensitivity_ocp.model.name += "_sensitivity"  # type:ignore

    sensitivity_ocp.ensure_solution_sensitivities_available()

    backward_batch_solver = create_batch_solver(
        sensitivity_ocp,  # type:ignore
        export_directory=export_directory,
        discount_factor=discount_factor,
        n_batch_max=n_batch_max,
        num_threads=num_threads,
    )

    return forward_batch_solver, backward_batch_solver


def _check_need_sensitivity_solver(ocp: AcadosOcp) -> bool:
    try:
        ocp.ensure_solution_sensitivities_available()
    except (ValueError, NotImplementedError):
        return True

    return False


def _turn_on_warmstart(acados_ocp: AcadosOcp):
    if not (
        acados_ocp.solver_options.qp_solver_warm_start
        and acados_ocp.solver_options.nlp_solver_warm_start_first_qp
        and acados_ocp.solver_options.nlp_solver_warm_start_first_qp_from_nlp
    ):
        print(
            "WARNING: Warmstart is not enabled. We will enable it for our initialization strategies to work properly."
        )
    acados_ocp.solver_options.qp_solver_warm_start = 0
    acados_ocp.solver_options.nlp_solver_warm_start_first_qp = True
    acados_ocp.solver_options.nlp_solver_warm_start_first_qp_from_nlp = True


def _set_discount_factor(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver, discount_factor: float
):
    if isinstance(ocp_solver, AcadosOcpSolver):
        for stage in range(ocp_solver.acados_ocp.solver_options.N_horizon + 1):  # type: ignore
            ocp_solver.cost_set(stage, "scaling", discount_factor**stage)
    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        for ocp_solver in ocp_solver.ocp_solvers:
            _set_discount_factor(ocp_solver, discount_factor)
    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, got {type(ocp_solver)}."
        )


def derive_sensitivity_ocp(ocp_sensitivity: AcadosOcp):
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
