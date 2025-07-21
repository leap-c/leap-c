from itertools import product

from acados_template import AcadosOcp
from acados_template.acados_ocp_batch_solver import AcadosOcpBatchSolver
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate
import casadi as ca
import numpy as np

from leap_c.ocp.acados.data import AcadosOcpSolverInput


# TODO (Jasper): The caching could be improved as soon as we save the whole
#    capsule in the context of the implicit function. Currently, this caching
#    is slightly less optimal, as we might need to prepare a solver multiple
#    times.
_PREPARE_CACHE = {}
_PREPARE_BACKWARD_CACHE = {}


def prepare_batch_solver(
    batch_solver: AcadosOcpBatchSolver,
    ocp_iterate: AcadosOcpFlattenedBatchIterate,
    solver_input: AcadosOcpSolverInput,
):
    # caching to improve performance
    if batch_solver in _PREPARE_CACHE:
        cached_ocp_iterate, cached_solver_input = _PREPARE_CACHE[batch_solver]
        if cached_ocp_iterate is ocp_iterate and cached_solver_input is solver_input:
            return
    _PREPARE_CACHE[batch_solver] = (ocp_iterate, solver_input)

    batch_size = solver_input.batch_size
    ocp: AcadosOcp = batch_solver.ocp_solvers[0].acados_ocp  # type:ignore
    active_solvers = batch_solver.ocp_solvers[:batch_size]
    N: int = ocp.solver_options.N_horizon  # type:ignore

    x0 = solver_input.x0
    u0 = solver_input.u0
    p_global = solver_input.p_global
    p_stagewise = solver_input.p_stagewise
    p_stagewise_sparse_idx = solver_input.p_stagewise_sparse_idx

    # iterate
    batch_solver.load_iterate_from_flat_obj(ocp_iterate)

    # set p_global
    if p_global is None and _is_param_legal(ocp.model.p_global):
        # if p_global is None and default exists, load default p_global
        for solver in active_solvers:
            solver.set_p_global_and_precompute_dependencies(ocp.p_global_values)
    elif p_global is not None:
        # if p_global is provided, set it
        for param, solver in zip(p_global, active_solvers):
            param = param.astype(np.float64)
            solver.set_p_global_and_precompute_dependencies(param)

    # set p_stagewise
    if (
        p_stagewise is None
        and _is_param_legal(ocp.model.p)
        and p_stagewise_sparse_idx is None
    ):
        # if p_stagewise is None and default exist, load default p
        param_default = np.tile(ocp.parameter_values, (batch_size, N + 1))
        param = param_default.reshape(batch_size, -1).astype(np.float64)
        batch_solver.set_flat("p", param)
    elif p_stagewise is not None and p_stagewise_sparse_idx is None:
        # if p_stagewise is provided, set it
        param = p_stagewise.reshape(batch_size, -1).astype(np.float64)
        batch_solver.set_flat("p", param)
    elif p_stagewise is not None and p_stagewise_sparse_idx is not None:
        # if p_stagewise is provided and sparse indices are provided, set it
        for idx, stage in product(range(batch_size), range(N + 1)):
            param = p_stagewise[idx, stage, :].astype(np.float64)
            solver = batch_solver.ocp_solvers[idx]
            solver.set_params_sparse(stage, param, p_stagewise_sparse_idx)

    # initial conditions
    for idx, solver in enumerate(active_solvers):
        solver.set(0, "x", x0[idx])
        solver.constraints_set(0, "lbx", x0[idx])
        solver.constraints_set(0, "ubx", x0[idx])

        if u0 is not None:
            solver.set(0, "u", u0[idx])
            solver.constraints_set(0, "lbu", u0[idx])
            solver.constraints_set(0, "ubu", u0[idx])
        else:
            solver.constraints_set(0, "lbu", ocp.constraints.lbu)
            solver.constraints_set(0, "ubu", ocp.constraints.ubu)


def prepare_batch_solver_for_backward(
    batch_solver: AcadosOcpBatchSolver,
    ocp_iterate: AcadosOcpFlattenedBatchIterate,
    solver_input: AcadosOcpSolverInput,
):
    if batch_solver in _PREPARE_BACKWARD_CACHE:
        cached_ocp_iterate, cached_solver_input = _PREPARE_BACKWARD_CACHE[batch_solver]
        if cached_ocp_iterate is ocp_iterate and cached_solver_input is solver_input:
            return
    _PREPARE_BACKWARD_CACHE[batch_solver] = (ocp_iterate, solver_input)

    prepare_batch_solver(batch_solver, ocp_iterate, solver_input)
    batch_solver.setup_qp_matrices_and_factorize(solver_input.batch_size)


def _is_param_legal(model_p) -> bool:
    # TODO (Leonard): Do we need this function?
    if model_p is None:
        return False
    elif isinstance(model_p, ca.SX):
        return 0 not in model_p.shape
    elif isinstance(model_p, np.ndarray):
        return model_p.size != 0
    elif isinstance(model_p, list) or isinstance(model_p, tuple):
        return len(model_p) != 0
    else:
        raise ValueError(f"Unknown case for model_p, type is {type(model_p)}")
