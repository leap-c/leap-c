from acados_template import AcadosOcp
from acados_template.acados_ocp_batch_solver import AcadosOcpBatchSolver
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate
import casadi as ca
import numpy as np

from leap_c.ocp.acados.data import AcadosSolverInput


# TODO: The caching could be improved as soon as we save the whole capsule
#    in the context of the implicit function. Currently, this caching
#    is slightly less optimal, as we might need to prepare a solver multiple
#    times.
_PREPARE_CACHE = {}
_PREPARE_BACKWARD_CACHE = {}


def batch_iterates_equal(
    obj1: AcadosOcpFlattenedBatchIterate,
    obj2: AcadosOcpFlattenedBatchIterate,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Compare two AcadosOcpFlattenedBatchIterate instances field by field.

    Args:
        obj1: First batch iterate
        obj2: Second batch iterate
        rtol: Relative tolerance for array comparisons
        atol: Absolute tolerance for array comparisons

    Returns:
        True if all fields are equal within tolerance
    """
    # Compare N_batch (integer field)
    if obj1.N_batch != obj2.N_batch:
        return False

    # Compare all array fields
    array_fields = ["x", "u", "z", "sl", "su", "pi", "lam"]

    for field_name in array_fields:
        arr1 = getattr(obj1, field_name)
        arr2 = getattr(obj2, field_name)

        if not np.allclose(arr1, arr2, rtol=rtol, atol=atol):
            return False

    return True


def solver_inputs_equal(
    input1: AcadosSolverInput,
    input2: AcadosSolverInput,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Compare two AcadosSolverInput instances for equality.

    Args:
        input1: First solver input
        input2: Second solver input
        rtol: Relative tolerance for array comparisons
        atol: Absolute tolerance for array comparisons

    Returns:
        True if all fields are equal within tolerance
    """
    # Compare x0 (required field)
    if not np.allclose(input1.x0, input2.x0, rtol=rtol, atol=atol):
        return False

    # Compare optional array fields
    optional_fields = ["u0", "p_global", "p_stagewise", "p_stagewise_sparse_idx"]

    for field_name in optional_fields:
        arr1 = getattr(input1, field_name)
        arr2 = getattr(input2, field_name)

        # Both None
        if arr1 is None and arr2 is None:
            continue

        # One is None, the other is not
        if arr1 is None or arr2 is None:
            return False

        # Both are arrays - compare with tolerance
        if not np.allclose(arr1, arr2, rtol=rtol, atol=atol):
            return False

    return True


def prepare_batch_solver(
    batch_solver: AcadosOcpBatchSolver,
    ocp_iterate: AcadosOcpFlattenedBatchIterate,
    solver_input: AcadosSolverInput,
):
    # caching to improve performance
    if batch_solver in _PREPARE_CACHE:
        cached_ocp_iterate, cached_solver_input = _PREPARE_CACHE[batch_solver]
        if batch_iterates_equal(
            cached_ocp_iterate, ocp_iterate
        ) and solver_inputs_equal(cached_solver_input, solver_input):
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
            solver.set_p_global_and_precompute_dependencies(param)

    # set p_stagewise
    if (
        p_stagewise is None
        and _is_param_legal(ocp.model.p)
        and p_stagewise_sparse_idx is None
    ):
        # if p_stagewise is None and default exist, load default p
        param_default = ocp.model.p.tile(batch_size, N + 1)  # type:ignore
        param = param_default.reshape(batch_size, -1)
        batch_solver.set_flat("p", param)
    elif p_stagewise is not None and p_stagewise_sparse_idx is None:
        # if p_stagewise is provided, set it
        param = p_stagewise.reshape(batch_size, -1)
        batch_solver.set_flat("p", param)
    elif p_stagewise is not None and p_stagewise_sparse_idx is not None:
        # if p_stagewise is provided and sparse indices are provided, set it
        for idx, stage in zip(range(batch_size), range(N + 1)):
            param = p_stagewise[idx, stage, :]
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


def prepare_batch_solver_for_backward(
    batch_solver: AcadosOcpBatchSolver,
    ocp_iterate: AcadosOcpFlattenedBatchIterate,
    solver_input: AcadosSolverInput,
):
    if batch_solver in _PREPARE_BACKWARD_CACHE:
        cached_ocp_iterate, cached_solver_input = _PREPARE_BACKWARD_CACHE[batch_solver]
        if cached_ocp_iterate == ocp_iterate and cached_solver_input == solver_input:
            return
    _PREPARE_BACKWARD_CACHE[batch_solver] = (ocp_iterate, solver_input)

    prepare_batch_solver(batch_solver, ocp_iterate, solver_input)
    batch_solver.setup_qp_matrices_and_factorize(solver_input.batch_size)


def _is_param_legal(model_p) -> bool:
    # TODO: Potentially remove this function.
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
