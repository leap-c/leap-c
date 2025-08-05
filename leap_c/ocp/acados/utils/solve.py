import time

from acados_template.acados_ocp_batch_solver import AcadosOcpBatchSolver
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate
import numpy as np

from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer
from leap_c.ocp.acados.utils.prepare_solver import prepare_batch_solver
from leap_c.ocp.acados.data import AcadosOcpSolverInput


def solve_with_retry(
    batch_solver: AcadosOcpBatchSolver,
    initializer: AcadosDiffMpcInitializer,
    ocp_iterate: AcadosOcpFlattenedBatchIterate | None,
    solver_input: AcadosOcpSolverInput,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve a batch of ocps, and retries in case of divergence.

    This function prepares the batch solver by loading the iterate, setting
    the initial conditions, and configuring the global and stage-wise
    parameters. If `p_global` or `p_stagewise` is not provided, it will
    check if the model has default parameters and load them accordingly.

    Args:
        batch_solver: The batch solver to use.
        initializer: The initializer used for retries.
        ocp_iterate: The iterate to load into the batch solver.
        solver_input: The input data for the solver, which includes initial
            conditions and parameters.

    Returns:
        The solving stats.
    """
    batch_size = solver_input.batch_size

    if ocp_iterate is None:
        ocp_iterate = initializer.batch_iterate(solver_input)
        with_retry = False
    else:
        with_retry = True

    prepare_batch_solver(batch_solver, ocp_iterate, solver_input)

    start = time.perf_counter()
    batch_solver.solve(n_batch=batch_size)
    time_solve = time.perf_counter() - start

    active_solvers = batch_solver.ocp_solvers[:batch_size]
    batch_status = np.array([solver.status for solver in active_solvers])

    if with_retry and any(status != 0 for status in batch_status):
        for idx, solver in enumerate(active_solvers):
            if batch_status[idx] == 0:
                continue
            single_iterate = initializer.single_iterate(solver_input.get_sample(idx))
            solver.load_iterate_from_flat_obj(single_iterate)

        start_retry = time.perf_counter()
        batch_solver.solve(n_batch=batch_size)
        time_solve += time.perf_counter() - start_retry

    batch_status_retry = np.array([solver.status for solver in active_solvers])

    stats = {
        "solving_time": time_solve,
        "success_rate": (batch_status_retry == 0).mean(),
        "retry_rate": (batch_status != 0).mean(),
    }

    return batch_status_retry, stats
