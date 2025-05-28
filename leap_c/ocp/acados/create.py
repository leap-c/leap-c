"""Utilities for creating an AcadosOcpBatchSolver from an AcadosOcp object."""
from pathlib import Path

from acados_template import AcadosOcp, AcadosOcpBatchSolver, AcadosOcpSolver

from .utils.file_manager import AcadosFileManager



def create_batch_solver(
    ocp: AcadosOcp,
    export_directory: Path | None = None,
    discount_factor: float = 0.99,
    n_batch_max: int = 256,
    num_threads: int = 4,
) -> AcadosOcpBatchSolver:
    """Creates an AcadosOcpBatchSolver from an AcadosOcp object.

    Args:
        ocp: Acados optimal control problem formulation.
        discount_factor: Discount factor. If None, acados default cost
            scaling is used, i.e. dt for intermediate stages, 1 for
            terminal stage.
        n_batch_max: Maximum batch size.
        num_threads: Number of threads used in the batch solver.
        export_directory: Directory to export the generated code.
    """

    _turn_on_warmstart(ocp)

     ocp.model.name += "_batch"  # type:ignore

    # TODO: Update the acados file manager
     batch_solver = afm_batch.setup_acados_ocp_batch_solver(
         ocp, self.n_batch_max, self._num_threads_in_batch_methods
     )

     if discount_factor is not None:
         _set_discount_factor(batch_solver, discount_factor)


     set_ocp_solver_to_default(
         batch_solver, self.default_full_mpcparameter, unset_u0=True
     )




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
