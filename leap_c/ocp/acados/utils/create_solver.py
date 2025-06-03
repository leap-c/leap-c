"""Utilities for creating an AcadosOcpBatchSolver from an AcadosOcp object."""
import atexit
import shutil
from pathlib import Path
from tempfile import mkdtemp

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosOcpBatchSolver

from pathlib import Path

from acados_template import AcadosOcp, AcadosOcpBatchSolver, AcadosOcpSolver



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


class AcadosFileManager:
    """A simple class to manage the export directory for acados solvers.

    This class is used to manage the export directory of acados solvers. If
    the export directory is not provided, the class will create a temporary
    directory in /tmp. The export directory is deleted when an instance is
    garbage collected, but only if the export directory was not provided.
    """

    def __init__(
        self,
        export_directory: Path | None = None,
    ):
        """Initialize the export directory manager.

        Args:
            export_directory: The export directory if None create a folder in /tmp.
        """
        self.export_directory = (
            Path(mkdtemp()) if export_directory is None else export_directory
        )

        if export_directory is None:
            atexit.register(self.__del__)

    def setup_acados_ocp_solver(
        self, ocp: AcadosOcp, generate_code: bool = True, build: bool = True
    ) -> AcadosOcpSolver:
        """Setup an acados ocp solver with path management.

        We set the json file and the code export directory.

        Args:
            ocp: The acados ocp object.
            generate_code: If True generate the code.
            build: If True build the code.

        Returns:
            AcadosOcpSolver: The acados ocp solver.
        """
        ocp.code_export_directory = str(self.export_directory / "c_generated_code")
        json_file = str(self.export_directory / "acados_ocp.json")

        solver = AcadosOcpSolver(
            ocp, json_file=json_file, generate=generate_code, build=build
        )

        # we add the acados file manager to the solver to ensure
        # the export directory is deleted when the solver is garbage collected
        solver.__acados_file_manager = self  # type: ignore

        return solver

    def setup_acados_sim_solver(
        self, sim: AcadosSim, generate_code: bool = True, build: bool = True
    ) -> AcadosSimSolver:
        """Setup an acados sim solver with path management.

        We set the json file and the code export directory.

        Args:
            sim: The acados sim object.
            generate_code: If True generate the code.
            build: If True build the code.

        Returns:
            AcadosSimSolver: The acados sim solver.
        """
        sim.code_export_directory = str(self.export_directory / "c_generated_code")
        json_file = str(self.export_directory / "acados_ocp.json")

        solver = AcadosSimSolver(
            sim, json_file=json_file, generate=generate_code, build=build
        )

        # we add the acados file manager to the solver to ensure
        # the export directory is deleted when the solver is garbage collected
        solver.__acados_file_manager = self  # type: ignore

        return solver

    def setup_acados_ocp_batch_solver(
        self, ocp: AcadosOcp, N_batch_max: int, num_threads_in_batch_methods: int
    ) -> AcadosOcpBatchSolver:
        """Setup an acados ocp batch solver with path management.

        We set the json file and the code export directory.

        Args:
            ocp: The acados ocp object.
            N_batch_max: The batch size.
            num_threads_in_batch_methods: The number of threads to use for the batched methods.

        Returns:
            AcadosOcpBatchSolver: The acados ocp batch solver.
        """
        ocp.code_export_directory = str(self.export_directory / "c_generated_code")
        json_file = str(self.export_directory / "acados_ocp.json")

        solver = AcadosOcpBatchSolver(
            ocp,
            json_file=json_file,
            N_batch_max=N_batch_max,
            num_threads_in_batch_solve=num_threads_in_batch_methods,
        )

        # we add the acados file manager to the solver to ensure
        # the export directory is deleted when the solver is garbage collected
        solver.__acados_file_manager = self  # type: ignore

        return solver

    def __del__(self):
        shutil.rmtree(self.export_directory, ignore_errors=True)
