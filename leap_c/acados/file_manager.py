import atexit
import shutil
from pathlib import Path
from tempfile import mkdtemp

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosOcpBatchSolver


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
