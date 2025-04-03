import atexit
import os
import random
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import casadi as ca
import numpy as np
import torch
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from acados_template.acados_ocp_batch_solver import AcadosOcpBatchSolver


def SX_to_labels(SX: ca.SX) -> list[str]:
    return SX.str().strip("[]").split(", ")


def find_idx_for_labels(sub_vars: ca.SX, sub_label: str) -> list[int]:
    """Return a list of indices where sub_label is part of the variable label."""
    return [
        idx
        for idx, label in enumerate(sub_vars.str().strip("[]").split(", "))
        if sub_label in label
    ]


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def collect_status(status: np.ndarray | torch.Tensor | list) -> list:
    """Count how many occurrences of the respective status number are given."""
    if isinstance(status, torch.Tensor) or isinstance(status, np.ndarray):
        return [(status == i).sum().item() for i in range(5)]
    elif isinstance(status, list):
        return [status.count(i) for i in range(5)]
    elif isinstance(status, int):
        template = [0, 0, 0, 0, 0]
        template[status] = 1
        return template


def put_each_index_of_tensor_as_entry_into(
    put_here: dict[str, Any], data: torch.Tensor | np.ndarray, name: str
):
    flat_data = data.reshape(-1)
    for i, entry in enumerate(flat_data):
        put_here[name + f"_{i}"] = entry.item()


def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


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
        self, ocp: AcadosOcp, N_batch: int, num_threads_in_batch_methods: int
    ) -> AcadosOcpBatchSolver:
        """Setup an acados ocp batch solver with path management.

        We set the json file and the code export directory.

        Args:
            ocp: The acados ocp object.
            N: The batch size.
            num_threads_in_batch_methods: The number of threads to use for the batched methods.

        Returns:
            AcadosOcpBatchSolver: The acados ocp batch solver.
        """
        ocp.code_export_directory = str(self.export_directory / "c_generated_code")
        json_file = str(self.export_directory / "acados_ocp.json")

        solver = AcadosOcpBatchSolver(
            ocp,
            json_file=json_file,
            N_batch=N_batch,
            num_threads_in_batch_solve=num_threads_in_batch_methods,
        )

        # we add the acados file manager to the solver to ensure
        # the export directory is deleted when the solver is garbage collected
        solver.__acados_file_manager = self  # type: ignore

        return solver

    def __del__(self):
        shutil.rmtree(self.export_directory, ignore_errors=True)


def add_prefix_extend(prefix: str, extended: dict, extending: dict) -> None:
    """
    Add a prefix to the keys of a dictionary and extend the with other dictionary with the result.
    Raises a ValueError if a key that has been extended with a prefix is already in the extended dict.
    """
    for k, v in extending.items():
        if extended.get(prefix + k, None) is not None:
            raise ValueError(f"Key {prefix + k} already exists in the dictionary.")
        extended[prefix + k] = v


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


def set_seed(seed: int):
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
