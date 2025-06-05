"""Provides logic for initializing AcadosOcpSolver."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import fields

from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
)
from acados_template.acados_ocp_solver import AcadosOcpSolver
import numpy as np

from leap_c.ocp.acados.data import (
    AcadosSolverInput,
    _collate_acados_flattened_iterate_fn,
)


class AcadosInitializer(ABC):
    """Abstract base class for initializing an AcadosOcpSolver.

    This class defines the interface for different initialization strategies
    for `AcadosOcpSolver` instances. Subclasses must implement the
    `single_iterate` method but can also overwrite the `batch_iterate` method
    for higher efficiency.
    """

    @abstractmethod
    def single_iterate(self, mpc_input: AcadosSolverInput) -> AcadosOcpFlattenedIterate:
        """Abstract method to generate an initial iterate for a single OCP.

        Subclasses must implement this method to provide a specific
        initialization strategy.

        Args:
            mpc_input: An `AcadosSolverInput` object containing the initial
                conditions and parameters for the OCP.

        Returns:
            An `AcadosOcpFlattenedIterate` representing the initial guess.
        """
        ...

    def batch_iterate(
        self, mpc_input: AcadosSolverInput
    ) -> AcadosOcpFlattenedBatchIterate:
        """Generates a batch of initial iterates for multiple OCPs.

        This method uses the `single_sample` method to generate an initial
        iterate for each OCP in the batch.

        Args:
            mpc_input: An `AcadosSolverInput` object containing the inputs for
                the batch of OCPs.

        Returns:
            A list of `AcadosOcpFlattenedIterate` objects, one for each OCP in
            the batch.
        """
        if not mpc_input.is_batched():
            raise ValueError("Batch sample requires a batched input.")

        iterates = [
            self.single_iterate(mpc_input.get_sample(i))
            for i in range(mpc_input.batch_size)
        ]

        return _collate_acados_flattened_iterate_fn(iterates)


class ZeroInitializer(AcadosInitializer):
    def __init__(self, solver: AcadosOcpSolver) -> None:
        iterate = solver.store_iterate_to_flat_obj()

        # Overwrite the iterate with zeros.
        for f in fields(iterate):
            n = f.name
            setattr(iterate, n, np.zeros_like(getattr(iterate, n)))  # type: ignore

        self.zero_iterate = iterate

    def single_iterate(
        self,
        mpc_input: AcadosSolverInput,
    ) -> AcadosOcpFlattenedIterate:
        return deepcopy(self.zero_iterate)
