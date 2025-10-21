"""Provides logic for initializing AcadosDiffMpc."""

from abc import ABC, abstractmethod
from copy import deepcopy

from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
)

from leap_c.ocp.acados.data import (
    AcadosOcpSolverInput,
    collate_acados_flattened_iterate_fn,
)


class AcadosDiffMpcInitializer(ABC):
    """Abstract base class for initializing an AcadosDiffMpc.

    This class defines the interface for different initialization strategies
    for `AcadosDiffMpc` instances. Subclasses must implement the
    `single_iterate` method but can also overwrite the `batch_iterate` method
    for higher efficiency.
    """

    @abstractmethod
    def single_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedIterate:
        """Abstract method to generate an initial iterate for a single problem instance.

        Subclasses must implement this method to provide a specific
        initialization strategy.

        Args:
            solver_input: An input object containing the initial
                conditions and parameters for the problem to solve.

        Returns:
            An iterate object representing the initial guess.
        """
        ...

    def batch_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedBatchIterate:
        """Generates a batch of initial iterates for a batch of problem instances.

        This method uses the `single_sample` method to generate an initial
        iterate for each OCP in the batch.

        Args:
            solver_input: An `AcadosOcpSolverInput` object containing the inputs for
                the batch of OCPs.

        Returns:
            A batched iterate object, containing one iterate object for each problem instance in
            the input batch.
        """
        iterates = [
            self.single_iterate(solver_input.get_sample(i)) for i in range(solver_input.batch_size)
        ]

        return collate_acados_flattened_iterate_fn(iterates)


class ZeroDiffMpcInitializer(AcadosDiffMpcInitializer):
    """An initializer that always returns an iterate with all values set to zero."""

    def __init__(self, ocp: AcadosOcp) -> None:
        self.zero_iterate = ocp.create_default_initial_iterate().flatten()

    def single_iterate(
        self,
        solver_input: AcadosOcpSolverInput,
    ) -> AcadosOcpFlattenedIterate:
        return deepcopy(self.zero_iterate)
