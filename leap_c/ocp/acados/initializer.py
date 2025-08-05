"""Provides logic for initializing AcadosDiffMpc."""

from abc import ABC, abstractmethod
from copy import deepcopy

from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
    AcadosOcpIterate,
)
import numpy as np

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
    def single_iterate(
        self, solver_input: AcadosOcpSolverInput
    ) -> AcadosOcpFlattenedIterate:
        """Abstract method to generate an initial iterate for a single OCP.

        Subclasses must implement this method to provide a specific
        initialization strategy.

        Args:
            solver_input: An `AcadosOcpSolverInput` object containing the initial
                conditions and parameters for the OCP.

        Returns:
            An `AcadosOcpFlattenedIterate` representing the initial guess.
        """
        ...

    def batch_iterate(
        self, solver_input: AcadosOcpSolverInput
    ) -> AcadosOcpFlattenedBatchIterate:
        """Generates a batch of initial iterates for multiple OCPs.

        This method uses the `single_sample` method to generate an initial
        iterate for each OCP in the batch.

        Args:
            solver_input: An `AcadosOcpSolverInput` object containing the inputs for
                the batch of OCPs.

        Returns:
            A list of `AcadosOcpFlattenedIterate` objects, one for each OCP in
            the batch.
        """
        if not solver_input.is_batched():
            raise ValueError("Batch sample requires a batched input.")

        iterates = [
            self.single_iterate(solver_input.get_sample(i))
            for i in range(solver_input.batch_size)
        ]

        return collate_acados_flattened_iterate_fn(iterates)


def create_zero_iterate_from_ocp(ocp: AcadosOcp) -> AcadosOcpIterate:
    # TODO (Jasper): Remove this function as soon as Acados updates the
    #   ocp.create_default_initial_iterate for slacked OCPs.

    ocp.make_consistent()
    dims = ocp.dims
    if ocp.constraints.has_x0:
        x_traj = (ocp.solver_options.N_horizon + 1) * [ocp.constraints.x0]  # type: ignore
    else:
        x_traj = (ocp.solver_options.N_horizon + 1) * [np.zeros(dims.nx)]  # type: ignore
    u_traj = ocp.solver_options.N_horizon * [np.zeros(dims.nu)]  # type: ignore
    z_traj = ocp.solver_options.N_horizon * [np.zeros(dims.nz)]  # type: ignore
    sl_traj = (
        [np.zeros(dims.ns_0)]
        + (ocp.solver_options.N_horizon - 1) * [np.zeros(dims.ns)]
        + [np.zeros(dims.ns_e)]
    )  # type: ignore
    su_traj = (
        [np.zeros(dims.ns_0)]
        + (ocp.solver_options.N_horizon - 1) * [np.zeros(dims.ns)]
        + [np.zeros(dims.ns_e)]
    )  # type: ignore

    pi_traj = ocp.solver_options.N_horizon * [np.zeros(dims.nx)]  # type: ignore

    ni_0 = dims.nbu + dims.nbx_0 + dims.nh_0 + dims.nphi_0 + dims.ng + dims.ns_0
    ni = dims.nbu + dims.nbx + dims.nh + dims.nphi + dims.ng + dims.ns
    ni_e = dims.nbx_e + dims.nh_e + dims.nphi_e + dims.ng_e + dims.ns_e
    lam_traj = (
        [np.zeros(2 * ni_0)]
        + (ocp.solver_options.N_horizon - 1) * [np.zeros(2 * ni)]
        + [np.zeros(2 * ni_e)]
    )  # type: ignore

    iterate = AcadosOcpIterate(
        x_traj=x_traj,  # type: ignore
        u_traj=u_traj,
        z_traj=z_traj,
        sl_traj=sl_traj,
        su_traj=su_traj,
        pi_traj=pi_traj,
        lam_traj=lam_traj,
    )
    return iterate


class ZeroDiffMpcInitializer(AcadosDiffMpcInitializer):
    def __init__(self, ocp: AcadosOcp) -> None:
        self.zero_iterate = create_zero_iterate_from_ocp(ocp).flatten()

    def single_iterate(
        self,
        solver_input: AcadosOcpSolverInput,
    ) -> AcadosOcpFlattenedIterate:
        return deepcopy(self.zero_iterate)
