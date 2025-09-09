from typing import NamedTuple, Sequence

import numpy as np
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
)


class AcadosOcpSolverInput(NamedTuple):
    """Input for an Acados solver representing a batch of concrete problem instances
    that should be solved.

    Attributes:
        x0: Initial state, shape (batch_size, nx)
        u0: Initial control input, shape (batch_size, nu), optional.
            If provided, the initial control input will be constrained to this.
        p_global: Global parameters, shape (batch_size, np_global), optional.
            If not provided, the default values set in the acados ocp object will be used.
        p_stagewise: Stage-wise parameters, shape (batch_size, N_horizon + 1, np_stagewise),
            or (batch_size, N_horizon + 1, len(p_stagewise_sparse_idx),
            if p_stagewise_sparse_idx is provided, optional.
            If not provided, the default values set in the acados ocp object will be used.
            Has to be provided if p_stagewise_sparse_idx is provided.
        p_stagewise_sparse_idx: If provided, the indices determine which elements of the
            total stagewise parameter in the solver should be overwritten
            by the provided p_stagewise values,
            shape (batch_size, N_horizon + 1, nindices), optional.
    """

    x0: np.ndarray
    u0: np.ndarray | None = None
    p_global: np.ndarray | None = None
    p_stagewise: np.ndarray | None = None
    p_stagewise_sparse_idx: np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self.x0.shape[0]

    def get_sample(self, idx: int) -> "AcadosOcpSolverInput":
        """Get the sample at index i from the batch."""

        def _g(data, idx):
            return None if data is None else data[idx]

        return AcadosOcpSolverInput(
            x0=self.x0[idx],
            u0=_g(self.u0, idx),
            p_global=_g(self.p_global, idx),
            p_stagewise=_g(self.p_stagewise, idx),
            p_stagewise_sparse_idx=_g(self.p_stagewise_sparse_idx, idx),
        )


def collate_acados_flattened_iterate_fn(
    batch: Sequence[AcadosOcpFlattenedIterate],
    collate_fn_map: dict | None = None,
) -> AcadosOcpFlattenedBatchIterate:
    return AcadosOcpFlattenedBatchIterate(
        x=np.stack([x.x for x in batch], axis=0),
        u=np.stack([x.u for x in batch], axis=0),
        z=np.stack([x.z for x in batch], axis=0),
        sl=np.stack([x.sl for x in batch], axis=0),
        su=np.stack([x.su for x in batch], axis=0),
        pi=np.stack([x.pi for x in batch], axis=0),
        lam=np.stack([x.lam for x in batch], axis=0),
        N_batch=len(batch),
    )


def collate_acados_flattened_batch_iterate_fn(
    batch: Sequence[AcadosOcpFlattenedBatchIterate],
    collate_fn_map: dict | None = None,
) -> AcadosOcpFlattenedBatchIterate:
    return AcadosOcpFlattenedBatchIterate(
        x=np.concat([x.x for x in batch], axis=0),
        u=np.concat([x.u for x in batch], axis=0),
        z=np.concat([x.z for x in batch], axis=0),
        sl=np.concat([x.sl for x in batch], axis=0),
        su=np.concat([x.su for x in batch], axis=0),
        pi=np.concat([x.pi for x in batch], axis=0),
        lam=np.concat([x.lam for x in batch], axis=0),
        N_batch=sum([x.N_batch for x in batch]),
    )


def _stack_safe(attr, batch):
    parts = [getattr(part, attr) for part in batch]

    if all(part is None for part in parts):
        return None

    return np.stack(parts, axis=0)


def collate_acados_ocp_solver_input(
    batch: Sequence[AcadosOcpSolverInput],
    collate_fn_map: dict | None = None,
) -> AcadosOcpSolverInput:
    """Collates a batch of AcadosOcpSolverInput objects into a single object."""

    return AcadosOcpSolverInput(
        x0=np.stack([input.x0 for input in batch], axis=0),
        u0=_stack_safe("u0", batch),
        p_global=_stack_safe("p_global", batch),
        p_stagewise=_stack_safe("p_stagewise", batch),
        p_stagewise_sparse_idx=_stack_safe("p_stagewise_sparse_idx", batch),
    )
