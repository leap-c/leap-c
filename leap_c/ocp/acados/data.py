from typing import NamedTuple, Sequence

from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
)
import numpy as np


class AcadosOcpSolverInput(NamedTuple):
    """Input for an Acados solver.

    Can be a batch of inputs, or a single input.
    """

    x0: np.ndarray
    u0: np.ndarray | None = None
    p_global: np.ndarray | None = None
    p_stagewise: np.ndarray | None = None
    p_stagewise_sparse_idx: np.ndarray | None = None

    def is_batched(self) -> bool:
        return self.x0.ndim == 2

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        if not self.is_batched():
            raise ValueError("Cannot get batch size from non-batched MPCInput.")
        return self.x0.shape[0]

    def get_sample(self, idx: int) -> "AcadosOcpSolverInput":
        """Get the sample at index i from the batch."""
        if not self.is_batched():
            raise ValueError("Cannot sample from non-batched MPCInput.")

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
    collate_fn_map: dict = None,
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
    collate_fn_map: dict = None,
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
    collate_fn_map: dict = None,
) -> AcadosOcpSolverInput:
    """Collates a batch of AcadosOcpSolverInput objects into a single object."""

    return AcadosOcpSolverInput(
        x0=np.stack([input.x0 for input in batch], axis=0),
        u0=_stack_safe("u0", batch),
        p_global=_stack_safe("p_global", batch),
        p_stagewise=_stack_safe("p_stagewise", batch),
        p_stagewise_sparse_idx=_stack_safe("p_stagewise_sparse_idx", batch),
    )
