from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np
from acados_template import AcadosOcp
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
)

if TYPE_CHECKING:
    import torch


class AcadosOcpSolverInput(NamedTuple):
    """Input for an AcadosOcpSolver representing a batch of problem instances to be solved.

    Attributes:
        x0: Initial state, shape (batch_size, nx)
        u0: Initial control input, shape (batch_size, nu), optional.
            If provided, the initial control input will be constrained to this.
        p_global: Global parameters, shape (batch_size, np_global), optional.
            If not provided, the default values set in the acados ocp object will be used.
        p_stagewise: Stage-wise parameters, shape (batch_size, N_horizon + 1, np_stagewise),
            or (batch_size, len(p_stagewise_sparse_idx), np_stagewise), if p_stagewise_sparse_idx is
            provided, optional.
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
        """Get the sample at index `idx` from the batch."""
        return AcadosOcpSolverInput(
            None if self.x0 is None else self.x0[idx],
            None if self.u0 is None else self.u0[idx],
            None if self.p_global is None else self.p_global[idx],
            None if self.p_stagewise is None else self.p_stagewise[idx],
            None if self.p_stagewise_sparse_idx is None else self.p_stagewise_sparse_idx[idx],
        )


def collate_acados_flattened_iterate_fn(
    batch: Sequence[AcadosOcpFlattenedIterate], collate_fn_map: dict | None = None
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
        x=np.concatenate([x.x for x in batch], axis=0),
        u=np.concatenate([x.u for x in batch], axis=0),
        z=np.concatenate([x.z for x in batch], axis=0),
        sl=np.concatenate([x.sl for x in batch], axis=0),
        su=np.concatenate([x.su for x in batch], axis=0),
        pi=np.concatenate([x.pi for x in batch], axis=0),
        lam=np.concatenate([x.lam for x in batch], axis=0),
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


def validate_forward_inputs(
    ocp: AcadosOcp,
    x0: "np.ndarray | torch.Tensor",
    u0: "np.ndarray | torch.Tensor | None",
) -> None:
    """Validate the shapes of ``x0`` and ``u0`` before the forward solve.

    Works on any array-like exposing ``ndim`` and ``shape`` (numpy, torch, jax), so it can be
    shared by the torch and jax layers. Run before solver initialization so dimension mismatches
    fail fast with a clear message rather than as a cryptic C-level error.

    Raises:
        ValueError: If ``x0`` is not 2-D, its last dimension is not ``ocp.dims.nx``, ``u0`` is
            not 2-D, its batch does not match ``x0``, or its last dimension is not ``ocp.dims.nu``.
    """
    nx = ocp.dims.nx
    nu = ocp.dims.nu
    if x0.ndim != 2:
        raise ValueError(f"x0 must be 2-D (B, nx), got shape {tuple(x0.shape)}.")
    if x0.shape[1] != nx:
        raise ValueError(f"x0 has {x0.shape[1]} states, expected nx={nx}.")
    if u0 is not None:
        if u0.ndim != 2:
            raise ValueError(f"u0 must be 2-D (B, nu), got shape {tuple(u0.shape)}.")
        if u0.shape[0] != x0.shape[0]:
            raise ValueError(
                f"u0 has batch size {u0.shape[0]}, expected {x0.shape[0]} (matching x0)."
            )
        if u0.shape[1] != nu:
            raise ValueError(f"u0 has {u0.shape[1]} controls, expected nu={nu}.")
