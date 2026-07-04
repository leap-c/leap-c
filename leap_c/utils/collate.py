"""Collate helpers for leap-c context objects.

These helpers are useful when batching nested data with
``AcadosDiffMpcCtx`` objects inside, for example to reuse single-sample contexts
as one batched warm start.

For a full user-facing example, see
``notebooks/getting_started/07_imitation_learning.py``.

"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from leap_c.diff_mpc.data import (
    collate_acados_flattened_batch_iterate_fn,
    collate_acados_ocp_solver_input,
)
from leap_c.diff_mpc.function import AcadosDiffMpcCtx
from leap_c.utils.dependencies import require_torch


def collate_acados_diff_mpc_ctx(
    batch: Sequence[AcadosDiffMpcCtx], collate_fn_map: dict[str, Callable] | None = None
) -> AcadosDiffMpcCtx:
    """Collate a batch of single-sample contexts into one batched context.

    Stacks the iterate and solver input arrays along a new batch axis and
    concatenates statuses into a 1-D array.  ``log`` is set to ``None`` since
    per-sample logs do not merge meaningfully.

    Args:
        batch: Non-empty sequence of :class:`AcadosDiffMpcCtx` objects, each
            holding a single-sample iterate and solver input.
        collate_fn_map: Optional PyTorch-style collate map.  Accepted for
            protocol compatibility but unused.

    Returns:
        A single :class:`AcadosDiffMpcCtx` whose ``iterate.N_batch`` equals
        ``len(batch)``.
    """
    return AcadosDiffMpcCtx(
        iterate=collate_acados_flattened_batch_iterate_fn([ctx.iterate for ctx in batch]),
        log=None,
        status=np.array([ctx.status for ctx in batch]),
        solver_input=collate_acados_ocp_solver_input([ctx.solver_input for ctx in batch]),
    )


def collate_torch(batch: Sequence[Any]) -> Any:
    """Collate a batch with PyTorch defaults plus leap-c context support.

    This is a convenience function for PyTorch ``DataLoader`` instances, replay
    buffers, or downstream training code that stores ``AcadosDiffMpcCtx`` values
    for batched warm starts.  PyTorch handles tensors, numpy arrays, mappings,
    tuples, scalars, etc.; leap-c only adds the missing rule for MPC contexts.

    Args:
        batch: Sequence of samples to collate.

    Returns:
        The PyTorch-collated batch.
    """
    require_torch()
    from torch.utils.data._utils.collate import collate, default_collate_fn_map

    return collate(
        batch,
        collate_fn_map={**default_collate_fn_map, **LEAPC_COLLATE_FN_MAP},
    )


LEAPC_COLLATE_FN_MAP = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}
"""PyTorch-style ``collate_fn_map`` for :class:`AcadosDiffMpcCtx`.

Use this when an external batching utility accepts a custom collate map and may
encounter ``AcadosDiffMpcCtx`` objects.  If users only need to stack contexts
directly, they can call :func:`collate_acados_diff_mpc_ctx` instead.
"""
