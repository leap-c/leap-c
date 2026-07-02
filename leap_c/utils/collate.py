"""Collate helpers for leap-c context objects.

These helpers are useful when batching nested data with
``AcadosDiffMpcCtx`` objects inside, for example to reuse single-sample contexts
as one batched warm start.

For a full user-facing example, see ``notebooks/minimal_mpc.py``.

"""

from collections.abc import Sequence
from typing import Any

from leap_c.diff_mpc.function import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.utils.dependencies import require_torch

ACADOS_DIFF_MPC_COLLATE_FN_MAP = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}
"""PyTorch-style ``collate_fn_map`` for :class:`AcadosDiffMpcCtx`.

Use this when an external batching utility accepts a custom collate map and may
encounter ``AcadosDiffMpcCtx`` objects.  If users only need to stack contexts
directly, they can call :func:`collate_acados_diff_mpc_ctx` instead.
"""


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
        collate_fn_map={**default_collate_fn_map, **ACADOS_DIFF_MPC_COLLATE_FN_MAP},
    )
