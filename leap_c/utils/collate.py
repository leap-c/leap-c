"""Collate helpers for leap-c context objects.

These helpers are useful when batching data structures that contain
``AcadosDiffMpcCtx`` objects, for example to reuse a list of single-sample
contexts as one batched warm start.

Example:
    ``collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}``

For PyTorch users, :func:`collate_torch` keeps the default PyTorch behavior for
tensors, numpy arrays, mappings, tuples, and scalars, and only adds the missing
custom rule for ``AcadosDiffMpcCtx``.
"""

from collections.abc import Callable, Sequence
from typing import Any

from leap_c.diff_mpc.function import AcadosDiffMpcCtx, collate_acados_diff_mpc_ctx
from leap_c.utils.dependencies import require_torch

CollateFnMap = dict[type | tuple[type, ...], Callable[..., Any]]

ACADOS_DIFF_MPC_COLLATE_FN_MAP: CollateFnMap = {
    AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx,
}
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
