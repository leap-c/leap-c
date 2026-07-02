"""Collate helpers for leap-c context objects.

The most common custom batching case in the core package is stacking
:class:`~leap_c.diff_mpc.function.AcadosDiffMpcCtx` objects.  The collate
function itself lives next to the context definition; this module exposes a
ready-made ``collate_fn_map`` entry for callers that use a PyTorch-style collate
utility.

Example:
    ``collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}``

For now this module intentionally does not provide a generic torch/jax collate
wrapper.  Those libraries have different batching conventions.  The PyTorch
helper below is intentionally thin: it delegates all standard cases to PyTorch's
default collate and only adds the custom rule for ``AcadosDiffMpcCtx``.
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
