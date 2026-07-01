"""Parameter type alias and numpy helpers for stage-varying MPC parameters."""

from typing import Literal

import numpy as np
from torch import Tensor

ParamSplits = list[int] | int | Literal["stagewise", "global"]
"""How a differentiable parameter varies across the MPC horizon.

- ``"global"``: one value shared across all stages.
- ``"stagewise"``: one independent value per stage (``N+1`` values total).
- ``int``: number of equal-sized stage segments.
- ``list[int]``: explicit stage boundaries (ascending), e.g. ``[4, 9]``.
"""


def n_segments(splits: ParamSplits, N_horizon: int) -> int:
    """Return the number of independent segments a ``splits`` specification produces.

    Args:
        splits: The split specification (see :data:`ParamSplits`).
        N_horizon: The MPC horizon length (number of shooting intervals).

    Returns:
        The number of segments: ``1`` for ``"global"``, ``N_horizon + 1`` for
        ``"stagewise"``, ``splits`` for an ``int``, and ``len(splits)`` for a list.
    """
    if splits == "global":
        return 1
    if splits == "stagewise":
        return N_horizon + 1
    if isinstance(splits, int):
        return splits
    return len(splits)


def broadcast_default_param(
    default: np.ndarray | dict[str, np.ndarray],
    obs: np.ndarray | Tensor | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    """Broadcast a per-stage default parameter to the batch shape implied by ``obs``.

    For a batched ``obs`` of shape ``(B, obs_dim)``, the default is broadcast to
    ``(B, *param_shape)``. Without ``obs`` (or a 1-D ``obs``) the unbatched default
    is returned. Handles both ``np.ndarray`` and ``dict[str, np.ndarray]`` defaults.
    """
    if isinstance(default, dict):
        out = {key: np.asarray(value) for key, value in default.items()}
        for key in out:
            if obs is not None and obs.ndim > 1:
                out[key] = np.broadcast_to(out[key], (*obs.shape[:-1], *out[key].shape))
        return out
    default = np.asarray(default)
    if obs is not None and obs.ndim > 1:
        default = np.broadcast_to(default, (*obs.shape[:-1], *default.shape))
    return default
