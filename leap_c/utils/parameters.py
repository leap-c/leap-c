"""Parameter type alias and numpy helpers for stage-varying MPC parameters."""

from typing import Literal

import numpy as np

ParamSplits = list[int] | int | Literal["stagewise", "global"]
"""How a learnable parameter varies across the MPC horizon.

- ``"global"``: one value shared across all stages.
- ``"stagewise"``: one independent value per stage (``N+1`` values total).
- ``int``: number of equal-sized stage segments.
- ``list[int]``: explicit stage boundaries (ascending), e.g. ``[4, 9]``.
"""


def stagewise_broadcast(
    value: np.ndarray,
    splits: ParamSplits,
    N_horizon: int,
) -> np.ndarray:
    """Broadcast a single-stage value to the public stagewise parameter shape.

    This helper intentionally returns a numpy array, not a Gymnasium space. Example planners can use
    it for bounds and defaults while keeping the public ``gym.spaces.Box`` construction explicit.
    """
    value = np.asarray(value)
    if splits == "global":
        return value
    Np1 = splits[-1] + 1 if isinstance(splits, list) else N_horizon + 1
    return np.broadcast_to(value, (Np1, *value.shape))
