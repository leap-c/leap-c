"""Parameter type alias and numpy helpers for stage-varying MPC parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
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


def _define_starts_and_ends(splits: ParamSplits, N_horizon: int) -> tuple[list[int], list[int]]:
    """Compute the start and end stage indices for each segment given a splits policy.

    Args:
        splits: The split policy. Must be one of ``"global"``, ``"stagewise"``, a
            positive ``int`` (number of equal-sized segments), or a ``list[int]`` of
            ascending stage boundaries.
        N_horizon: The horizon length. Stages are indexed ``0`` to ``N_horizon``
            (inclusive), giving ``N_horizon + 1`` stages in total.

    Returns:
        A ``(starts, ends)`` pair of lists of equal length, where ``starts[i]`` and
        ``ends[i]`` are the inclusive start and end stage indices of segment ``i``.

    Example:
        ``_define_starts_and_ends([2, 5], 5)`` returns ``([0, 3], [2, 5])`` — two
        segments covering stages 0-2 and 3-5.
    """
    if not (
        splits in ("global", "stagewise")
        or (isinstance(splits, int) and splits > 0)
        or (isinstance(splits, list) and all(isinstance(x, int) for x in splits))
    ):
        raise ValueError(
            f"Invalid splits value: {splits!r}. Expected 'global', 'stagewise', a positive int, "
            "or a list[int]."
        )

    if splits == "global":
        ends = [N_horizon]
    elif splits == "stagewise":
        ends = list(range(N_horizon + 1))
    elif isinstance(splits, int):
        split_size = (N_horizon + 1) // splits
        remainder = (N_horizon + 1) % splits
        sizes = [split_size] * splits
        for i in range(remainder):
            sizes[i] += 1
        ends = (np.cumsum(sizes) - 1).tolist()
    elif isinstance(splits, list):
        ends = splits
    starts = [0] + [v + 1 for v in ends if v + 1 <= N_horizon]
    return starts, ends


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
