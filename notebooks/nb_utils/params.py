"""Helpers for locating and reshaping parameters in the flat ``p_global`` layout.

NOTE: these helpers reach into leap-c private internals
(``manager._differentiable_parameter_store.indices`` and
``leap_c.parameters.utils._define_starts_and_ends``) — they are candidates
for promotion into ``leap_c.parameters`` proper.
"""

import numpy as np

from leap_c.parameters import AcadosParameterManager
from leap_c.parameters.utils import _define_starts_and_ends


def p_global_slice(manager: AcadosParameterManager, name: str) -> slice:
    """Return the slice of parameter ``name`` in the flat ``p_global`` vector.

    Follows the manager's registration order, so no index needs to be
    hard-coded. For a stage-varying parameter the slice covers all its
    segments in stage order (e.g. ``splits="stagewise"`` gives a slice of
    length ``(N_horizon + 1) * param_dim``).
    """
    param = manager.parameters[name]
    if param.interface != "differentiable":
        raise ValueError(f"Parameter '{name}' is not differentiable, so it has no p_global slice.")
    indices = manager._differentiable_parameter_store.indices
    if not param.is_stage_varying:
        start, end = indices[name]
        return slice(start, end)
    starts, ends = _define_starts_and_ends(param.splits, manager.N_horizon)
    segments = [indices[f"{name}_{a}_{b}"] for a, b in zip(starts, ends)]
    return slice(segments[0][0], segments[-1][1])


def expand_to_stages(
    manager: AcadosParameterManager, name: str, values: np.ndarray
) -> np.ndarray:
    """Expand per-segment values of a scalar parameter to per-stage values.

    ``values`` holds one value per segment (a single value for a global
    parameter); the result has ``N_horizon + 1`` entries, repeating each
    segment's value over the stages it covers. Useful for plotting what the
    OCP actually sees at every stage.
    """
    n_stages = manager.N_horizon + 1
    values = np.asarray(values, dtype=float).reshape(-1)
    param = manager.parameters[name]
    if not param.is_stage_varying:
        return np.full(n_stages, values.item())
    starts, ends = _define_starts_and_ends(param.splits, manager.N_horizon)
    out = np.zeros(n_stages)
    for seg, (a, b) in enumerate(zip(starts, ends)):
        out[a : b + 1] = values[seg]
    return out


def average_per_segment(
    manager: AcadosParameterManager, name: str, stage_values: np.ndarray
) -> np.ndarray:
    """Project per-stage values ``(N_horizon + 1,)`` of a scalar parameter onto its segments.

    Averages within each segment — the best a coarser ``splits`` layout can do
    to represent a finer stagewise signal. Returns ``(n_segments,)``
    (``(1,)`` for a global parameter).
    """
    stage_values = np.asarray(stage_values, dtype=float).reshape(-1)
    param = manager.parameters[name]
    if not param.is_stage_varying:
        return np.array([stage_values.mean()])
    starts, ends = _define_starts_and_ends(param.splits, manager.N_horizon)
    return np.array([stage_values[a : b + 1].mean() for a, b in zip(starts, ends)])
