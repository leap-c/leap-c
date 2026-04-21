"""Declarative logging for the race-car baseline run.

A ``Channel`` bundles "how to extract a variable from an env step" with "how to
render it". Add or remove entries in :func:`default_channels` to adapt both the
saved data and the interactive figure — the logger and renderer iterate the
same list, so neither side needs changes when the set of logged variables
changes.

Kinds
-----
Each channel may carry one or more extractor callables. The non-None ones
determine how the variable is stored:

- ``scalar``       : float                 -> (T,)       one line on the panel
- ``sequence``     : array shape (K,)      -> (T, K)     overlay, slider-driven
- ``matrix``       : array shape (H, W)    -> (T, H, W)  imshow, slider-driven
- ``scalars_dict`` : dict[str, float]      -> multiple (T,) arrays

All extractors receive ``(obs, info, action, ctx, reward)`` where ``obs`` is
the **pre-step** observation (the same ``obs`` the MPC solved from, so
``obs == ctx.iterate.x[0][:nx]``), ``info`` is the post-step info dict
returned by ``env.step``, ``action`` is the action just applied, ``ctx`` is
the policy context produced by that solve, and ``reward`` is the scalar step
reward.

Alignment convention
--------------------
One slider tick = one MPC tick. At the cursor:

- ``obs`` = ``ctx.iterate.x[0][:nx]`` (pre-step Frenet state)
- ``ctx.iterate.u[0]`` is the action applied from the cursor to cursor + dt
- Sequence overlays (state predictions, control plan, ``s_ref``) extend forward from the cursor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from leap_c.examples.race_car.env import RaceCarEnv

Extract = Callable[[dict, dict, Any, Any, float], Any]


@dataclass
class Channel:
    """A logged variable. See module docstring."""

    name: str
    scalar: Extract | None = None
    sequence: Extract | None = None
    matrix: Extract | None = None
    scalars_dict: Extract | None = None

    ylabel: str = ""
    unit: str = ""
    panel: str | None = None
    cmap: str = "RdBu_r"

    def kinds(self) -> list[str]:
        return [
            k
            for k in ("scalar", "sequence", "matrix", "scalars_dict")
            if getattr(self, k) is not None
        ]

    def resolved_panel(self) -> str:
        return self.panel if self.panel is not None else self.name

    def keys(self) -> dict[str, str]:
        """Array keys used in the NPZ for each array kind.

        When a channel carries both a scalar and a sequence (or other combo)
        the non-scalar kinds get a suffix to avoid collisions.
        """
        has_scalar = self.scalar is not None
        out: dict[str, str] = {}
        if has_scalar:
            out["scalar"] = self.name
        if self.sequence is not None:
            out["sequence"] = f"{self.name}_hrz" if has_scalar else self.name
        if self.matrix is not None:
            out["matrix"] = f"{self.name}_mat" if has_scalar or self.sequence else self.name
        return out


class ChannelLogger:
    """Record channel values per step, then save as a single NPZ + JSON."""

    def __init__(self, channels: list[Channel]):
        self.channels = channels
        self.scalars: dict[str, list[float]] = {}
        self.sequences: dict[str, list[np.ndarray]] = {}
        self.matrices: dict[str, list[np.ndarray]] = {}

    def record(self, obs: dict, info: dict, action: Any, ctx: Any, reward: float) -> None:
        for ch in self.channels:
            keys = ch.keys()
            if ch.scalar is not None:
                v = _safe_call(ch, "scalar", obs, info, action, ctx, reward)
                self.scalars.setdefault(keys["scalar"], []).append(
                    float("nan") if v is None else float(v)
                )
            if ch.scalars_dict is not None:
                d = _safe_call(ch, "scalars_dict", obs, info, action, ctx, reward) or {}
                for k, v in d.items():
                    self.scalars.setdefault(f"{ch.name}.{k}", []).append(float(v))
            if ch.sequence is not None:
                v = _safe_call(ch, "sequence", obs, info, action, ctx, reward)
                if v is not None:
                    self.sequences.setdefault(keys["sequence"], []).append(np.asarray(v))
            if ch.matrix is not None:
                v = _safe_call(ch, "matrix", obs, info, action, ctx, reward)
                if v is not None:
                    self.matrices.setdefault(keys["matrix"], []).append(np.asarray(v))

    def save(self, out_dir, step: int, header: dict) -> tuple[Any, Any]:
        out_dir = Path(out_dir)
        arrays: dict[str, np.ndarray] = {}
        for name, vals in self.scalars.items():
            arrays[name] = np.asarray(vals, dtype=np.float64)
        for name, vals in self.sequences.items():
            arrays[name] = np.stack(vals)
        for name, vals in self.matrices.items():
            arrays[name] = np.stack(vals)

        npz_path = out_dir / f"val_log_step{step}.npz"
        json_path = out_dir / f"val_log_step{step}.json"
        np.savez_compressed(npz_path, **arrays)

        present = set(arrays.keys())
        metadata = {
            "header": header,
            "channels": [_channel_metadata(ch, present) for ch in self.channels],
        }
        json_path.write_text(json.dumps(metadata, indent=2))
        return npz_path, json_path


def _safe_call(ch: Channel, kind: str, obs, info, action, ctx, reward):
    fn = getattr(ch, kind)
    try:
        return fn(obs, info, action, ctx, reward)
    except Exception as e:  # noqa: BLE001 — isolate a single channel's failure
        print(f"[channel {ch.name}.{kind}] {type(e).__name__}: {e}")
        return None


def _channel_metadata(ch: Channel, present_keys: set[str]) -> dict:
    """Emit serializable metadata for a channel.

    Limited to kinds that actually produced data this run.
    """
    ch_keys = ch.keys()
    kinds: list[str] = []
    out_keys: dict[str, str] = {}
    for kind in ("scalar", "sequence", "matrix"):
        if kind in ch_keys and ch_keys[kind] in present_keys:
            kinds.append(kind)
            out_keys[kind] = ch_keys[kind]
    if ch.scalars_dict is not None and any(k.startswith(f"{ch.name}.") for k in present_keys):
        kinds.append("scalars_dict")
    return {
        "name": ch.name,
        "kinds": kinds,
        "keys": out_keys,
        "panel": ch.resolved_panel(),
        "ylabel": ch.ylabel,
        "unit": ch.unit,
        "cmap": ch.cmap,
    }


STATE_NAMES = ("s", "n", "alpha", "v", "D", "delta")
STATE_UNITS = ("m", "m", "rad", "m/s", "-", "rad")
CONTROL_NAMES = ("derD", "derDelta")
CONTROL_UNITS = ("1/s", "rad/s")

NX = 6
NU = 2

__all__ = ["Channel", "ChannelLogger", "default_channels", "header_from"]


def default_channels(
    env: RaceCarEnv, planner: Any, *, compute_sensitivities: bool = False
) -> list[Channel]:
    """Channel list for a race-car validation run.

    Logs:
    - 6 state channels: scalar (current pre-step value) + sequence (MPC prediction).
    - 2 control channels: scalar (applied) + sequence (MPC plan).
    - ``s_ref`` sequence — derived deterministically from the current ``s`` and the planner's
      ``sref_lookahead``; overlaid on the ``s`` panel for direct comparison.
    - reward (scalar).
    - Two slack sequences for the soft constraints on ``a_long`` (idxsh[0]) and ``n`` (idxsh[1]).
    - Optional ``du_dp`` sensitivity matrix when ``compute_sensitivities=True``.
    """
    N = planner.cfg.N_horizon
    sref_lookahead = float(planner.cfg.sref_lookahead)

    def _scalar_state(i: int):
        return lambda obs, info, a, ctx, r: float(np.asarray(obs).flat[i])

    def _seq_state(i: int):
        def _fn(obs, info, a, ctx, r):
            if ctx is None or getattr(ctx, "iterate", None) is None:
                return None
            return ctx.iterate.x[0].reshape(N + 1, NX)[:, i]

        return _fn

    def _scalar_action(j: int):
        return lambda obs, info, a, ctx, r: float(np.asarray(a).flat[j])

    def _seq_action(j: int):
        def _fn(obs, info, a, ctx, r):
            if ctx is None or getattr(ctx, "iterate", None) is None:
                return None
            return ctx.iterate.u[0].reshape(N, NU)[:, j]

        return _fn

    channels: list[Channel] = []
    for i, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
        channels.append(
            Channel(
                name=name,
                scalar=_scalar_state(i),
                sequence=_seq_state(i),
                ylabel=f"{name} [{unit}]",
            )
        )
    for j, (name, unit) in enumerate(zip(CONTROL_NAMES, CONTROL_UNITS)):
        channels.append(
            Channel(
                name=name,
                scalar=_scalar_action(j),
                sequence=_seq_action(j),
                ylabel=f"{name} [{unit}]",
            )
        )

    def _s_ref_seq(obs, info, a, ctx, r):
        s0 = float(np.asarray(obs).flat[0])
        return s0 + sref_lookahead * np.arange(N + 1, dtype=np.float64) / N

    channels.append(
        Channel(
            name="s_ref",
            sequence=_s_ref_seq,
            panel="s",  # overlay on the s panel
            ylabel="s_ref [m]",
        )
    )

    channels.append(
        Channel(
            name="reward",
            scalar=lambda obs, info, a, ctx, r: float(r),
            ylabel="reward (= v*dt) [m]",
        )
    )

    # Slack sequences for the two soft constraints (idxsh = [0, 2] in the OCP).
    # Stage-major layout: ``sl`` has shape (N, 2). Pad stage 0 with NaN to align with the
    # (N+1,)-long state-prediction sequences.
    nsh = 2

    def _sl_per_stage(constraint_idx: int):
        def _fn(obs, info, a, ctx, r):
            if ctx is None or getattr(ctx, "iterate", None) is None:
                return None
            sl = np.asarray(ctx.iterate.sl)
            sl = sl[0] if sl.ndim > 1 else sl
            try:
                per_stage = sl.reshape(-1, nsh)[:, constraint_idx]
            except ValueError:
                return None
            out = np.full(N + 1, np.nan)
            out[1 : 1 + per_stage.shape[0]] = per_stage
            return out

        return _fn

    channels.append(
        Channel(name="sl_a_long", sequence=_sl_per_stage(0), ylabel="sl (a_long bound)")
    )
    channels.append(Channel(name="sl_n", sequence=_sl_per_stage(1), ylabel="sl (n bound)"))

    if compute_sensitivities:
        from leap_c.ocp.acados.utils.prepare_solver import (  # noqa: WPS433
            prepare_batch_solver_for_backward,
        )

        def _du_dp(obs, info, a, ctx, r):
            if ctx is None or getattr(ctx, "iterate", None) is None:
                return None
            diff_mpc_fun = planner.diff_mpc.diff_mpc_fun
            bwd = diff_mpc_fun.backward_batch_solver
            prepare_batch_solver_for_backward(bwd, ctx.iterate, ctx.solver_input)
            one = np.ones((1, 1, 1))
            rows = [
                bwd.eval_adjoint_solution_sensitivity([], [(k, one)], "p_global", False)[0, 0]
                for k in range(N)
            ]
            return np.stack(rows)

        channels.append(
            Channel(
                name="du_dp",
                matrix=_du_dp,
                cmap="RdBu_r",
                ylabel=r"$\partial u[k]\,/\,\partial p[j]$",
            )
        )

    return channels


def header_from(env: RaceCarEnv, planner: Any) -> dict:
    return {
        "delta_t_s": float(env.cfg.dt),
        "N_horizon": int(planner.cfg.N_horizon),
        "state_keys": list(STATE_NAMES),
    }
