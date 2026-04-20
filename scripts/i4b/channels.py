"""Declarative logging for the i4b baseline run.

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
``obs["state"] == ctx.iterate.x[0]``), ``info`` is the post-step info dict
returned by ``env.step``, ``action`` is the action just applied, ``ctx`` is
the policy context produced by that solve, and ``reward`` is the scalar step
reward.

Alignment convention
--------------------
One slider tick = one MPC tick. At the cursor:

- ``obs["state"]`` = ``ctx.iterate.x[0]`` (pre-step state)
- ``ctx.iterate.u[0]`` is the action applied from the cursor to cursor+dt
  (equal to ``info["T_hp_sup"]`` modulo clipping)
- Sequence overlays extend forward from the cursor: ``x[k]``, ``u[k]``
  plotted at ``cursor + k*dt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from leap_c.examples.i4b.env import I4bEnv
from leap_c.ocp.acados.utils.prepare_solver import prepare_batch_solver_for_backward

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
        import json
        from pathlib import Path

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


def default_channels(
    env: I4bEnv, planner: Any, *, compute_sensitivities: bool = False
) -> list[Channel]:
    """Build the default channel list for an i4b baseline validation run.

    The list is deliberately flat — to add or remove a logged variable, edit
    this function directly.
    """
    nx = planner._nx
    N = planner.cfg.N_horizon
    state_keys = tuple(env.state_keys)
    idx = {k: i for i, k in enumerate(state_keys)}
    has_forecast = env.cfg.N_forecast > 0

    def _obs(obs, *keys):
        v = obs
        for k in keys:
            v = v[k]
        return float(np.asarray(v).flat[0])

    def _state(obs, name):
        return float(obs["state"].flat[idx[name]])

    channels: list[Channel] = []

    # ── Per-state channels: scalar (post-step) + MPC prediction sequence ──
    def _make_state_pred(i):
        def _pred(obs, info, a, ctx, r):
            if ctx is None or getattr(ctx, "iterate", None) is None:
                return None
            return ctx.iterate.x[0].reshape(N + 1, nx)[:, i]

        return _pred

    for name in state_keys:
        channels.append(
            Channel(
                name=name,
                scalar=(lambda n: lambda obs, info, a, ctx, r: _state(obs, n))(name),
                sequence=_make_state_pred(idx[name]),
                ylabel=f"{name} [\u00b0C]",
            )
        )

    # ── Control: applied T_HP + MPC plan ──
    channels.append(
        Channel(
            name="T_hp_sup",
            scalar=lambda obs, info, a, ctx, r: float(info.get("T_hp_sup", float("nan"))),
            sequence=lambda obs, info, a, ctx, r: (
                None
                if ctx is None or getattr(ctx, "iterate", None) is None
                else ctx.iterate.u[0].reshape(N, -1)[:, 0]
            ),
            ylabel="T_hp_sup [\u00b0C]",
        )
    )

    # ── Setpoints (share the T_room panel) ──
    channels.append(
        Channel(
            name="T_set_lower",
            scalar=lambda obs, info, a, ctx, r: _obs(obs, "setpoints", "T_set_lower"),
            panel="T_room",
        )
    )
    channels.append(
        Channel(
            name="T_set_upper",
            scalar=lambda obs, info, a, ctx, r: _obs(obs, "setpoints", "T_set_upper"),
            panel="T_room",
        )
    )

    if False:
        # ── Disturbances ──
        channels.append(
            Channel(
                name="T_amb",
                scalar=lambda obs, info, a, ctx, r: _obs(obs, "disturbances", "T_amb"),
                ylabel="T_amb [\u00b0C]",
            )
        )

    if False:
        channels.append(
            Channel(
                name="Qdot_gains",
                scalar=lambda obs, info, a, ctx, r: _obs(obs, "disturbances", "Qdot_gains"),
                ylabel="Qdot_gains [W]",
            )
        )

    if False:
        # ── Solar irradiance (from forecast[:, 0] if present) ──
        if has_forecast:
            for key, label in [
                ("dhi", "dhi [W/m^2]"),
                ("ghi", "ghi [W/m^2]"),
                ("dni", "dni [W/m^2]"),
            ]:
                channels.append(
                    Channel(
                        name=key,
                        scalar=(
                            lambda k: lambda obs, info, a, ctx, r: float(
                                np.asarray(obs["forecast"][k]).flat[0]
                            )
                        )(key),
                        ylabel=label,
                    )
                )

    # ── Step metrics ──
    if False:
        channels.append(
            Channel(
                name="E_el_kWh",
                scalar=lambda obs, info, a, ctx, r: float(info.get("E_el_kWh", float("nan"))),
                ylabel="E_el [kWh]",
            )
        )
        channels.append(
            Channel(
                name="reward",
                scalar=lambda obs, info, a, ctx, r: float(r),
                ylabel="reward",
            )
        )

        # ── Solver stats (dict -> multiple scalar columns) ──
        def _solver_stats(obs, info, a, ctx, r):
            if ctx is None or not getattr(ctx, "stats", None):
                return {}
            s = ctx.stats[0] if isinstance(ctx.stats, list) else ctx.stats
            return {
                k: float(v)
                for k, v in s.items()
                if (k.startswith("time") or k == "sqp_iter") and np.isscalar(v)
            }

        channels.append(Channel(name="solver", scalars_dict=_solver_stats))

    if True:
        # ── ctx.iterate extras: slacks (sequence kind) ──
        def _seq_iter(attr):
            def _fn(obs, info, a, ctx, r):
                if ctx is None or getattr(ctx, "iterate", None) is None:
                    return None
                v = getattr(ctx.iterate, attr, None)
                if v is None:
                    return None
                arr = np.asarray(v)
                return arr[0] if arr.ndim > 1 else arr

            return _fn

        channels.append(Channel(name="slack_lower", sequence=_seq_iter("sl"), ylabel="sl"))
        channels.append(Channel(name="slack_upper", sequence=_seq_iter("su"), ylabel="su"))

    # ── du_dp sensitivity matrix (manual adjoint loop, optional) ──
    if compute_sensitivities:

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
            return np.stack(rows)  # (N, N+1)

        channels.append(
            Channel(
                name="du_dp",
                matrix=_du_dp,
                cmap="RdBu_r",
                ylabel=r"$\partial T_{HP}[k]\,/\,\partial \dot{Q}_{gains}[j]$  [\u00b0C/W]",
            )
        )

        # Precomputed sensitivity fields carried on ctx (populated only when a
        # gradient pass requests them; we try anyway and skip if absent).
        def _ctx_field(attr):
            def _fn(obs, info, a, ctx, r):
                v = getattr(ctx, attr, None) if ctx is not None else None
                if v is None:
                    return None
                arr = np.asarray(v)
                return arr[0] if arr.ndim > 1 else arr

            return _fn

        channels.append(Channel(name="du0_dp_global", sequence=_ctx_field("du0_dp_global")))
        channels.append(
            Channel(name="dx_dp_global", matrix=_ctx_field("dx_dp_global"), cmap="RdBu_r")
        )
        channels.append(Channel(name="dvalue_dp_global", sequence=_ctx_field("dvalue_dp_global")))

    return channels


def header_from(env: I4bEnv, planner: Any) -> dict:
    """Small metadata header saved alongside channels."""
    return {
        "delta_t_s": float(env.cfg.delta_t),
        "N_horizon": int(planner.cfg.N_horizon),
        "state_keys": list(env.state_keys),
    }
