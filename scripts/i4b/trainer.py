"""Custom SAC-FOP trainer for the i4b environment with per-episode validation logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import Env

from leap_c.controller import CtxType, ParameterizedController
from leap_c.torch.rl.mpc_actor import (
    StochasticMPCActorOutput,
)
from leap_c.torch.rl.sac_fop import SacFopTrainer, SacFopTrainerConfig


class I4bSacFopTrainer(SacFopTrainer):
    """SacFopTrainer with per-episode validation hooks for the i4b environment.

    Adds three capabilities on top of SacFopTrainer:

    1. ``act()`` stashes ``pi_output.param`` so the per-step callback can access
       the predicted ``Qdot_gains`` without requiring changes to the trainer
       signature.

    2. ``_make_val_step_callback()`` collects per-step records (true and
       predicted Qdot_gains, temperature, setpoints, solver status) and detects
       episode boundaries by watching the step counter reset to 1.

    3. ``validate()`` override calls ``_on_episode_end()`` once for each
       completed episode (including the last one, which the step-boundary
       detector cannot see) and then ``_on_validation_end()`` once after all
       episodes.
    """

    def __init__(
        self,
        cfg: SacFopTrainerConfig,
        val_env: Env | None,
        output_path: str | Path,
        device: int | str | torch.device,
        dtype: torch.dtype,
        train_env: Env,
        controller: ParameterizedController[CtxType],
        extractor_cls=None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            val_env=val_env,
            output_path=output_path,
            device=device,
            dtype=dtype,
            train_env=train_env,
            controller=controller,
            extractor_cls=extractor_cls,
        )
        # Set by act(); read by the step callback.
        self._last_act_param: np.ndarray | None = None
        # Filled by _make_val_step_callback, flushed by validate().
        self._pending_episode_flush: Any | None = None  # callable | None

    # ------------------------------------------------------------------
    # act() — stash predicted param for the callback
    # ------------------------------------------------------------------

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        state: CtxType | None = None,
    ) -> tuple[np.ndarray, CtxType | None, dict[str, float] | None]:
        obs = self.buffer.collate([obs])
        with torch.inference_mode():
            pi_output: StochasticMPCActorOutput = self.pi(obs, state, deterministic)
        action = pi_output.action.cpu().numpy()[0]  # type: ignore[union-attr]
        self._last_act_param = pi_output.param.cpu().numpy()[0]
        return action, pi_output.ctx, pi_output.stats

    # ------------------------------------------------------------------
    # per-step callback — detects episode boundaries
    # ------------------------------------------------------------------

    def _make_val_step_callback(self):
        episode_records: list[dict] = []
        prev_step_ref = [0]

        def _flush():
            if episode_records:
                self._on_episode_end(list(episode_records))
                episode_records.clear()

        # Expose flush so validate() can call it for the last episode.
        self._pending_episode_flush = _flush

        def callback(step: int, obs, action, reward, info, ctx) -> None:
            # step resets to 1 at the start of every new episode.
            if step == 1 and prev_step_ref[0] > 0:
                _flush()
            prev_step_ref[0] = step

            Qdot_gains_pred = (
                float(self._last_act_param[0]) if self._last_act_param is not None else float("nan")
            )
            Qdot_gains_true = float(obs["disturbances"]["Qdot_gains"].flat[0])
            T_set_lower = float(obs["setpoints"]["T_set_lower"].flat[0])
            T_set_upper = float(obs["setpoints"]["T_set_upper"].flat[0])
            T_room = float(info.get("T_room", float("nan")))

            record = {
                "step": step,
                # learnable parameter
                "Qdot_gains_pred": Qdot_gains_pred,
                "Qdot_gains_true": Qdot_gains_true,
                # context for the plot
                "T_amb": float(obs["disturbances"]["T_amb"].flat[0]),
                "quarter_hour": int(obs["forecast"]["quarter_hour"].flat[0]),
                "day_of_year": int(obs["forecast"]["day_of_year"].flat[0]),
                "day_of_week": int(obs["forecast"]["day_of_week"].flat[0]),
                # thermal comfort
                "T_set_lower": T_set_lower,
                "T_set_upper": T_set_upper,
                "T_room": T_room,
                "T_set_violated": int(T_room < T_set_lower or T_room > T_set_upper),
                # economics / reliability
                "E_el_kWh": float(info.get("E_el_kWh", float("nan"))),
                "reward": float(reward),
                "solver_status": (
                    int(ctx.status.flat[0]) if ctx is not None and hasattr(ctx, "status") else -1
                ),
            }
            episode_records.append(record)

        return callback

    # ------------------------------------------------------------------
    # episode-level hook — override to add custom logging/plotting
    # ------------------------------------------------------------------

    def _on_episode_end(self, records: list[dict]) -> None:
        """Called once per completed validation episode.

        ``records`` is a list of per-step dicts (one entry per env step).
        Override or extend this method to add custom metrics or plots.

        The default implementation:
        * logs aggregate scalars via ``report_stats`` (→ WandB / TensorBoard / CSV)
        * sends a matplotlib figure to WandB if a run is active
        """
        if not records:
            return

        df = pd.DataFrame(records)

        # --- scalar metrics -------------------------------------------------
        mae = float((df["Qdot_gains_true"] - df["Qdot_gains_pred"]).abs().mean())
        bias = float((df["Qdot_gains_pred"] - df["Qdot_gains_true"]).mean())
        n_violations = int(df["T_set_violated"].sum())
        solver_failures = int((df["solver_status"] != 0).sum())

        self.report_stats(
            "val_i4b",
            {
                "Qdot_gains_mae": mae,
                "Qdot_gains_bias": bias,
                "T_set_violations": n_violations,
                "solver_failures": solver_failures,
            },
            with_smoothing=False,
        )

        # --- wandb figure ---------------------------------------------------
        try:
            import wandb

            if wandb.run is not None:
                fig = plot_qdot_gains_episode(df)
                wandb.log(
                    {"val_i4b/qdot_gains": wandb.Image(fig)},
                    step=self.state.step,
                )
                plt.close(fig)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # validation-level hook — aggregate across all episodes
    # ------------------------------------------------------------------

    def _on_validation_end(self) -> None:
        """Called once after all validation episodes have completed.

        Override to add cross-episode aggregation.  The default is a no-op.
        """

    # ------------------------------------------------------------------
    # validate() override — wire episode / validation end hooks
    # ------------------------------------------------------------------

    def validate(self) -> float:
        self._pending_episode_flush = None
        score = super().validate()
        # Flush the last episode (the step-boundary detector inside the callback
        # cannot see it because no subsequent episode starts after it).
        if self._pending_episode_flush is not None:
            self._pending_episode_flush()
        self._on_validation_end()
        return score


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_qdot_gains_episode(df: pd.DataFrame) -> plt.Figure:
    """Return a two-panel matplotlib figure for a single validation episode.

    Top panel  : predicted vs true Qdot_gains [W], with T_amb on a secondary y-axis.
    Bottom panel: T_room vs comfort-band [T_set_lower, T_set_upper].

    Args:
        df: DataFrame with one row per env step, as produced by the step callback.
            Required columns: step, Qdot_gains_pred, Qdot_gains_true, T_amb,
            T_room, T_set_lower, T_set_upper.

    Returns:
        A ``matplotlib.figure.Figure``.  The caller is responsible for closing it
        (``plt.close(fig)``).
    """
    steps = df["step"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- top: Qdot_gains ---------------------------------------------------
    ax1 = axes[0]
    ax1.plot(steps, df["Qdot_gains_true"], label="Qdot_gains true", color="steelblue", lw=1.5)
    ax1.plot(
        steps,
        df["Qdot_gains_pred"],
        label="Qdot_gains pred",
        color="tomato",
        lw=1.5,
        ls="--",
    )
    ax1.set_ylabel("Heat gains [W]")
    ax1.legend(loc="upper left", fontsize=8)

    ax1r = ax1.twinx()
    ax1r.plot(steps, df["T_amb"], color="grey", lw=1, alpha=0.5, label="T_amb")
    ax1r.set_ylabel("T_amb [degC]", color="grey")
    ax1r.tick_params(axis="y", labelcolor="grey")

    mae = (df["Qdot_gains_true"] - df["Qdot_gains_pred"]).abs().mean()
    ax1.set_title(f"Qdot_gains  (MAE = {mae:.1f} W)")

    # --- bottom: thermal comfort -------------------------------------------
    ax2 = axes[1]
    ax2.plot(steps, df["T_room"], label="T_room", color="steelblue", lw=1.5)
    ax2.fill_between(
        steps,
        df["T_set_lower"],
        df["T_set_upper"],
        alpha=0.15,
        color="green",
        label="comfort band",
    )
    ax2.plot(steps, df["T_set_lower"], color="green", lw=0.8, ls=":")
    ax2.plot(steps, df["T_set_upper"], color="green", lw=0.8, ls=":")

    n_viol = int(df["T_set_violated"].sum())
    ax2.set_title(f"Thermal comfort  ({n_viol} violations)")
    ax2.set_ylabel("Temperature [degC]")
    ax2.set_xlabel("Step")
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig
