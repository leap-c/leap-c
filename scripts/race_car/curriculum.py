"""Stage-based curriculum presets for race_car SAC training.

One ``StageConfig`` per stage; ``stage_config(N)`` returns the preset that
``run_sac_fop.py`` / ``run_sac_zop.py`` apply when launched with ``--stage N``.

Stages turn on one source of difficulty at a time:

    1. Sanity     -- progress reward, plant == MPC model, no randomization.
    2. Time       -- hybrid reward (progress - c*dt + bonus - violation),
                     plant == MPC model.
    3. Lap-time   -- pure sparse lap-time reward, plant == MPC model.
    4. Mismatch   -- hybrid reward, fixed plant/model mismatch (current default).
    5. Robust     -- hybrid reward, Gaussian-perturbed plant. Each env instance
                     samples its plant once at construction and holds it for
                     the env's lifetime; parallel envs get diverse plants via
                     distinct ``random_seed`` values.
    6. Hardening  -- stage 5 plus per-reset uniform jitter on the init state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from leap_c.examples.race_car.env import RaceCarRewardConfig

RewardMode = Literal["progress", "lap_time", "hybrid"]

# Default early-stage validation cadence: short enough to see learning progress
# during sanity stages, long enough that solver wall-clock doesn't dominate.
_VAL_FREQ_EARLY = 5_000
_VAL_FREQ_LATE = 25_000
_VAL_NUM_ROLLOUTS = 5


@dataclass(kw_only=True)
class StageConfig:
    """Curriculum stage preset.

    Attributes:
        stage: Stage index (1..6).
        description: Short human-readable summary.
        reward_mode: Reward preset passed to ``RaceCarRewardConfig.<mode>()``.
        cm1_scale, cr0_scale, cr2_scale: Deterministic plant/model mismatch
            scales. Define the centre of the plant; if ``randomize_params``
            is True, Gaussian noise is added on top at env construction.
        randomize_params: If True, perturb every dynamics parameter once at
            env construction (Gaussian, std = ``param_noise_scale * |value|``).
        param_noise_scale: Std of the Gaussian perturbation applied when
            ``randomize_params=True``.
        randomize_init_state: If True, jitter the init state at each reset.
        init_state_jitter: Per-component half-width of the uniform jitter on
            ``[s, n, alpha]``. The remaining components stay at zero.
        val_freq: Trainer ``val_freq`` for this stage.
        val_num_rollouts: Trainer ``val_num_rollouts`` for this stage.
    """

    stage: int
    description: str
    reward_mode: RewardMode
    cm1_scale: float = 1.0
    cr0_scale: float = 1.0
    cr2_scale: float = 1.0
    randomize_params: bool = False
    param_noise_scale: float = 0.2
    randomize_init_state: bool = False
    init_state_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0)
    val_freq: int = _VAL_FREQ_LATE
    val_num_rollouts: int = _VAL_NUM_ROLLOUTS

    def build_reward(self) -> RaceCarRewardConfig:
        """Build the ``RaceCarRewardConfig`` matching this stage's preset."""
        if self.reward_mode == "progress":
            return RaceCarRewardConfig.progress()
        if self.reward_mode == "hybrid":
            return RaceCarRewardConfig.hybrid()
        if self.reward_mode == "lap_time":
            return RaceCarRewardConfig.lap_time()
        raise ValueError(f"Unknown reward_mode: {self.reward_mode!r}")


_STAGES: dict[int, StageConfig] = {
    1: StageConfig(
        stage=1,
        description="Sanity / wiring: progress reward, plant == MPC model.",
        reward_mode="progress",
        val_freq=_VAL_FREQ_EARLY,
    ),
    2: StageConfig(
        stage=2,
        description="Time pressure: hybrid reward, plant == MPC model.",
        reward_mode="hybrid",
        val_freq=_VAL_FREQ_EARLY,
    ),
    3: StageConfig(
        stage=3,
        description="Pure sparse lap-time reward, plant == MPC model.",
        reward_mode="lap_time",
        val_freq=_VAL_FREQ_EARLY,
    ),
    4: StageConfig(
        stage=4,
        description="Fixed plant/model mismatch (Cm1=0.85, Cr0=Cr2=1.30).",
        reward_mode="hybrid",
        cm1_scale=0.85,
        cr0_scale=1.30,
        cr2_scale=1.30,
    ),
    5: StageConfig(
        stage=5,
        description="Per-env-init Gaussian-randomized plant (held for the env's lifetime).",
        reward_mode="hybrid",
        cm1_scale=0.9,
        cr0_scale=1.2,
        cr2_scale=1.2,
        randomize_params=True,
        param_noise_scale=0.2,
    ),
    6: StageConfig(
        stage=6,
        description="Stage 5 plus per-reset init-state jitter.",
        reward_mode="hybrid",
        cm1_scale=0.9,
        cr0_scale=1.2,
        cr2_scale=1.2,
        randomize_params=True,
        param_noise_scale=0.2,
        randomize_init_state=True,
        # Half-widths on (s, n, alpha). Keep n inside the track; alpha small.
        init_state_jitter=(0.5, 0.05, 0.05),
    ),
}


STAGES: tuple[int, ...] = tuple(sorted(_STAGES))


def stage_config(stage: int) -> StageConfig:
    """Return the ``StageConfig`` for ``stage`` (1..6)."""
    if stage not in _STAGES:
        raise ValueError(f"Unknown stage {stage!r}; available: {STAGES}")
    return _STAGES[stage]
