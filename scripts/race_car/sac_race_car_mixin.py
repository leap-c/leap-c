"""Shared pieces for race_car SAC-ZOP / SAC-FOP training scripts.

Two utilities:

- ``RaceCarValChannelsMixin`` — Trainer mixin that plugs ``ChannelLogger``
  into ``_make_val_step_callback`` / ``validate``, so every SAC validation
  writes ``val_log_step*.{npz,json}`` alongside the training checkpoints.
  Same channel layout as ``scripts/race_car/run_baseline.py`` uses, so
  ``scripts/race_car/render_baseline.py`` reads SAC logs without changes.

- ``make_mismatched_env`` — builds a ``RaceCarEnv`` whose plant uses a
  perturbed copy of the nominal ``RaceCarDynamicsParameters``. The MPC keeps
  the nominal params, so the controller's internal model is deliberately wrong.
  Default perturbation: motor 15% weaker, rolling resistance +30%
  (linear and quadratic drag). Optionally adds Gaussian noise on top
  (``randomize_params=True``) and per-reset jitter on the initial state
  (``randomize_init_state=True``).
"""

from __future__ import annotations

import argparse
from typing import Any

from channels import ChannelLogger, default_channels, header_from
from episode_stats_wrapper import RaceCarEpisodeStats

from leap_c.examples.race_car.dynamics import RaceCarDynamicsParameters
from leap_c.examples.race_car.env import RaceCarEnv, RaceCarEnvConfig, RaceCarRewardConfig

REWARD_MODES = ("progress", "lap_time", "hybrid")
REWARD_WEIGHT_NAMES = (
    "w_progress",
    "w_time",
    "w_bonus",
    "w_violation",
    "w_lateral",
    "w_slip",
)


def _resolve_planner(trainer) -> Any:
    """Return the race_car planner if the trainer has one wired up, else None.

    SAC trainers store the controller under ``self.pi.controller`` (the
    ``HierachicalMPCActor``); the controller itself is a
    ``ControllerFromPlanner`` that exposes ``.planner``. Custom trainers that
    hold the controller directly (e.g. ``BaselineTrainer``) are supported via
    the fallback ``self.controller`` lookup.
    """
    pi = getattr(trainer, "pi", None)
    controller = getattr(pi, "controller", None) if pi is not None else None
    if controller is None:
        # Intentionally bypass nn.Module.__getattr__ semantics by using
        # __dict__ — ``self.controller`` on an nn.Module raises AttributeError
        # with a misleading message if the attribute is missing.
        controller = trainer.__dict__.get("controller", None)
    return getattr(controller, "planner", None)


class RaceCarValChannelsMixin:
    """Write race_car validation channels to NPZ/JSON each ``validate()`` call.

    Meant to be mixed in *before* a concrete SAC trainer, e.g.
    ``class RaceCarSacZopTrainer(RaceCarValChannelsMixin, SacZopTrainer): ...``.

    Assumes:
    - ``self.eval_env.unwrapped`` is a ``RaceCarEnv``.
    - The actor (``self.pi``) wraps a controller whose ``.planner`` is a
      ``RaceCarPlanner``.
    - The per-step callback's ``ctx`` argument is an ``AcadosDiffMpcCtx``
      with ``ctx.iterate.x[0]`` and ``ctx.iterate.u[0]`` populated
      (``default_channels`` reads pre-step state from there, so we pass
      ``obs_post`` as the ``obs`` arg and the channel extractors ignore it).
    """

    _val_logger: ChannelLogger | None

    def _make_val_step_callback(self):
        env = self.eval_env.unwrapped if self.eval_env is not None else None
        planner = _resolve_planner(self)
        if not isinstance(env, RaceCarEnv) or planner is None:
            return super()._make_val_step_callback()
        self._val_logger = ChannelLogger(default_channels(env, planner))

        def cb(step: int, obs_post, action, reward: float, info: dict, ctx: Any) -> None:
            # channels.default_channels reads pre-step state from ctx.iterate.x[0];
            # the ``obs`` argument is unused by the default extractors.
            self._val_logger.record(obs_post, info, action, ctx, reward)

        return cb

    def validate(self) -> float:
        self._val_logger = None
        score = super().validate()
        logger = getattr(self, "_val_logger", None)
        planner = _resolve_planner(self)
        env = self.eval_env.unwrapped if self.eval_env is not None else None
        if (
            logger is None
            or planner is None
            or not isinstance(env, RaceCarEnv)
            or not (logger.scalars or logger.sequences)
        ):
            return score
        header = header_from(env, planner)
        npz_path, json_path = logger.save(self.output_path, self.state.step, header)
        print(f"Channel log saved to: {npz_path}")
        print(f"Channel metadata saved to: {json_path}")
        return score


def make_mismatched_dynamics_params(
    cm1_scale: float = 0.85,
    cr0_scale: float = 1.30,
    cr2_scale: float = 1.30,
) -> RaceCarDynamicsParameters:
    """Return a perturbed ``RaceCarDynamicsParameters``.

    Perturbs the three parameters that appear in the ``dv/dt`` equation:
    motor coefficient ``Cm1`` (acceleration) and rolling resistance
    coefficients ``Cr0`` / ``Cr2`` (drag). Lateral dynamics (``C1``, ``C2``)
    are left untouched so the MPC's path constraint structure stays valid.
    """
    p = RaceCarDynamicsParameters()
    return RaceCarDynamicsParameters(
        m=p.m,
        C1=p.C1,
        C2=p.C2,
        Cm1=p.Cm1 * cm1_scale,
        Cm2=p.Cm2,
        Cr0=p.Cr0 * cr0_scale,
        Cr2=p.Cr2 * cr2_scale,
    )


def make_mismatched_env(
    *,
    render_mode: str | None = None,
    cm1_scale: float = 0.85,
    cr0_scale: float = 1.30,
    cr2_scale: float = 1.30,
    max_steps: int = 4000,
    reward: RaceCarRewardConfig | None = None,
    randomize_params: bool = False,
    param_noise_scale: float = 0.3,
    random_seed: int = 0,
    randomize_init_state: bool = False,
    init_state_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0),
    with_episode_stats: bool = True,
):
    """Build a race-car env whose plant uses perturbed dynamics params.

    The plant centre is set deterministically by ``cm1_scale, cr0_scale,
    cr2_scale``. With ``randomize_params=True`` a Gaussian perturbation
    (std = ``param_noise_scale * |value|`` per field) is sampled once at
    env construction and held for the env's lifetime. With
    ``randomize_init_state=True`` the first three components of the init
    state ``(s, n, alpha)`` are jittered uniformly by ``init_state_jitter``
    on every ``reset()``.

    Plant diversity across parallel envs is obtained by passing a different
    ``random_seed`` per env (mirrors HVAC).

    The returned env is wrapped in ``RaceCarEpisodeStats`` so each finished
    lap publishes a ``info["train_episode"]`` summary. Pass
    ``with_episode_stats=False`` to disable the wrapper (e.g. for
    rendering-only envs).
    """
    cfg = RaceCarEnvConfig(
        dynamics_params=make_mismatched_dynamics_params(cm1_scale, cr0_scale, cr2_scale),
        max_steps=max_steps,
        reward=reward if reward is not None else RaceCarRewardConfig(),
        randomize_params=randomize_params,
        param_noise_scale=param_noise_scale,
        random_seed=random_seed,
        randomize_init_state=randomize_init_state,
        init_state_jitter=init_state_jitter,
    )

    env = RaceCarEnv(render_mode=render_mode, cfg=cfg)
    if with_episode_stats:
        env = RaceCarEpisodeStats(env)
    return env


def add_reward_cli(parser: argparse.ArgumentParser) -> None:
    """Register ``--reward-mode`` plus one ``--reward-<weight>`` flag per term.

    Paired with ``build_reward_from_args``. The mode flag picks a preset; the
    per-weight flags default to ``None`` and override the preset when set,
    so a single CLI can reproduce any wiki option or arbitrary mix.
    """
    group = parser.add_argument_group("reward")
    group.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="progress",
        help=(
            "Preset for the step reward. 'progress' (default) is Option A "
            "(arc-length ds); 'lap_time' is Option B; 'hybrid' is Option C. "
            "Individual term weights can be overridden via --reward-<weight>."
        ),
    )
    for name in REWARD_WEIGHT_NAMES:
        flag = "--reward-" + name.replace("_", "-")
        group.add_argument(
            flag, type=float, default=None, help=f"Override {name} in the reward config."
        )


def build_reward_from_args(args: argparse.Namespace) -> RaceCarRewardConfig:
    """Build a ``RaceCarRewardConfig`` from argparse results registered by ``add_reward_cli``.

    Applies the ``--reward-mode`` preset first, then overrides individual weights
    from any ``--reward-w-*`` flags that were explicitly set. ``lap_time`` /
    ``hybrid`` read their preset scalars (``bonus``, ``c``, ``violation``) from
    the corresponding weight overrides when provided.
    """
    mode = args.reward_mode
    overrides = {name: getattr(args, f"reward_{name}") for name in REWARD_WEIGHT_NAMES}

    if mode == "progress":
        reward = RaceCarRewardConfig.progress()
    elif mode == "lap_time":
        bonus = overrides["w_bonus"] if overrides["w_bonus"] is not None else 100.0
        reward = RaceCarRewardConfig.lap_time(bonus=bonus)
    elif mode == "hybrid":
        c = overrides["w_time"] if overrides["w_time"] is not None else 0.1
        bonus = overrides["w_bonus"] if overrides["w_bonus"] is not None else 100.0
        violation = overrides["w_violation"] if overrides["w_violation"] is not None else 50.0
        reward = RaceCarRewardConfig.hybrid(c=c, bonus=bonus, violation=violation)
    else:
        raise ValueError(f"Unknown --reward-mode: {mode!r}")

    for name, value in overrides.items():
        if value is not None:
            setattr(reward, name, value)
    return reward
