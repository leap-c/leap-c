"""Shared pieces for race_car SAC-ZOP / SAC-FOP training scripts.

Two utilities:

- ``RaceCarValChannelsMixin`` — Trainer mixin that plugs ``ChannelLogger``
  into ``_make_val_step_callback`` / ``validate``, so every SAC validation
  writes ``val_log_step*.{npz,json}`` alongside the training checkpoints.
  Same channel layout as ``scripts/race_car/run_baseline.py`` uses, so
  ``scripts/race_car/render_baseline.py`` reads SAC logs without changes.

- ``make_mismatched_env`` — builds a ``RaceCarEnv`` whose plant uses a
  perturbed copy of ``VEHICLE_PARAMS_DEFAULT``. The MPC keeps the nominal
  params, so the controller's internal model is deliberately wrong.
  Default perturbation: motor 15% weaker, rolling resistance +30%
  (linear and quadratic drag).
"""

from __future__ import annotations

from typing import Any

from channels import ChannelLogger, default_channels, header_from

from leap_c.examples.race_car.bicycle_model import VEHICLE_PARAMS_DEFAULT
from leap_c.examples.race_car.env import RaceCarEnv, RaceCarEnvConfig


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


def make_mismatched_vehicle_params(
    cm1_scale: float = 0.85,
    cr0_scale: float = 1.30,
    cr2_scale: float = 1.30,
) -> dict[str, float]:
    """Return a perturbed copy of ``VEHICLE_PARAMS_DEFAULT``.

    Perturbs the three parameters that appear in the ``dv/dt`` equation:
    motor coefficient ``Cm1`` (acceleration) and rolling resistance
    coefficients ``Cr0`` / ``Cr2`` (drag). Lateral dynamics (``C1``, ``C2``)
    are left untouched so the MPC's path constraint structure stays valid.
    """
    p = dict(VEHICLE_PARAMS_DEFAULT)
    p["Cm1"] = p["Cm1"] * cm1_scale
    p["Cr0"] = p["Cr0"] * cr0_scale
    p["Cr2"] = p["Cr2"] * cr2_scale
    return p


def make_mismatched_env(
    *,
    render_mode: str | None = None,
    cm1_scale: float = 0.85,
    cr0_scale: float = 1.30,
    cr2_scale: float = 1.30,
    max_steps: int = 4000,
) -> RaceCarEnv:
    """Build a ``RaceCarEnv`` whose plant uses perturbed vehicle params."""
    cfg = RaceCarEnvConfig(
        vehicle_params=make_mismatched_vehicle_params(cm1_scale, cr0_scale, cr2_scale),
        max_steps=max_steps,
    )
    return RaceCarEnv(render_mode=render_mode, cfg=cfg)
