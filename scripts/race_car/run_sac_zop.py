r"""Train a SAC-ZOP agent on the race_car environment (with plant/model mismatch).

Mirrors ``scripts/run_sac_zop.py`` but:
- Hardcodes ``env = "race_car"`` and exposes ``--controller`` between
  ``{race_car, race_car_stagewise}``.
- Builds train / eval envs with a perturbed ``vehicle_params`` dict
  (see ``sac_race_car_mixin.make_mismatched_env``), giving SAC something
  to learn that the fixed-weight MPC cannot handle.
- Writes per-step validation channels (NPZ + JSON) via
  ``RaceCarValChannelsMixin`` so ``scripts/race_car/render_baseline.py``
  can visualise SAC validation laps with no changes.

Example:
-------
    python scripts/race_car/run_sac_zop.py --seed 0 --controller race_car \\
        --output-path output/race_car_sac_zop --with-val --reuse-code
"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from sac_race_car_mixin import (
    RaceCarValChannelsMixin,
    add_reward_cli,
    build_reward_from_args,
    make_mismatched_env,
)

from leap_c.examples import ExampleControllerName, create_controller
from leap_c.examples.race_car.env import RaceCarRewardConfig
from leap_c.run import (
    default_controller_code_path,
    default_name,
    default_output_path,
    init_run,
    validate_torch_device_arg,
    validate_torch_dtype_arg,
)
from leap_c.torch.nn.extractor import ExtractorName
from leap_c.torch.rl.sac_zop import SacZopTrainer, SacZopTrainerConfig

RACE_CAR_CONTROLLERS = ("race_car", "race_car_stagewise")


class RaceCarSacZopTrainer(RaceCarValChannelsMixin, SacZopTrainer):
    """SAC-ZOP trainer with race_car channel logging on validation."""


@dataclass
class RunSacZopRaceCarConfig:
    """Configuration for running SAC-ZOP on race_car."""

    controller: ExampleControllerName = "race_car"
    trainer: SacZopTrainerConfig = field(default_factory=SacZopTrainerConfig)
    extractor: ExtractorName = "identity"
    cm1_scale: float = 0.85
    cr0_scale: float = 1.30
    cr2_scale: float = 1.30
    max_steps: int = 4000
    reward: RaceCarRewardConfig = field(default_factory=RaceCarRewardConfig)


def create_cfg(
    controller: ExampleControllerName,
    seed: int,
    ckpt_modus: Literal["best", "last", "all", "none"] = "last",
) -> RunSacZopRaceCarConfig:
    cfg = RunSacZopRaceCarConfig()
    cfg.controller = controller
    cfg.extractor = "identity"

    # ---- Trainer ----
    cfg.trainer.seed = seed
    cfg.trainer.train_steps = 500_000
    cfg.trainer.train_start = 5_000
    cfg.trainer.val_freq = 25_000
    cfg.trainer.val_num_rollouts = 3
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 1
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = ckpt_modus
    cfg.trainer.batch_size = 256
    cfg.trainer.buffer_size = 1_000_000
    # Longer effective horizon than the default 2 s (dt=0.02, gamma=0.99) so the
    # agent values lap-completion more strongly.
    cfg.trainer.gamma = 0.995
    cfg.trainer.tau = 0.005
    cfg.trainer.soft_update_freq = 1
    cfg.trainer.lr_q = 3e-4
    cfg.trainer.lr_pi = 3e-4
    cfg.trainer.lr_alpha = 3e-4
    cfg.trainer.init_alpha = 0.05
    cfg.trainer.target_entropy = None
    cfg.trainer.entropy_reward_bonus = True
    cfg.trainer.num_critics = 2
    cfg.trainer.update_freq = 4

    # ---- Logger ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 1_000
    cfg.trainer.log.window = 10_000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = True
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}

    # ---- Critic MLP ----
    cfg.trainer.critic_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.critic_mlp.activation = "relu"
    cfg.trainer.critic_mlp.weight_init = "orthogonal"

    # ---- Actor (HierachicalMPCActor) ----
    cfg.trainer.actor.noise = "param"
    cfg.trainer.actor.extractor_name = cfg.extractor
    cfg.trainer.actor.distribution_name = "squashed_gaussian"
    cfg.trainer.actor.residual = False
    cfg.trainer.actor.entropy_correction = False
    cfg.trainer.actor.mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.actor.mlp.activation = "relu"
    cfg.trainer.actor.mlp.weight_init = "orthogonal"

    return cfg


def run_sac_zop(
    cfg: RunSacZopRaceCarConfig,
    output_path: str | Path,
    device: int | str | torch.device,
    dtype: torch.dtype,
    reuse_code_dir: Path | None = None,
    with_val: bool = False,
) -> float:
    def _make_env(render_mode: str | None = None):
        return make_mismatched_env(
            render_mode=render_mode,
            cm1_scale=cfg.cm1_scale,
            cr0_scale=cfg.cr0_scale,
            cr2_scale=cfg.cr2_scale,
            max_steps=cfg.max_steps,
            reward=cfg.reward,
        )

    val_env = _make_env(render_mode="rgb_array") if with_val else None
    trainer = RaceCarSacZopTrainer(
        cfg=cfg.trainer,
        val_env=val_env,
        output_path=output_path,
        device=device,
        dtype=dtype,
        train_env=_make_env(),
        controller=create_controller(cfg.controller, reuse_code_dir),
        extractor_cls=cfg.extractor,
    )
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train SAC-ZOP on race_car (with plant/model mismatch).",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    g = parser.add_argument_group("Run settings")
    g.add_argument("--output-path", type=Path, default=None)
    g.add_argument("--device", type=validate_torch_device_arg, default="cpu")
    g.add_argument("--dtype", type=validate_torch_dtype_arg, default="float32")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("-r", "--reuse-code", action="store_true")
    g.add_argument("--reuse-code-dir", type=Path, default=None)

    g = parser.add_argument_group("Train and eval")
    g.add_argument(
        "--controller",
        type=str,
        choices=list(RACE_CAR_CONTROLLERS),
        default="race_car",
    )
    g.add_argument("--with-val", action="store_true")
    g.add_argument(
        "--ckpt-modus",
        type=str,
        default=None,
        choices=["none", "last", "all", "best"],
    )
    g.add_argument("--max-steps", type=int, default=4000)

    g = parser.add_argument_group("Plant/model mismatch")
    g.add_argument(
        "--mismatch-cm1",
        type=float,
        default=0.85,
        help="Scale factor on Cm1 (motor coefficient) in the plant.",
    )
    g.add_argument(
        "--mismatch-cr0",
        type=float,
        default=1.30,
        help="Scale factor on Cr0 (rolling resistance) in the plant.",
    )
    g.add_argument(
        "--mismatch-cr2",
        type=float,
        default=1.30,
        help="Scale factor on Cr2 (quadratic drag) in the plant.",
    )

    add_reward_cli(parser)

    g = parser.add_argument_group("W&B logging")
    g.add_argument("--use-wandb", action="store_true")
    g.add_argument("--wandb-entity", type=str, default=None)
    g.add_argument("--wandb-project", type=str, default="leap-c")
    g.add_argument("--wandb-group", type=str, default="SAC-ZOP-race_car")

    args = parser.parse_args()

    if args.ckpt_modus is not None:
        ckpt_modus = args.ckpt_modus
    elif args.with_val:
        ckpt_modus = "best"
    else:
        ckpt_modus = "last"

    cfg = create_cfg(args.controller, args.seed, ckpt_modus)
    cfg.cm1_scale = args.mismatch_cm1
    cfg.cr0_scale = args.mismatch_cr0
    cfg.cr2_scale = args.mismatch_cr2
    cfg.max_steps = args.max_steps
    cfg.reward = build_reward_from_args(args)

    tags = ["sac_zop", "race_car", args.controller]

    if args.use_wandb:
        config_dict = asdict(cfg)
        cfg.trainer.log.wandb_logger = True
        cfg.trainer.log.wandb_init_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "group": args.wandb_group,
            "name": default_name(args.seed, tags=tags),
            "config": config_dict,
        }

    if args.output_path is None:
        output_path = default_output_path(seed=args.seed, tags=tags)
    else:
        output_path = args.output_path

    if args.reuse_code and args.reuse_code_dir is None:
        reuse_code_dir = default_controller_code_path()
    elif args.reuse_code_dir is not None:
        reuse_code_dir = args.reuse_code_dir
    else:
        reuse_code_dir = None

    run_sac_zop(cfg, output_path, args.device, args.dtype, reuse_code_dir, args.with_val)
