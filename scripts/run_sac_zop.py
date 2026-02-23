"""Main script to run SAC-ZOP experiments."""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

from leap_c.examples import ExampleControllerName, ExampleEnvName, create_controller, create_env
from leap_c.run import (
    default_controller_code_path,
    default_name,
    default_output_path,
    init_run,
    validate_torch_dtype_arg,
)
from leap_c.torch.nn.extractor import ExtractorName
from leap_c.torch.rl.sac_zop import SacZopTrainer, SacZopTrainerConfig


@dataclass
class RunSacZopConfig:
    """Configuration for running SAC-ZOP experiments."""

    env: ExampleEnvName = "cartpole"
    controller: ExampleControllerName = "cartpole"
    trainer: SacZopTrainerConfig = field(default_factory=SacZopTrainerConfig)
    extractor: ExtractorName = "identity"


def create_cfg(
    env: ExampleEnvName,
    controller: ExampleControllerName,
    seed: int,
    ckpt_modus: Literal["best", "last", "all", "none"] = "last",
) -> RunSacZopConfig:
    # ---- Configuration ----
    cfg = RunSacZopConfig()
    cfg.env = env
    cfg.controller = controller if controller is not None else env
    cfg.extractor = "identity" if env != "hvac" else "hvac"

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = seed
    cfg.trainer.train_steps = 1_000_000 if env == "pointmass" else 200_000
    cfg.trainer.train_start = 0
    cfg.trainer.val_freq = 10_000 if env != "hvac" else 50_000
    cfg.trainer.val_num_rollouts = 20 if env != "hvac" else 100
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 0
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = ckpt_modus
    cfg.trainer.batch_size = 64
    cfg.trainer.buffer_size = 1_000_000
    cfg.trainer.gamma = 0.99
    cfg.trainer.tau = 0.005
    cfg.trainer.soft_update_freq = 1
    cfg.trainer.lr_q = 0.001
    cfg.trainer.lr_pi = 0.001
    cfg.trainer.lr_alpha = 0.001
    cfg.trainer.init_alpha = 0.02
    cfg.trainer.target_entropy = None
    cfg.trainer.entropy_reward_bonus = True
    cfg.trainer.num_critics = 2
    cfg.trainer.update_freq = 4

    if env == "hvac":
        cfg.trainer.log.cumulative_metrics = [
            "train/money_spent",
            "train/energy_kwh",
            "train/constraint_violation",
        ]

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 1000
    cfg.trainer.log.window = 10_000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = True
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}

    # ---- Section: cfg.trainer.critic_mlp ----
    cfg.trainer.critic_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.critic_mlp.activation = "relu"
    cfg.trainer.critic_mlp.weight_init = "orthogonal"

    # ---- Section: cfg.trainer.actor ----
    cfg.trainer.actor.noise = "param"
    cfg.trainer.actor.extractor_name = cfg.extractor
    cfg.trainer.actor.distribution_name = "squashed_gaussian"
    cfg.trainer.actor.residual = True if env == "hvac" else False
    cfg.trainer.actor.entropy_correction = False

    # ---- Section: cfg.trainer.actor.mlp ----
    cfg.trainer.actor.mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.actor.mlp.activation = "relu"
    cfg.trainer.actor.mlp.weight_init = "orthogonal"

    return cfg


def run_sac_zop(
    cfg: RunSacZopConfig,
    output_path: str | Path,
    device: int | str | torch.device,
    dtype: torch.dtype,
    reuse_code_dir: Path | None = None,
    with_val: bool = False,
) -> float:
    """Run the SAC-ZOP trainer.

    Args:
        cfg: The configuration for running the controller.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint.
        device: The device to use.
        dtype: The dtype to use.
        reuse_code_dir: The directory to reuse compiled code from, if any.
        with_val: Whether to use a validation environment.
    """
    val_env = create_env(cfg.env, render_mode="rgb_array") if with_val else None
    trainer = SacZopTrainer(
        cfg=cfg.trainer,
        val_env=val_env,
        output_path=output_path,
        device=device,
        dtype=dtype,
        train_env=create_env(cfg.env),
        controller=create_controller(cfg.controller, reuse_code_dir),
        extractor_cls=cfg.extractor,
    )
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=validate_torch_dtype_arg, default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="cartpole")
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--with-val", action="store_true", help="Enable validation environment")
    parser.add_argument(
        "--ckpt-modus",
        type=str,
        default=None,
        choices=["none", "last", "all", "best"],
        help="Checkpoint mode. Defaults to 'best' with --with-val, 'last' otherwise.",
    )
    parser.add_argument(
        "-r",
        "--reuse_code",
        action="store_true",
        help="Reuse compiled code. The first time this is run, it will compile the code.",
    )
    parser.add_argument("--reuse_code_dir", type=Path, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="leap-c")
    args = parser.parse_args()

    if args.ckpt_modus is not None:
        ckpt_modus = args.ckpt_modus
    elif args.with_val:
        ckpt_modus = "best"
    else:
        ckpt_modus = "last"

    cfg = create_cfg(args.env, args.controller, args.seed, ckpt_modus)

    if args.use_wandb:
        config_dict = asdict(cfg)
        cfg.trainer.log.wandb_logger = True
        cfg.trainer.log.wandb_init_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": default_name(args.seed, tags=["sac_zop", args.env, args.controller]),
            "config": config_dict,
        }

    if args.output_path is None:
        output_path = default_output_path(
            seed=args.seed, tags=["sac_zop", args.env, args.controller]
        )
    else:
        output_path = args.output_path

    if args.reuse_code and args.reuse_code_dir is None:
        reuse_code_dir = default_controller_code_path()
    elif args.reuse_code_dir is not None:
        reuse_code_dir = args.reuse_code_dir
    else:
        reuse_code_dir = None

    run_sac_zop(
        cfg=cfg,
        output_path=output_path,
        device=args.device,
        dtype=args.dtype,
        reuse_code_dir=reuse_code_dir,
        with_val=args.with_val,
    )
