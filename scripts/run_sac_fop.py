"""Main script to run SAC-FOP experiments."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from leap_c.examples import ExampleControllerName, ExampleEnvName, create_controller, create_env
from leap_c.run import (
    add_common_args,
    default_ckpt_modus,
    init_run,
    resolve_output_path,
    resolve_reuse_code_dir,
    setup_wandb,
)
from leap_c.torch.nn.extractor import ExtractorName
from leap_c.torch.rl.sac_fop import SacFopTrainer, SacFopTrainerConfig


@dataclass
class RunSacFopConfig:
    """Configuration for running SAC-FOP experiments.

    Attributes:
        env: The environment name.
        controller: The controller name.
        trainer: The trainer configuration.
        extractor: The feature extractor to use.
    """

    env: ExampleEnvName = "cartpole"
    controller: ExampleControllerName = "cartpole"
    trainer: SacFopTrainerConfig = field(default_factory=SacFopTrainerConfig)
    extractor: ExtractorName = "identity"


def create_cfg(
    env: ExampleEnvName,
    controller: ExampleControllerName | None,
    seed: int,
    variant: Literal["fop", "fopc", "foa"] = "fop",
    ckpt_modus: Literal["best", "last", "all", "none"] = "last",
) -> RunSacFopConfig:
    # ---- Configuration ----
    cfg = RunSacFopConfig()
    cfg.env = env
    cfg.controller = controller if controller is not None else env
    cfg.extractor = "identity"

    # Validate variant
    if variant not in ["fop", "fopc", "foa"]:
        raise ValueError(f"Invalid variant '{variant}'. Must be one of: fop, fopc, foa")

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = seed
    cfg.trainer.train_steps = 1_000_000 if env == "pointmass" else 200_000
    cfg.trainer.train_start = 0
    cfg.trainer.val_freq = 10_000
    cfg.trainer.val_num_rollouts = 20
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

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 1_000
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
    # Configure based on variant:
    # - fop: normal parameter noise, no entropy correction
    # - fopc: parameter noise with entropy correction
    # - foa: action noise (deterministic parameters, stochastic actions)
    if variant == "foa":
        cfg.trainer.actor.noise = "action"
        cfg.trainer.actor.entropy_correction = False
    else:
        cfg.trainer.actor.noise = "param"
        cfg.trainer.actor.entropy_correction = True if variant == "fopc" else False

    cfg.trainer.actor.extractor_name = cfg.extractor
    cfg.trainer.actor.distribution_name = "squashed_gaussian"
    cfg.trainer.actor.residual = False

    # ---- Section: cfg.trainer.actor.mlp ----
    cfg.trainer.actor.mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.actor.mlp.activation = "relu"
    cfg.trainer.actor.mlp.weight_init = "orthogonal"

    return cfg


def run_sac_fop(
    cfg: RunSacFopConfig,
    output_path: str | Path,
    device: int | str | torch.device,
    dtype: torch.dtype,
    reuse_code_dir: Path | None = None,
    with_val: bool = False,
) -> float:
    """Run the SAC-FOP trainer.

    Args:
        cfg: The configuration for running the controller.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint.
        device: The device to use.
        dtype: The torch dtype to use.
        reuse_code_dir: The directory to reuse compiled code from, if any.
        with_val: Whether to use a validation environment.
    """
    val_env = create_env(cfg.env, render_mode="rgb_array") if with_val else None
    trainer = SacFopTrainer(
        val_env=val_env,
        train_env=create_env(cfg.env),
        controller=create_controller(cfg.controller, reuse_code_dir),
        output_path=output_path,
        device=device,
        dtype=dtype,
        cfg=cfg.trainer,
        extractor_cls=cfg.extractor,
    )
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training of SAC-FOP agents.", formatter_class=ArgumentDefaultsHelpFormatter
    )
    groups = add_common_args(
        parser, has_controller=True, has_with_val=True, wandb_group_default="SAC-FOP"
    )
    groups["eval"].add_argument(
        "--variant",
        type=str,
        choices=("fop", "fopc", "foa"),
        default="fop",
        help="Variant of SAC-FOP to run. 'fop' is the standard version with parameter noise "
        "and no entropy correction, 'fopc' includes entropy correction, and 'foa' uses "
        "action noise instead of parameter noise.",
    )
    args = parser.parse_args()

    cfg = create_cfg(args.env, args.controller, args.seed, args.variant, default_ckpt_modus(args))
    tags = [f"sac_{args.variant}", args.env, args.controller]
    setup_wandb(args, cfg, tags)
    output_path = resolve_output_path(args, tags)
    reuse_code_dir = resolve_reuse_code_dir(args)

    run_sac_fop(cfg, output_path, args.device, args.dtype, reuse_code_dir, args.with_val)
