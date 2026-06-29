"""Main script to run SAC experiments."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from leap_c.examples import ExampleEnvName, create_env
from leap_c.run import (
    add_common_args,
    default_ckpt_modus,
    init_run,
    resolve_output_path,
    setup_wandb,
)
from leap_c.torch.nn.extractor import ExtractorName
from leap_c.torch.rl.sac import SacTrainer, SacTrainerConfig


@dataclass
class RunSacConfig:
    """Configuration for running SAC experiments."""

    env: ExampleEnvName = "cartpole"
    trainer: SacTrainerConfig = field(default_factory=SacTrainerConfig)
    extractor: ExtractorName = "identity"


def create_cfg(
    env: ExampleEnvName,
    seed: int,
    ckpt_modus: Literal["best", "last", "all", "none"] = "last",
) -> RunSacConfig:
    """Return the default configuration for running SAC experiments."""
    # ---- Configuration ----
    cfg = RunSacConfig()
    cfg.env = env
    cfg.extractor = "identity"

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
    cfg.trainer.distribution_name = "squashed_gaussian"

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

    # ---- Section: cfg.trainer.actor_mlp ----
    cfg.trainer.actor_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.actor_mlp.activation = "relu"
    cfg.trainer.actor_mlp.weight_init = "orthogonal"

    return cfg


def run_sac(
    cfg: RunSacConfig,
    output_path: str | Path,
    device: int | str | torch.device,
    dtype: torch.dtype,
    with_val: bool = False,
) -> float:
    """Run the SAC trainer.

    Args:
        cfg: The configuration for running the controller.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint.
        device: The device to use.
        dtype: The dtype to use.
        with_val: Whether to use a validation environment.
    """
    val_env = create_env(cfg.env, render_mode="rgb_array") if with_val else None
    trainer = SacTrainer(
        cfg=cfg.trainer,
        val_env=val_env,
        output_path=output_path,
        device=device,
        dtype=dtype,
        train_env=create_env(cfg.env),
        extractor_cls=cfg.extractor,
    )
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training of SAC agents.", formatter_class=ArgumentDefaultsHelpFormatter
    )
    add_common_args(parser, has_controller=False, has_with_val=True, wandb_group_default="SAC")
    args = parser.parse_args()

    cfg = create_cfg(args.env, args.seed, default_ckpt_modus(args))
    tags = ["sac", args.env]
    setup_wandb(args, cfg, tags)
    output_path = resolve_output_path(args, tags)

    run_sac(cfg, output_path, args.device, args.dtype, args.with_val)
