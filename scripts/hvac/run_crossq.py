"""Main script to run CrossQ experiments."""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from leap_c.examples import ExampleEnvName, create_env
from leap_c.run import default_name, default_output_path, init_run
from leap_c.torch.nn.extractor import ExtractorName, get_extractor_cls
from leap_c.torch.rl.crossq import CrossQTrainer, CrossQTrainerConfig


@dataclass
class RunCrossQConfig:
    """Configuration for running CrossQ experiments."""

    env: ExampleEnvName = "hvac"
    trainer: CrossQTrainerConfig = field(default_factory=CrossQTrainerConfig)
    extractor_name: ExtractorName = "hvac"


def create_cfg(
    env: ExampleEnvName,
    seed: int,
    ckpt_modus: Literal["best", "last", "all", "none"] = "last",
) -> RunCrossQConfig:
    """Return the default configuration for running CrossQ experiments."""
    # ---- Configuration ----
    cfg = RunCrossQConfig()
    cfg.env = env
    cfg.extractor_name = "hvac"

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = seed
    cfg.trainer.train_steps = 1_000_000 if env == "pointmass" else 200_000
    cfg.trainer.train_start = 0
    cfg.trainer.val_freq = 50_000
    cfg.trainer.val_num_rollouts = 100
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 0
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = ckpt_modus
    cfg.trainer.batch_size = 512
    cfg.trainer.buffer_size = 1_000_000
    cfg.trainer.gamma = 0.999
    cfg.trainer.lr_q = 0.0001
    cfg.trainer.lr_pi = 0.0001
    cfg.trainer.lr_alpha = 0.0001
    cfg.trainer.init_alpha = 0.02
    cfg.trainer.target_entropy = None
    cfg.trainer.entropy_reward_bonus = True
    cfg.trainer.num_critics = 10
    cfg.trainer.update_freq = 4
    cfg.trainer.distribution_name = "squashed_gaussian"

    # cumulative metrics to reduce noise in performance estimate
    cfg.trainer.log.cumulative_metrics = [
        "train/money_spent",
        "train/energy_kwh",
        "train/constraint_violation",
    ]

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 1_000
    cfg.trainer.log.window = 10_000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = True
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}

    # ---- Section: cfg.trainer.critic_mlp ----
    cfg.trainer.critic_mlp.hidden_dims = (512, 512, 512)
    cfg.trainer.critic_mlp.activation = "leaky_relu"
    cfg.trainer.critic_mlp.weight_init = "orthogonal"

    # ---- Section: cfg.trainer.actor_mlp ----
    cfg.trainer.actor_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.actor_mlp.activation = "leaky_relu"
    cfg.trainer.actor_mlp.weight_init = "orthogonal"

    return cfg


def run_crossq(
    cfg: RunCrossQConfig,
    output_path: str | Path,
    device: str = "cuda",
    with_val: bool = False,
) -> float:
    """Run the CrossQ trainer.

    Args:
        cfg: The configuration for running the controller.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint.
        device: The device to use.
        with_val: Whether to use a validation environment.
    """
    val_env = create_env(cfg.env, render_mode="rgb_array") if with_val else None
    trainer = CrossQTrainer(
        cfg=cfg.trainer,
        val_env=val_env,
        output_path=output_path,
        device=device,
        train_env=create_env(cfg.env),
    )
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="cartpole")
    parser.add_argument("--with-val", action="store_true", help="Enable validation environment")
    parser.add_argument(
        "--ckpt-modus",
        type=str,
        default=None,
        choices=["none", "last", "all", "best"],
        help="Checkpoint mode. Defaults to 'best' with --with-val, 'last' otherwise.",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="leap-c")
    args = parser.parse_args()

    if args.output_path is None:
        output_path = default_output_path(seed=args.seed, tags=["crossq", args.env])
    else:
        output_path = args.output_path

    if args.ckpt_modus is not None:
        ckpt_modus = args.ckpt_modus
    elif args.with_val:
        ckpt_modus = "best"
    else:
        ckpt_modus = "last"

    cfg = create_cfg(args.env, args.seed, ckpt_modus)

    if args.use_wandb:
        config_dict = asdict(cfg)
        cfg.trainer.log.wandb_logger = True
        cfg.trainer.log.wandb_init_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": default_name(args.seed, tags=["crossq", args.env]),
            "config": config_dict,
        }

    run_crossq(cfg, output_path, args.device, args.with_val)
