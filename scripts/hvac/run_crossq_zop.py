"""Main script to run CrossQ-ZOP experiments."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, get_args

from leap_c.examples import ExampleControllerName, ExampleEnvName, create_controller, create_env
from leap_c.run import (
    default_controller_code_path,
    default_name,
    default_output_path,
    init_run,
    validate_torch_device_arg,
)
from leap_c.torch.nn.extractor import ExtractorName
from leap_c.torch.rl.crossq_zop import CrossQZop, CrossQZopConfig


@dataclass
class RunCrossQZopConfig:
    """Configuration for running CrossQ-ZOP experiments."""

    env: ExampleEnvName = "hvac"
    controller: ExampleControllerName = "hvac"
    trainer: CrossQZopConfig = field(default_factory=CrossQZopConfig)
    extractor: ExtractorName = "hvac"


def create_cfg(
    env: ExampleEnvName,
    controller: ExampleControllerName,
    seed: int,
    ckpt_modus: Literal["best", "last", "all", "none"] = "last",
) -> RunCrossQZopConfig:
    # ---- Configuration ----
    cfg = RunCrossQZopConfig()
    cfg.env = env
    cfg.controller = controller if controller is not None else env
    cfg.extractor = "hvac"

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
    cfg.trainer.batch_size = 512
    cfg.trainer.buffer_size = 1_000_000
    cfg.trainer.gamma = 0.999
    cfg.trainer.lr_q = 0.00005
    cfg.trainer.lr_pi = 0.00005
    cfg.trainer.lr_alpha = 0.00005
    cfg.trainer.init_alpha = 0.02
    cfg.trainer.target_entropy = None
    cfg.trainer.entropy_reward_bonus = True
    cfg.trainer.num_critics = 10
    cfg.trainer.update_freq = 4

    # cumulative metrics to reduce noise in performance estimate
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
    cfg.trainer.critic_mlp.hidden_dims = (512, 512, 512)
    cfg.trainer.critic_mlp.activation = "leaky_relu"
    cfg.trainer.critic_mlp.weight_init = "orthogonal"

    # ---- Section: cfg.trainer.actor ----
    cfg.trainer.actor.noise = "param"
    cfg.trainer.actor.extractor_name = cfg.extractor
    cfg.trainer.actor.distribution_name = "squashed_gaussian"
    cfg.trainer.actor.residual = True if env == "hvac" else False
    cfg.trainer.actor.entropy_correction = False

    # ---- Section: cfg.trainer.actor.mlp ----
    cfg.trainer.actor.mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.actor.mlp.activation = "leaky_relu"
    cfg.trainer.actor.mlp.weight_init = "orthogonal"

    return cfg


def run_crossq_zop(
    cfg: RunCrossQZopConfig,
    output_path: str | Path,
    device: str = "cuda",
    reuse_code_dir: Path | None = None,
    with_val: bool = False,
) -> float:
    """Run the CrossQ-ZOP trainer.

    Args:
        cfg: The configuration for running the controller.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint.
        device: The device to use.
        reuse_code_dir: The directory to reuse compiled code from, if any.
        with_val: Whether to use a validation environment.
    """
    val_env = create_env(cfg.env, render_mode="rgb_array") if with_val else None
    trainer = CrossQZop(
        cfg=cfg.trainer,
        val_env=val_env,
        output_path=output_path,
        device=device,
        train_env=create_env(cfg.env),
        controller=create_controller(cfg.controller, reuse_code_dir),
        extractor_cls=cfg.extractor,
    )
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training of CrossQ-ZOP agents.", formatter_class=ArgumentDefaultsHelpFormatter
    )
    group = parser.add_argument_group("Run settings")
    group.add_argument(
        "--output-path", type=Path, default=None, help="Path to outputs (e.g., logs)."
    )
    group.add_argument(
        "--device", type=validate_torch_device_arg, default="cpu", help="Device to run on."
    )
    group.add_argument("--seed", type=int, default=0, help="RNG seed.")
    group.add_argument(
        "-r",
        "--reuse-code",
        action="store_true",
        help="Reuse compiled code. The first time this is run, it will compile the code.",
    )
    group.add_argument(
        "--reuse-code-dir", type=Path, default=None, help="Directory for compiled code."
    )
    group = parser.add_argument_group("Train and eval")
    group.add_argument(
        "--env",
        type=str,
        choices=get_args(ExampleEnvName),
        default="hvac",
        help="Environment to train on.",
    )
    group.add_argument(
        "--controller",
        type=str,
        choices=get_args(ExampleControllerName),
        default=None,
        help="MPC controller to use as actor. If not provided, it is taken from `--env`.",
    )
    group.add_argument("--with-val", action="store_true", help="Enables validation environment.")
    group.add_argument(
        "--ckpt-modus",
        type=str,
        default=None,
        choices=["none", "last", "all", "best"],
        help="Checkpoint mode. Defaults to 'best' with --with-val, 'last' otherwise.",
    )
    group = parser.add_argument_group("W&B logging")
    group.add_argument("--use-wandb", action="store_true", help="Whether to use W&B logging.")
    group.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name.")
    group.add_argument("--wandb-project", type=str, default="leap-c", help="W&B project name.")
    group.add_argument("--wandb-group", type=str, default="CrossQ-ZOP", help="W&B group name.")
    args = parser.parse_args()

    if args.ckpt_modus is not None:
        ckpt_modus = args.ckpt_modus
    elif args.with_val:
        ckpt_modus = "best"
    else:
        ckpt_modus = "last"

    cfg = create_cfg(args.env, args.controller, args.seed, ckpt_modus)

    # Include variant in tags
    tags = ["crossq_zop", args.env, args.controller]

    if args.use_wandb:
        config_dict = asdict(cfg)
        cfg.trainer.log.wandb_logger = True
        cfg.trainer.log.wandb_init_kwargs = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
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

    run_crossq_zop(
        cfg=cfg,
        output_path=output_path,
        device=args.device,
        reuse_code_dir=reuse_code_dir,
        with_val=args.with_val,
    )
