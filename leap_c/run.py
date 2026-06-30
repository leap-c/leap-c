"""Module for running experiments."""

import datetime
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, get_args

import torch
from yaml import safe_dump

import leap_c
from leap_c.examples import ExampleControllerName, ExampleEnvName
from leap_c.trainer import CtxType, Trainer, TrainerConfigType
from leap_c.utils.cfg import cfg_as_python
from leap_c.utils.git import log_git_hash_and_diff

OUTPUT_DIR = Path("output")


def default_name(seed: int, tags: Iterable[Any] | None = None) -> str:
    """Generate a default name for the run based on the seed and optional tags."""
    if tags is None:
        return f"seed_{seed}"
    return "_".join(map(str, tags)) + f"_seed_{seed}"


def default_output_path(seed: int, tags: Iterable[Any] | None = None) -> Path:
    """Return the default path to store experiment outputs, such as logs.

    Based on the provided seed and tags, a directory name is created.

    Args:
        seed: The RNG seed used for the experiment.
        tags: Optional iterable of tags to include in the directory name.

    Returns:
        A `Path` to the default output directory.
    """
    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d")
    time = now.strftime("%H_%M_%S")
    return OUTPUT_DIR / date / f"{time}_{default_name(seed, tags)}"


def default_controller_code_path() -> Path:
    """Returns the default path to store compiled controller code.

    Returns:
        A `Path` to the default directory for compiled controller code.
    """
    return OUTPUT_DIR / "controller_code"


def add_common_args(
    parser: ArgumentParser,
    *,
    has_controller: bool,
    has_with_val: bool,
    wandb_group_default: str,
) -> dict[str, Any]:
    """Add common command-line arguments shared across run scripts.

    Args:
        parser: The argument parser to add arguments to.
        has_controller: Whether to add controller-related args (--controller, --reuse-code*).
        has_with_val: Whether to add validation-related args (--with-val, --ckpt-modus).
        wandb_group_default: Default value for --wandb-group.

    Returns:
        A dict mapping group names ("run", "eval", "wandb") to the argument groups,
        so the caller can add script-specific args to the same groups.
    """
    run_group = parser.add_argument_group("Run settings")
    run_group.add_argument(
        "--output-path", type=Path, default=None, help="Path to outputs (e.g., logs)."
    )
    run_group.add_argument(
        "--device", type=validate_torch_device_arg, default="cpu", help="Device to run on."
    )
    run_group.add_argument(
        "--dtype",
        type=validate_torch_dtype_arg,
        default="float32",
        help="Data type to use during training and evaluation.",
    )
    run_group.add_argument("--seed", type=int, default=0, help="RNG seed.")

    if has_controller:
        run_group.add_argument(
            "-r",
            "--reuse-code",
            action="store_true",
            help="Reuse compiled code. The first time this is run, it will compile the code.",
        )
        run_group.add_argument(
            "--reuse-code-dir", type=Path, default=None, help="Directory for compiled code."
        )

    eval_group = parser.add_argument_group("Train and eval")
    eval_group.add_argument(
        "--env",
        type=str,
        choices=get_args(ExampleEnvName),
        default="cartpole",
        help="Environment to train on.",
    )

    if has_controller:
        eval_group.add_argument(
            "--controller",
            type=str,
            choices=get_args(ExampleControllerName),
            default=None,
            help="MPC controller to use as actor. If not provided, it is taken from `--env`.",
        )

    if has_with_val:
        eval_group.add_argument(
            "--with-val", action="store_true", help="Enables validation environment."
        )
        eval_group.add_argument(
            "--ckpt-modus",
            type=str,
            default=None,
            choices=["none", "last", "all", "best"],
            help="Checkpoint mode. Defaults to 'best' with --with-val, 'last' otherwise.",
        )

    wandb_group = parser.add_argument_group("W&B logging")
    wandb_group.add_argument("--use-wandb", action="store_true", help="Whether to use W&B logging.")
    wandb_group.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name.")
    wandb_group.add_argument(
        "--wandb-project", type=str, default="leap-c", help="W&B project name."
    )
    wandb_group.add_argument(
        "--wandb-group", type=str, default=wandb_group_default, help="W&B group name."
    )

    return {"run": run_group, "eval": eval_group, "wandb": wandb_group}


def default_ckpt_modus(args: Namespace) -> Literal["best", "last", "all", "none"]:
    """Return the checkpoint mode, defaulting based on --with-val.

    Args:
        args: Parsed argparse namespace with `ckpt_modus` and `with_val` attributes.

    Returns:
        The resolved checkpoint mode.
    """
    if args.ckpt_modus is not None:
        return args.ckpt_modus
    if args.with_val:
        return "best"
    return "last"


def resolve_output_path(args: Namespace, tags: Iterable[Any] | None = None) -> Path:
    """Resolve the output path, using a default if not provided.

    Args:
        args: Parsed argparse namespace with `output_path` and `seed` attributes.
        tags: Tags for the default output path name.

    Returns:
        The resolved output path.
    """
    if args.output_path is None:
        return default_output_path(seed=args.seed, tags=tags)
    return args.output_path


def resolve_reuse_code_dir(args: Namespace) -> Path | None:
    """Resolve the reuse code directory from parsed args.

    Args:
        args: Parsed argparse namespace with `reuse_code` and `reuse_code_dir` attributes.

    Returns:
        The resolved reuse code directory, or None if not reusing code.
    """
    if args.reuse_code and args.reuse_code_dir is None:
        return default_controller_code_path()
    if args.reuse_code_dir is not None:
        return args.reuse_code_dir
    return None


def setup_wandb(args: Namespace, cfg, tags: Iterable[Any] | None = None) -> None:
    """Set up W&B logging on the config if --use-wandb is set.

    Args:
        args: Parsed argparse namespace with wandb-related attributes.
        cfg: The run config dataclass with a `trainer.log` nested config.
        tags: Tags for the default run name.
    """
    if not args.use_wandb:
        return
    cfg.trainer.log.wandb_logger = True
    cfg.trainer.log.wandb_init_kwargs = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "group": args.wandb_group,
        "name": default_name(args.seed, tags=tags),
        "config": asdict(cfg),
    }


def init_run(trainer: Trainer[TrainerConfigType, CtxType], cfg, output_path: str | Path) -> None:
    """Init function to run experiments.

    If the output path already exists, the run will continue from the last checkpoint.

    Args:
        trainer: The trainer for the experiment.
        cfg: The configuration that was used to create the experiment.
        output_path: Path to save output to.

    Returns:
        The final score of the trainer.
    """
    output_path = Path(output_path)
    continue_run = output_path.exists()

    trainer_name = type(trainer).__name__

    print(f"Starting {trainer_name} run")
    print(f"\nOutput path: \n{output_path}")
    print("\nConfiguration:")
    print(cfg_as_python(cfg))
    print("\n")

    # persist the run config as yaml for potential future loading
    with open(output_path / "run_config.yaml", "w") as f:
        safe_dump(asdict(cfg), f)

    if continue_run and (output_path / "ckpts").exists():
        trainer.load(output_path)

    # store git hash and diff
    if leap_c.__file__ is not None:
        module_root = Path(leap_c.__file__).parent.parent
    else:
        module_root = Path(leap_c.__path__[0]).parent
    log_git_hash_and_diff(output_path / "git.txt", module_root)


def validate_torch_device_arg(arg: str) -> torch.device:
    """Validate the provided string argument as a valid torch device.

    Args:
        arg: String representation of the torch device (e.g., "cpu", "cuda:0", etc.).

    Returns:
        The corresponding torch device object for the provided string.

    Raises:
        ArgumentTypeError: If the provided value is not a valid torch device.
    """
    try:
        return torch.device(arg.lower())
    except RuntimeError as e:
        devices = []
        if torch.cpu.is_available():
            devices.append("`cpu`")
        if torch.cuda.is_available():
            devices.extend(f"`cuda:{i}`" for i in range(torch.cuda.device_count()))
        devices_str = ", ".join(devices) if devices else "no available devices"
        raise ArgumentTypeError(f"`{arg}` is not a valid torch device: {devices_str}.") from e


def validate_torch_dtype_arg(arg: str) -> torch.dtype:
    """Validate the provided string argument as a valid torch data type.

    Args:
        arg: String representation of the torch dtype (e.g., "float32", "float64", etc.).

    Returns:
        The corresponding torch dtype object for the provided string.

    Raises:
        ArgumentTypeError: If the provided value is not a valid torch type or is not floating.
    """
    result = getattr(torch, arg.lower(), None)
    if not isinstance(result, torch.dtype) or not result.is_floating_point:
        valid_dtypes = ", ".join(
            f"`{name}`"
            for name, dtype in torch.__dict__.items()
            if isinstance(dtype, torch.dtype) and dtype.is_floating_point
        )
        raise ArgumentTypeError(f"`{arg}` is not a valid torch dtype: {valid_dtypes}.")
    return result
