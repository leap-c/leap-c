"""Module for running experiments."""

import datetime
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import leap_c
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
    """Returns the default path to store experiment outputs, such as logs. Based on the provided
    seed and tags, a directory name is created.

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

    if continue_run and (output_path / "ckpts").exists():
        trainer.load(output_path)

    # store git hash and diff
    if leap_c.__file__ is not None:
        module_root = Path(leap_c.__file__).parent.parent
    else:
        module_root = Path(leap_c.__path__[0]).parent
    log_git_hash_and_diff(output_path / "git.txt", module_root)
