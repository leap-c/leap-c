"""Module for running experiments."""

import datetime
from pathlib import Path

import leap_c

# import leap_c.examples  # noqa: F401
from leap_c.trainer import Trainer
from leap_c.utils.cfg import cfg_as_python
from leap_c.utils.git import log_git_hash_and_diff


def default_output_path(seed: int, tags: list[str] | None = None) -> Path:
    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d")
    time = now.strftime("%H_%M_%S")

    tag_str = "".join([f"_{v}" for v in tags]) if tags else ""

    return Path(f"output/{date}/{time}{tag_str}_seed_{seed}")


def default_controller_code_path():
    return Path("output/controller_code")


def init_run(trainer: Trainer, cfg, output_path: str | Path):
    """Init function to run experiments.

    If the output path already exists, the run will continue from the last
    checkpoint.

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
