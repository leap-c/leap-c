"""Module for running experiments."""
from argparse import ArgumentParser
import datetime
from pathlib import Path

import leap_c.examples  # noqa: F401
from leap_c.trainer import Trainer
from leap_c.utils.cfg import cfg_as_python
from leap_c.utils.git import log_git_hash_and_diff


def default_output_path(seed: int, tags: dict | None = None) -> Path:
    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d")
    time = now.strftime("%H_%M_%S")

    tag_str = sum((f"_{k}_{v}" for k, v in tags.items()), "") if tags else ""

    return Path(f"output/{date}/{time}{tag_str}_seed_{seed}")


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

    print("Starting a trainer with:")
    print(f"\noutput_path: \n{output_path}")
    print("\nConfiguration:")
    print(cfg_as_python(cfg))
    print("\n")

    if continue_run and (output_path / "ckpts").exists():
        trainer.load(output_path)

    # store git hash and diff
    log_git_hash_and_diff(output_path / "git.txt") 


def create_parser() -> ArgumentParser:
    """Create an argument parser for a script.

    Returns:
        An ArgumentParser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    return parser
