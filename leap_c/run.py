"""Module for experiments and resurrecting trainers."""
from argparse import ArgumentParser
import datetime
from pathlib import Path

import leap_c.examples  # noqa: F401
from leap_c.registry import create_default_cfg, create_task, create_trainer
import leap_c.rl
from leap_c.trainer import BaseConfig
from leap_c.utils.cfg import cfg_as_python
from leap_c.utils.git import log_git_hash_and_diff


def print_inputs(
    trainer_name: str,
    task_name: str,
    output_path: Path | None,
    device: str,
    cfg: BaseConfig,
):
    print("Running RL with the following inputs:")
    print(f"trainer_name:\t{trainer_name}")
    print(f"task_name:\t{task_name}")
    print(f"output_path:\t{output_path}")
    print(f"device:  \t{device}")
    print(f"cfg.seed:    \t{cfg.seed}")

    # Report on the configuration
    print("\nConfiguration:")
    print(cfg_as_python(cfg))
    print("\n")


def default_output_path(trainer_name: str, task_name: str, seed: int) -> Path:
    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d")
    time = now.strftime("%H_%M_%S")

    return Path(f"output/{date}/{time}_{task_name}_{trainer_name}_seed_{seed}")


def main(
    trainer_name: str,
    task_name: str,
    cfg: BaseConfig,
    output_path: Path,
    device: str,
) -> float:
    """Main function to run experiments.

    If the output path already exists, the run continues from the last checkpoint.

    Args:
        trainer_name: Name of the trainer to use.
        task_name: Name of the task to use.
        cfg: The configuration to use.
        output_path: Path to save output to.
        device: Device to run on.

    Returns:
        The final score of the trainer.
    """
    continue_run = output_path.exists()

    task = create_task(task_name)

    print_inputs(
        trainer_name=trainer_name,
        task_name=task_name,
        output_path=output_path,
        device=device,
        cfg=cfg,
    )

    trainer = create_trainer(trainer_name, task, output_path, device, cfg)

    if continue_run and (output_path / "ckpts").exists():
        trainer.load(output_path)

    # store task name and trainer name for resurrection
    with open(output_path / "resurrect.txt", "w") as f:
        f.write(f"{trainer_name}\n")
        f.write(f"{task_name}\n")

    # store git hash and diff
    log_git_hash_and_diff(output_path / "git.txt") 

    return trainer.run()


def create_parser() -> ArgumentParser:
    """Create an argument parser for the script.

    Returns:
        An ArgumentParser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--trainer", type=str, default="sac_fop")
    parser.add_argument("--task", type=str, default="pendulum_swingup")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default="1_000_000")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    cfg = create_default_cfg(args.trainer)
    cfg.seed = args.seed
    cfg.train.steps = args.steps
    cfg.log.verbose = args.verbose

    if args.output_path is None:
        output_path = default_output_path(args.trainer, args.task, cfg.seed)
    else:
        output_path = args.output_path

    main(args.trainer, args.task, cfg, output_path, args.device)
