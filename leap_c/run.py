"""Main script to run experiments."""

import datetime
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import yaml

import leap_c.examples  # noqa: F401
import leap_c.rl  # noqa: F401
from leap_c.registry import create_default_cfg, create_task, create_trainer
from leap_c.trainer import BaseConfig


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
    print(yaml.dump(asdict(cfg), default_flow_style=False))


def default_output_path(trainer_name: str, task_name: str, seed: int) -> Path:
    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d")
    time = now.strftime("%H_%M_%S")

    return Path(f"output/{date}/{time}_{task_name}_{trainer_name}_seed_{seed}")


def create_cfg(trainer_name: str, seed: int) -> BaseConfig:
    cfg = create_default_cfg(trainer_name)
    cfg.seed = seed

    return cfg


def main(
    trainer_name: str,
    task_name: str,
    cfg: BaseConfig,
    output_path: Path,
    device: str,
):
    """Main function to run experiments.

    Args:
        trainer_name: Name of the trainer to use.
        task_name: Name of the task to use.
        cfg: The configuration to use.
        output_path: Path to save output to.
        device: Device to run on.
    """
    if output_path.exists():
        raise ValueError(f"Output path {output_path} already exists")

    task = create_task(task_name)

    print_inputs(
        trainer_name=trainer_name,
        task_name=task_name,
        output_path=output_path,
        device=device,
        cfg=cfg,
    )

    trainer = create_trainer(trainer_name, task, output_path, device, cfg)
    trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--trainer", type=str, default="sac")
    parser.add_argument("--task", type=str, default="half_cheetah")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = create_cfg(args.trainer, args.seed)

    if args.output_path is None:
        output_path = default_output_path(args.trainer, args.task, cfg.seed)
    else:
        output_path = args.output_path

    main(args.trainer, args.task, cfg, output_path, args.device)
