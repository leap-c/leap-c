"""Provides functions to resurrect a trainer, task or cfg from a previous run."""
from pathlib import Path

import yaml

from leap_c.registry import create_default_cfg, create_task, create_trainer
from leap_c.task import Task
from leap_c.trainer import Trainer, BaseConfig
from leap_c.utils import update_dataclass_from_dict


def trainer_and_task_names(path: Path) -> tuple[str, str]:
    with open(path / "resurrect.txt", "r") as f:
        trainer_name = f.readline().strip()
        task_name = f.readline().strip()

    return trainer_name, task_name


def resurrect_cfg(path: Path) -> BaseConfig:
    """Resurrects the config from a previous run.

    Args:
        path: Path to the run directory.

    Returns:
        The config object.
    """
    trainer_name = trainer_and_task_names(path)[0]
    cfg = create_default_cfg(trainer_name)

    with open(path / "config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)

    update_dataclass_from_dict(cfg, cfg_dict)

    return cfg  # type: ignore


def resurrect_task(path: Path) -> Task:
    """Resurrects a task from a previous run.

    Args:
        path: Path to the run directory.

    Returns:
        The task object.
    """
    task_name = trainer_and_task_names(path)[1]
    task = create_task(task_name)

    return task


def resurrect_trainer(path: Path, device="cpu") -> Trainer:
    """Resurrects a trainer previous run.

    Args:
        path: Path to the run directory.
        device: Device to run on.

    Returns:
        The trainer object.
    """

    trainer_name, task_name = trainer_and_task_names(path)
    cfg = resurrect_cfg(path)

    task = create_task(task_name)
    trainer = create_trainer(trainer_name, task, path, device, cfg)

    trainer.load(path)

    return trainer

