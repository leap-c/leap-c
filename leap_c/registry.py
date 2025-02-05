from typing import Dict, Type

from pathlib import Path
from leap_c.task import Task
from leap_c.trainer import BaseConfig, Trainer


TASK_REGISTRY: Dict[str, Type[Task]] = {}
DEFAULT_CFG_REGISTRY: Dict[str, BaseConfig] = {}
TRAINER_REGISTRY: Dict[str, Type[Trainer]] = {}


def register_task(name: str):
    def decorator(cls):
        if name in TASK_REGISTRY:
            raise ValueError(f"Cannot register duplicate task ({name})")
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


def register_trainer(name: str, default_cfg: BaseConfig):
    def decorator(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError(f"Cannot register duplicate trainer ({name})")
        TRAINER_REGISTRY[name] = cls
        DEFAULT_CFG_REGISTRY[name] = default_cfg
        return cls

    return decorator


def create_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}")
    return TASK_REGISTRY[name]()  # type: ignore


def create_default_cfg(name: str) -> BaseConfig:
    if name not in DEFAULT_CFG_REGISTRY:
        raise ValueError(f"Unknown trainer: {name}")
    return DEFAULT_CFG_REGISTRY[name]  # type: ignore


def create_trainer(
    name: str,
    task: Task,
    output_path: str | Path,
    device: str,
    cfg: None | BaseConfig = None,
) -> Trainer:
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown trainer: {name}")
    return TRAINER_REGISTRY[name](task, output_path, device, cfg)  # type: ignore
