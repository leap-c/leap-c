from argparse import ArgumentParser
from dataclasses import asdict
import datetime
from pathlib import Path
import yaml

import leap_c.examples  # noqa: F401
import leap_c.rl  # noqa: F401
from leap_c.registry import create_task, create_default_cfg, create_trainer
from leap_c.trainer import BaseConfig


def print_inputs(
    trainer_name: str,
    task_name: str,
    output_path: Path,
    device: str,
    seed: int,
    cfg: BaseConfig,
):
    print("Running RL with the following inputs:")
    print(f"trainer_name:\t{trainer_name}")
    print(f"task_name:\t{task_name}")
    print(f"output_path:\t{output_path}")
    print(f"device:  \t{device}")

    # Report on the configuration
    print("\nConfiguration:")
    print(yaml.dump(asdict(cfg), default_flow_style=False))


def default_output_path() -> Path:
    # derive output path from date and time
    now = datetime.datetime.now()
    return Path(f"output/{now.strftime('%Y_%m_%d/%H_%M_%S')}")


def main(
    trainer_name: str, task_name: str, output_path: Path | None, device: str, seed: int
):
    if output_path is None:
        output_path = default_output_path()
        if output_path.exists():
            raise ValueError(f"Output path {output_path} already exists")

    task = create_task(task_name)
    cfg = create_default_cfg(trainer_name)
    cfg.seed = seed
    cfg.val.num_render_rollouts = 1
    cfg.val.num_rollouts = 1
    cfg.val.interval = 2000
    cfg.val.deterministic = True
    # cfg.sac.update_freq = 20  # type: ignore

    print_inputs(
        trainer_name=trainer_name,
        task_name=task_name,
        output_path=output_path,
        device=device,
        seed=seed,
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

    main(args.trainer, args.task, args.output_path, args.device, args.seed)
