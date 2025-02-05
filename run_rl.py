from argparse import ArgumentParser
from pathlib import Path

import leap_c.examples  # noqa: F401
import leap_c.rl  # noqa: F401
from leap_c.registry import create_task, create_default_cfg, create_trainer


def default_output_path(trainer_name: str, task_name: str, seed: int) -> Path:
    return Path(f"output/{trainer_name}/{task_name}/{seed}")


def main(
    trainer_name: str, task_name: str, output_path: Path | None, device: str, seed: int
):

    if output_path is None:
        output_path = default_output_path(trainer_name, task_name, seed)
        if output_path.exists():
            raise ValueError(f"Output path {output_path} already exists")

    task = create_task(task_name)
    cfg = create_default_cfg(trainer_name)
    cfg.seed = seed
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

