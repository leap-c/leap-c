from argparse import ArgumentParser
from pathlib import Path
import re

from leap_c.task import create_task
from leap_c.examples.mujoco.task import HalfCheetahTask
from leap_c.rl.sac import SACBaseConfig, SACTrainer


def default_output_path(alg_cls: type, task_cls: type, seed: int) -> Path:
    # convert to snake case
    alg_name = re.sub("([A-Z])", "_\\1", alg_cls.__name__).lower().strip("_")
    task_name = re.sub("([A-Z])", "_\\1", task_cls.__name__).lower().strip("_")

    return Path(f"output/{alg_name}/{task_name}_{seed}")


def main(output_path: Path | None, device: str, seed: int):
    task = create_task("half_cheetah")

    if output_path is None:
        output_path = default_output_path(SACTrainer, task, seed)

    cfg = SACBaseConfig(seed=seed)
    task = HalfCheetahTask()

    trainer = SACTrainer(task, cfg, output_path, device)
    trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="HalfCheetahTask")
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args.output_path, args.device, args.seed)

