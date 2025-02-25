from argparse import ArgumentParser
from enum import Enum

from run import create_cfg, default_output_path, main


class Experiment(Enum):
    POINTMASS_PLAINRL = 0
    POINTMASS_FOP_PREVIOUS = 1
    POINTMASS_FOP_DEFAULTINIT = 2
    POINTMASS_FOP_RELOAD = 3
    POINTMASS_FOP_NN = 4


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    experiment = Experiment(args.experiment)
    device = args.device
    seed = args.seed

    if "FOP" in experiment.name:
        trainer_name = "sac_fop"
    elif "ZO" in experiment.name:
        trainer_name = "sac_zo"
    elif "PLAINRL" in experiment.name:
        trainer_name = "sac"
    else:
        raise ValueError("Unknown trainer")

    if "POINTMASS" in experiment.name:
        task_name = "point_mass"
    else:
        raise ValueError("Unknown task")

    if "PREVIOUS" in experiment.name:
        pass
    else:
        if "PLAINRL" not in experiment.name:
            raise ValueError("Unknown initialization")

    cfg = create_cfg(trainer_name="sac_fop", seed=seed)
    output_path = str.join(
        "_",
        [
            default_output_path(
                trainer_name=trainer_name,
                task_name=task_name,
                seed=seed,  # type:ignore
            ),
            experiment.name.split("_")[-1],
        ],
    )  # type:ignore
    main(
        trainer_name=trainer_name,
        task_name=task_name,
        cfg=cfg,
        output_path=output_path,  # type:ignore
        device=device,
    )
