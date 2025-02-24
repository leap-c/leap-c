from enum import Enum

from run import create_cfg, default_output_path, main


class Experiment(Enum):
    POINTMASS_PLAINRL = 0
    POINTMASS_FOU_SHIFT = 1
    POINTMASS_FOU_DEFAULTINIT = 2
    POINTMASS_FOU_RELOAD = 3
    POINTMASS_FOU_NN = 4


if __name__ == "__main__":
    experiment = Experiment.POINTMASS_FOU_SHIFT
    device = "cuda:0"
    seed = 1337

    if "FOU" in experiment.name:
        trainer_name = "sac_fou"
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

    cfg = create_cfg(trainer_name="sac_fou", seed=seed)
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
