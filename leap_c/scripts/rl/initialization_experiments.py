from argparse import ArgumentParser
from enum import Enum

from leap_c.run import create_cfg, default_output_path, main

import wandb


class Experiment(Enum):
    POINTMASS_PLAINRL = 0
    POINTMASS_FOP_PREVIOUS = 1
    POINTMASS_FOP_PREVIOUS_PRIMAL = 2
    POINTMASS_FOP_CONST = 3
    POINTMASS_FOP_RELOAD = 4
    POINTMASS_FOP_RELOAD_PRIMAL = 5
    POINTMASS_FOP_LOADANDWRITEBACK = 6
    POINTMASS_FOP_NEIGHBOURS = 7
    # TODO: Dont forget to also try out only using primals


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("-t", "--wandbtags", action="append", type=str)
    args = parser.parse_args()

    experiment = Experiment(args.experiment)
    device = args.device
    seed = args.seed

    if "FOP" in experiment.name:
        if "PREVIOUS" in experiment.name:
            trainer_name = "sac_fop_previous"
        elif "RELOAD" in experiment.name:
            trainer_name = "sac_fop"
        elif "CONST" in experiment.name:
            trainer_name = "sac_fop_const"
        elif "LOADANDWRITEBACK" in experiment.name:
            trainer_name = "sac_fop_writeback"
        elif "NEIGHBOURS" in experiment.name:
            trainer_name = "sac_fop_neighbours"
        else:
            if "PLAINRL" not in experiment.name:
                raise ValueError("Unknown initialization")
    elif "ZO" in experiment.name:
        raise NotImplementedError()
    elif "PLAINRL" in experiment.name:
        trainer_name = "sac"
    else:
        raise ValueError("Unknown trainer")
    if "PRIMAL" in experiment.name:
        trainer_name += "_primal"  # type:ignore

    if "POINTMASS" in experiment.name:
        task_name = "point_mass_homo_center"
    else:
        raise ValueError("Unknown task")

    cfg = create_cfg(trainer_name=trainer_name, seed=seed)

    cfg.sac.update_freq = 1  # type:ignore

    wandb.login()
    cfg.log.wandb_logger = True
    cfg.log.wandb_name = "_".join((experiment.name, "seed", str(seed)))
    cfg.log.wandb_tags = args.wandbtags
    output_path = default_output_path(
        trainer_name=trainer_name,
        task_name=task_name,
        seed=seed,  # type:ignore
    )
    main(
        trainer_name=trainer_name,
        task_name=task_name,
        cfg=cfg,
        output_path=output_path,  # type:ignore
        device=device,
    )
