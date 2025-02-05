from argparse import ArgumentParser
from leap_c.rl.sac import SACTrainer, SACBaseConfig
from leap_c.examples.mountain_car.task import MountainCarTask


def main(output_path: str | None, device: str, seed: int):
    if output_path is None:
        # create a path based on the current time
        from datetime import datetime
        now = datetime.now()
        output_path = f"output/sac_mountain_car/{now.strftime('%Y%m%d_%H%M%S')}"

    cfg = SACBaseConfig(seed=seed)
    cfg.train.start = 10000
    task = MountainCarTask()

    trainer = SACTrainer(task, cfg, output_path, device)
    trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args.output_path, args.device, args.seed)

