"""Main script to run experiments."""

import datetime
from argparse import ArgumentParser
from pathlib import Path

from leap_c.rl.sac_fop import SacFopBaseConfig
from leap_c.run import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = SacFopBaseConfig()
    output_path = Path(
        f"output/reacher/sac_{args.seed}_{datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=2))).strftime('%Y%m%d%H%M%S')}"
    )

    main(
        trainer_name="sac_fop",
        task_name="reacher",
        cfg=cfg,
        output_path=output_path,
        device=args.device,
    )
