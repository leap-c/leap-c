"""Main script to run experiments."""
from dataclasses import dataclass, field
from pathlib import Path

from leap_c.examples.cartpole.controller import CartPoleController
from leap_c.examples.cartpole.env import CartPoleEnv
from leap_c.run import init_run, create_parser, default_output_path
from leap_c.torch.rl.sac_fop import SacFopTrainer, SacFopTrainerConfig


@dataclass
class RunSacConfig:
    """Configuration for running SAC experiments."""

    device: str = "cuda"  # or 'cpu'
    trainer: SacFopTrainerConfig = field(default_factory=SacFopTrainerConfig)


def run_sac(
    output_path: str | Path, seed: int = 0, device: str = "cuda"
) -> float:
    cfg = RunSacConfig(device=device)
    cfg.trainer.seed = seed

    trainer = SacFopTrainer(
        val_env=CartPoleEnv(render_mode="rgb_array"),
        train_env=CartPoleEnv(),
        output_path=output_path,
        device=args.device,
        cfg=cfg.trainer,
        controller=CartPoleController()
    )
    init_run(trainer, cfg, output_path)

    return trainer.run()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    output_path = default_output_path(seed=args.seed, tags={"trainer": "sac"})

    run_sac(output_path, seed=args.seed, device=args.device)
