"""Main script to run experiments."""
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym

from leap_c.run import init_run, create_parser, default_output_path
from leap_c.torch.rl.sac import SacTrainer, SacTrainerConfig


@dataclass
class RunSacConfig:
    """Configuration for running SAC experiments."""

    env: str = "HalfCheetah-v4"
    device: str = "cuda"  # or 'cpu'
    trainer: SacTrainerConfig = field(default_factory=SacTrainerConfig)


def run_sac(
    output_path: str | Path, seed: int = 0, env: str = "HalfCheetah-v4", device: str = "cuda"
) -> float:
    cfg = RunSacConfig(env=env, device=device)
    cfg.env = env
    cfg.trainer.seed = seed

    trainer = SacTrainer(
        val_env=gym.make(cfg.env, render_mode="rgb_array"),
        train_env=gym.make(cfg.env),
        output_path=output_path,
        device=args.device,
        cfg=cfg.trainer,
    )
    init_run(trainer, cfg, output_path)

    return trainer.run()


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    args = parser.parse_args()

    output_path = default_output_path(seed=args.seed, tags={"trainer": "sac"})

    run_sac(output_path, seed=args.seed, env=args.env, device=args.device)
