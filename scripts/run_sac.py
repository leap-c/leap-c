"""Main script to run experiments."""
from dataclasses import dataclass
from pathlib import Path

from leap_c.run import init_run, create_parser, default_output_path
from leap_c.torch.rl.sac import SacBaseConfig, SacTrainer


@dataclass
class RunSacConfig:
    """Configuration for running SAC experiments."""

    env: str = "HalfCheetah-v4"
    device: str = "cuda"  # or 'cpu'
    trainer: SacBaseConfig = SacBaseConfig()


def run_sac(
    output_path: str | Path, seed: int = 0, env: str = "HalfCheetah-v4", device: str = "cuda"
) -> float:
    cfg = RunSacConfig(env=env, device=device)
    cfg.env = env
    cfg.trainer.seed = seed

    trainer = SacTrainer(
        env=env,
        output_path=output_path,
        device=args.device,
        cfg=cfg,
    )
    init_run(trainer, cfg, output_path)

    return trainer.run()


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    args = parser.parse_args()

    output_path = default_output_path(seed=args.seed, tags={"trainer": "sac"})
