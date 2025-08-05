"""Main script to run experiments."""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

from leap_c.run import init_run, default_output_path
from leap_c.examples import create_env
from leap_c.torch.nn.extractor import ExtractorName
from leap_c.torch.rl.sac import SacTrainer, SacTrainerConfig


@dataclass
class RunSacConfig:
    """Configuration for running SAC experiments."""

    env: str = "cartpole"
    trainer: SacTrainerConfig = field(default_factory=SacTrainerConfig)
    extractor: ExtractorName = "identity"  # for hvac use "scaling"


def create_cfg() -> RunSacConfig:
    """Return the default configuration for running SAC experiments."""
    # ---- Configuration ----
    cfg = RunSacConfig()
    cfg.env = "cartpole"
    cfg.extractor = "identity"  # for hvac use "scaling"

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = 0
    cfg.trainer.train_steps = 1000000
    cfg.trainer.train_start = 0
    cfg.trainer.val_interval = 10000
    cfg.trainer.val_num_rollouts = 20
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 0
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_render_deterministic = True
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = "best"
    cfg.trainer.batch_size = 64
    cfg.trainer.buffer_size = 1000000
    cfg.trainer.gamma = 0.99
    cfg.trainer.tau = 0.005
    cfg.trainer.soft_update_freq = 1
    cfg.trainer.lr_q = 0.001
    cfg.trainer.lr_pi = 0.001
    cfg.trainer.lr_alpha = 0.001
    cfg.trainer.init_alpha = 0.02
    cfg.trainer.target_entropy = None
    cfg.trainer.entropy_reward_bonus = True
    cfg.trainer.num_critics = 2
    cfg.trainer.report_loss_freq = 100
    cfg.trainer.update_freq = 4

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 1000
    cfg.trainer.log.window = 10000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = True
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}

    # ---- Section: cfg.trainer.critic_mlp ----
    cfg.trainer.critic_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.critic_mlp.activation = "relu"
    cfg.trainer.critic_mlp.weight_init = "orthogonal"

    # ---- Section: cfg.trainer.actor_mlp ----
    cfg.trainer.actor_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.actor_mlp.activation = "relu"
    cfg.trainer.actor_mlp.weight_init = "orthogonal"

    return cfg


def run_sac(cfg: RunSacConfig, output_path: str | Path, device: str = "cuda") -> float:
    trainer = SacTrainer(
        cfg=cfg.trainer,
        val_env=create_env(cfg.env, render_mode="rgb_array"),
        output_path=output_path,
        device=device,
        train_env=create_env(cfg.env),
        extractor_cls=cfg.extractor,
    )
    init_run(trainer, cfg, output_path)

    return trainer.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="cartpole")
    args = parser.parse_args()

    output_path = default_output_path(seed=args.seed, tags=["sac", args.env])

    cfg = create_cfg()
    cfg.trainer.seed = args.seed
    cfg.env = args.env

    run_sac(cfg, output_path, args.device)
