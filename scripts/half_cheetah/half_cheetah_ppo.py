from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from leap_c.nn.mlp import MlpConfig
from leap_c.run import main
from leap_c.rl.ppo import PpoBaseConfig

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-team", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    cfg = PpoBaseConfig()
    cfg.ppo.actor_mlp = MlpConfig(
        hidden_dims=(64, 64),
        activation="tanh",
        weight_init="orthogonal",
    )
    cfg.ppo.critic_mlp = MlpConfig(
        hidden_dims=(64, 64),
        activation="tanh",
        weight_init="orthogonal",
    )
    cfg.train.steps = 1_000_000
    cfg.train.num_envs = 1
    cfg.train.vectorized = True
    cfg.val.interval = 50_000
    cfg.ppo.lr_q = 3e-4
    cfg.ppo.lr_pi = 3e-4
    cfg.ppo.update_epochs = 10
    cfg.ppo.num_steps = 2048
    cfg.ppo.num_mini_batches = 32
    cfg.ppo.clipping_epsilon = 0.2
    cfg.ppo.l_vf_weight = 0.25
    cfg.ppo.l_ent_weight = 0.0
    cfg.ppo.gamma = 0.99
    cfg.ppo.gae_lambda = 0.95
    cfg.seed = args.seed

    if args.wandb_team is not None and args.wandb_project is not None:
        cfg.log.wandb_logger = True
        cfg.log.wandb_init_kwargs = {
            "entity": args.wandb_team,
            "project": args.wandb_project,
            "name": f"half_cheetah/ppo_{args.seed}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "config": {
                **cfg.ppo.__dict__,
                **cfg.train.__dict__,
                "seed": args.seed
            }
        }

    output_path = Path(f"output/half_cheetah/ppo_{args.seed}_{datetime.now().strftime('%Y%m%d%H%M%S')}")

    main("ppo", "half_cheetah", cfg, output_path, args.device)
