from pathlib import Path

from leap_c.nn.mlp import MlpConfig
from leap_c.run import main
from leap_c.rl.ppo import PpoBaseConfig

if __name__ == "__main__":
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
    cfg.train.steps = 512 * 2048
    cfg.train.num_envs = 8
    cfg.train.vectorized = True
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

    output_path = Path(f"output/half_cheetah/ppo")

    main("ppo", "half_cheetah", cfg, output_path, "cuda")
