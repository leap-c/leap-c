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
    cfg.train.steps = 100 * 4 * 512
    cfg.val.interval = 10 * 4 * 512

    output_path = Path(f"output/half_cheetah/ppo")

    main("ppo", "half_cheetah", cfg, output_path, "cuda")
