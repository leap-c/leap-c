import datetime
import os
import random
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from leap_c.examples.linear_system import LinearSystemMPC, LinearSystemOcpEnv
from leap_c.mpc import MPC, MPCParameter
from leap_c.rl.replay_buffer import ReplayBuffer
from leap_c.rl.sac import (
    SACActor,
    SACConfig,
    SACQNet,
    SACTrainer,
)
from leap_c.logging import NumberLogger, WandbLogger
from leap_c.torch_modules import (
    FOUMPCNetwork,
    MeanStdMLP,
    TanhNormalActionNetwork,
    create_mlp,
    string_to_activation,
)
from leap_c.util import create_dir_if_not_exists, tensor_to_numpy


# TODO: Move to shared location probably to torch modules.
class LinearWithActivation(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = string_to_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.fc(x))


class StateEmbed(LinearWithActivation):
    def forward(self, obs: tuple[torch.Tensor, MPCParameter]) -> torch.Tensor:
        """NOTE: Ignores the parameters."""
        state, params = obs
        return self.activation(self.fc(state))


class Scenario(Enum):
    STANDARD_SAC = 0
    FO_U_SAC = 1


@dataclass(kw_only=True)
class LinearSystemSACConfig(SACConfig):
    # General
    name: str

    # env
    # TODO: this should probably be not changed and removed)
    max_time: float
    dt: float
    a_dim: int | None  # NOTE: Will be inferred from env and is None until then
    s_dim: int | None  # NOTE: Will be inferred from env and is None until then
    p_learnable_dim: (
        int | None
    )  # NOTE: Will be inferred from env and is None until then
    default_params: dict[str, np.ndarray]
    learnable_params: list[str]
    # MPC-specific
    accept_n_nonconvergences_per_batch: int
    N_horizon: int

    # Networks, i.e., mlps
    hidden_dims: list[int]
    q_embed_size: int  # should be half of hidden size
    activation: str
    log_tensors_bit_by_bit: bool


def create_qnet(config: LinearSystemSACConfig) -> SACQNet:
    if config.s_dim is None or config.a_dim is None:
        raise ValueError("s_dim and a_dim must be set before creating the qnet.")
    s_embed = StateEmbed(config.s_dim, config.q_embed_size, config.activation)
    a_embed = LinearWithActivation(config.a_dim, config.q_embed_size, config.activation)
    embed_to_q = create_mlp(
        input_dim=config.hidden_dims[0],
        output_dim=1,
        hidden_dims=config.hidden_dims[1:],
        activation=config.activation,
        use_activation_on_output=False,
    )
    return SACQNet(
        state_embed=s_embed,
        action_embed=a_embed,
        embed_to_q=embed_to_q,
        soft_update_factor=config.soft_update_factor,
    )


def create_actor(
    config: LinearSystemSACConfig, scenario: Scenario, mpc: MPC
) -> SACActor:
    if config.s_dim is None or config.a_dim is None or config.target_entropy is None:
        raise ValueError(
            "s_dim, a_dim and target_entropy must be set before creating the actor."
        )
    """NOTE: Mpc only needed for FO scenarios."""
    if scenario.value == Scenario.STANDARD_SAC.value:
        mlp = MeanStdMLP(
            s_dim=config.s_dim,
            mean_dim=config.a_dim,
            std_dim=config.a_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        )
        actor_net = TanhNormalActionNetwork(
            mean_std_module=mlp,
            minimal_std=config.minimal_std,
            log_tensors=config.log_tensors_bit_by_bit,
        )
        return SACActor(
            obs_to_action_module=actor_net,
            module_contains_mpc=False,
            init_entropy_scaling=config.init_entropy_scaling,
            target_entropy=config.target_entropy,
        )

    elif scenario.value == Scenario.FO_U_SAC.value:
        if config.p_learnable_dim is None:
            raise ValueError(
                "param_dim must be set before creating the first order actor."
            )
        mlp = MeanStdMLP(
            s_dim=config.s_dim,
            mean_dim=config.p_learnable_dim,
            std_dim=config.a_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        )
        actor_net = FOUMPCNetwork(
            mpc=mpc,
            param_mean_action_std_model=mlp,
            param_factor=config.param_factor,
            param_shift=config.param_shift,
            minimal_std=config.minimal_std,
            log_tensors=config.log_tensors_bit_by_bit,
        )
        return SACActor(
            obs_to_action_module=actor_net,
            module_contains_mpc=True,
            init_entropy_scaling=config.init_entropy_scaling,
            target_entropy=config.target_entropy,
            accept_n_nonconvergences_per_batch=config.accept_n_nonconvergences_per_batch,
            num_batch_dimensions=1,
        )
    else:
        raise ValueError(
            f"Unknown scenario, cannot create actor. Given scenario is {scenario.name}"
        )


def create_replay_buffer(config: LinearSystemSACConfig) -> ReplayBuffer:
    return ReplayBuffer(
        buffer_limit=config.replay_buffer_size,
        device=config.device,
    )


def create_mpc(
    config: LinearSystemSACConfig,
) -> LinearSystemMPC:
    return LinearSystemMPC(
        params=config.default_params,
        learnable_params=config.learnable_params,
        discount_factor=config.discount_factor,
        n_batch=config.batch_size,
        N_horizon=config.N_horizon,
    )


class LinearSystemSACTrainer(SACTrainer):
    # TODO: Could we put the code somewhere without subclassing?
    def act(
        self, obs: tuple[np.ndarray, MPCParameter], deterministic: bool = False
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state, params = obs
        state_tensor = torch.tensor(
            state, dtype=self.replay_buffer.obs_dtype, device=self.device
        )
        output = self.actor((state_tensor, params), deterministic=deterministic)
        return tensor_to_numpy(output[0]), output[-1]


def standard_config_dict(scenario: Scenario, savefile_directory_path: str) -> dict:
    """Contains standard values for the config, except for
    target_entropy: Will be inferred later as -a_dim, if not set before.
    a_dim: Will be inferred from env.
    s_dim: Will be inferred from env.
    p_learnable_dim: Will be inferred from env.
    """
    return dict(
        name=scenario.name,
        seed=1337,
        device="cpu",
        max_episodes=1000,
        training_steps_per_episode=20,
        dont_train_until_this_many_transitions=1000,
        val_interval=50,
        n_val_rollouts=5,
        save_interval=100,
        save_directory_path=savefile_directory_path,
        no_grad_during_rollout=False,
        crude_memory_debugging=False,
        # SACTrainer
        actor_lr=5 * 1e-4,
        critic_lr=5 * 1e-4,
        entropy_scaling_lr=1e-3,
        discount_factor=0.99,
        batch_size=64,
        # Replay Buffer
        replay_buffer_size=10000,
        # Actor
        init_entropy_scaling=0.01,
        target_entropy=None,  # NOTE: target_entropy = -a_dim will be inferred from env, but only when still set as None
        minimal_std=1e-3,  # Will be used in every TanhNormal to avoid collapse of the distribution
        param_factor=np.array([1.0, 1.0], dtype=np.float32),  # Only in Actor with MPC
        param_shift=np.array([0, 0], dtype=np.float32),  # Only in Actor with MPC
        # Critic
        soft_update_factor=1e-2,
        soft_update_frequency=1,
        # Environment
        render_mode=None,  # rgb_array or human
        video_directory_path=None,
        render_interval_exploration=50,
        render_interval_validation=5,
        max_time=10.0,
        dt=0.1,
        a_dim=None,  # NOTE: a_dim  Will be inferred from env
        s_dim=None,  # NOTE: s_dim  Will be inferred from env
        p_learnable_dim=None,  # NOTE: p_learnable_dim Will be inferred from env
        default_params={
            "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
            "B": np.array([[0.03125], [0.25]]),
            "Q": np.identity(2),
            "R": np.identity(1),
            "b": np.array([[0.0], [0.0]]),
            "f": np.array([[0.0], [0.0], [0.0]]),
            "V_0": np.array([1e-3]),
        },
        learnable_params=["b"],
        # MPC-specific
        accept_n_nonconvergences_per_batch=0,
        N_horizon=20,
        # Networks
        q_embed_size=64,  # should be half of hidden size
        hidden_dims=[128, 128],
        activation="leaky_relu",
        log_tensors_bit_by_bit=True,
        # Logging
        save_frequency=20,
        moving_average_width=20,
    )


def run_linear_system_sac(
    scenario: Scenario,
    savefile_directory_path: str,
    config_kwargs: dict,
    wandb_init_kwargs: dict | None = None,
) -> float:
    """Run SAC on the linear system environment.
    Parameters:
        scenario: The scenario to run.
        savefile_directory_path: The path to the directory where the models (networks) will be saved.
        config_kwargs: Kwargs that should be overwritten in the config.
        wandb_init_kwargs: Will use the wandb logger and initialize with these kwargs
            and the final config, if not None. NEEDS to contain project_name, run_name and mode.
    Returns:
        The last validation performance for testing purposes.
    """

    standard_config = standard_config_dict(
        scenario, savefile_directory_path=savefile_directory_path
    )
    config = LinearSystemSACConfig(**{**standard_config, **config_kwargs})
    mpc = create_mpc(config)
    env = LinearSystemOcpEnv(
        mpc, dt=config.dt, max_time=config.max_time, render_mode=config.render_mode
    )
    config.p_learnable_dim = env.p_learnable_space.shape[0]  # type:ignore

    a_space = env.action_space
    a_dim = a_space.shape[0]  # type:ignore
    s_space = env.state_space
    config.s_dim = s_space.shape[0]  # type:ignore
    config.a_dim = a_dim
    config.target_entropy = (
        -a_dim if config.target_entropy is None else config.target_entropy
    )
    actor = create_actor(config, scenario, mpc)

    if not config.q_embed_size * 2 == config.hidden_dims[0]:
        raise ValueError("The q_embed_size should be half of the hidden size.")
    critic1 = create_qnet(config)
    critic2 = create_qnet(config)
    critic1_target = create_qnet(config)
    critic2_target = create_qnet(config)
    buffer = create_replay_buffer(config)
    if wandb_init_kwargs is not None:
        logger = WandbLogger(
            config.save_directory_path,
            config.save_frequency,
            config.moving_average_width,
        )
        wandb_init_kwargs["config"] = asdict(config)
        project_name = wandb_init_kwargs.pop("project_name")
        run_name = wandb_init_kwargs.pop("run_name")
        mode = wandb_init_kwargs.pop("mode")
        logger.init(
            project_name=project_name, run_name=run_name, mode=mode, **wandb_init_kwargs
        )
    else:
        logger = NumberLogger(
            config.save_directory_path,
            config.save_frequency,
            config.moving_average_width,
        )
    trainer = LinearSystemSACTrainer(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        critic1_target=critic1_target,
        critic2_target=critic2_target,
        replay_buffer=buffer,
        logger=logger,
        config=config,
    )
    if config.render_mode is not None:
        config.video_directory_path = os.path.join(savefile_directory_path, "videos")

    return trainer.training_loop(env, config)


if __name__ == "__main__":
    scenario = Scenario.STANDARD_SAC
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 1337

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    savefile_directory_path = os.path.join(os.getcwd(), "output")
    create_dir_if_not_exists(savefile_directory_path)
    savefile_directory_path = os.path.join(savefile_directory_path, "linear_system_sac")
    create_dir_if_not_exists(savefile_directory_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    savefile_directory_path = os.path.join(
        savefile_directory_path, scenario.name + "_" + timestamp
    )
    video_path = os.path.join(savefile_directory_path, "videos")
    create_dir_if_not_exists(savefile_directory_path)

    # NOTE: You can uncomment this and use it in run_linear_system_sac, if you want to use the Wandblogger.
    wandb_init_kwargs = dict(
        project_name="Leap-C", run_name="test", mode="online", tags=["test"]
    )

    max_val = run_linear_system_sac(
        scenario,
        savefile_directory_path,
        dict(device=device, seed=seed, max_episodes=500, render_mode="rgb_array"),
        wandb_init_kwargs=wandb_init_kwargs,
    )
    print("Max validation score: ", max_val)
