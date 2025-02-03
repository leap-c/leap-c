from dataclasses import dataclass, field
import torch.nn as nn
from leap_c.nn.gaussian import Gaussian
from leap_c.rl.replay_buffer import ReplayBuffer
from leap_c.trainer import Trainer, BaseConfig, TrainConfig, TrainerState, ValConfig, LogConfig
from leap_c.nn.mlp import MLP, MLPConfig
from leap_c.task import Task
import torch
import gymnasium as gym
from functools import partial


@dataclass(kw_only=True)
class SACAlgorithmConfig:
    """Contains the necessary information for a SACTrainer.

    Attributes:
        batch_size: The batch size for training.
        buffer_size: The size of the replay buffer.
        gamma: The discount factor.
        tau: The soft update factor.
        soft_update_freq: The frequency of soft updates.
        lr_q: The learning rate for the Q networks.
        lr_pi: The learning rate for the policy network.
        target_entropy: The target entropy for the policy network.
        lr_alpha: The learning rate for the temperature parameter.
    """
    critic_mlp: MLPConfig = field(default_factory=MLPConfig)
    actor_mlp: MLPConfig = field(default_factory=MLPConfig)
    batch_size: int = 32
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005
    soft_update_freq: int = 1
    lr_q: float = 3e-4
    lr_pi: float = 3e-4
    target_entropy: float = -2.0
    lr_alpha: float = 3e-4
    num_critics: int = 2


@dataclass(kw_only=True)
class SACBaseConfig(BaseConfig):
    """Contains the necessary information for a Trainer.

    Attributes:
        sac: The SAC algorithm configuration.
        train: The training configuration.
        val: The validation configuration.
        log: The logging configuration.
        seed: The seed for the trainer.
    """

    sac: SACAlgorithmConfig
    train: TrainConfig
    val: ValConfig
    log: LogConfig
    seed: int


class SACCritic(nn.Module):
    def __init__(self, extractor_factory, mlp_cfg: MLPConfig, num_critics: int):
        super().__init__()

        self.extractor = nn.ModuleList([extractor_factory(self.train_env) for _ in range(num_critics)])
        self.mlp = nn.ModuleList([
            MLP(
                input_dims=[qe.output_size, self.train_env.action_space.shape[0]],  # type: ignore
                output_dims=1,
                mlp_cfg=mlp_cfg,
            )
            for qe in self.extractor
        ])

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        return [mlp(qe(x), a) for qe, mlp in zip(self.extractor, self.mlp)]


class SACActor(nn.Module):
    def __init__(self, extractor_factory, mlp_cfg: MLPConfig, action_space: gym.spaces.Box):
        super().__init__()

        self.extractor = extractor_factory(self.train_env)
        self.mlp = MLP(
            input_dims=self.extractor.output_size,
            output_dims=(action_space.shape[0], action_space.shape[0]),  # type: ignore
            mlp_cfg=mlp_cfg,
        )
        self.gaussian = Gaussian(action_space)

    def forward(self, x: torch.Tensor, deterministic=False):
        mean, std = self.mlp(x)
        return self.gaussian(mean, std, deterministic=deterministic)


class SAC(Trainer):
    cfg: SACBaseConfig

    def __init__(self, task: Task, cfg: SACBaseConfig, output_path: str, device: str):
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            task: The task to be solved by the trainer.
            cfg: The configuration for the trainer.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
        """
        super().__init__(task, cfg, output_path, device)

        extractor_factory = partial(task.extractor_factory, self.train_env)

        self.q = SACCritic(extractor_factory, cfg.sac.critic_mlp, cfg.sac.num_critics).to(device)
        self.q_target = SACCritic(extractor_factory, cfg.sac.critic_mlp, cfg.sac.num_critics).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.sac.lr_q)

        self.pi = SACActor(extractor_factory, cfg.sac.actor_mlp, self.train_env.action_space).to(device)  # type: ignore
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.sac.lr_pi)

        self.alpha = self.register_buffer("alpha", torch.tensor(1.0))
        self.alpha_optim = torch.optim.Adam([self.alpha], lr=cfg.sac.lr_alpha)

        self.buffer = ReplayBuffer(cfg.sac.buffer_size, device=device)


    def train_step(self) -> dict[str, float]:

        if self.state.is_terminated or self.state.is_truncated:
            obs, _ = self.train_env.reset()
            self.state.is_terminated = False
            self.state.is_truncated = False

        action = self.act(obs)
        obs_prime, reward, is_terminated, is_truncated, _ = self.train_env.step(action)

        self.buffer.put((obs, action, reward, obs_prime, is_terminated))

        obs = obs_prime

            if self.state.step < self.cfg.train.start:
                continue

            # sample batch
            o, a, r, o_prime, te = self.buffer.sample(self.cfg.sac.batch_size)

            # sample action
            a_pi, log_p = self.pi(o)

            # update temperature
            # alpha_loss = -torch.mean(self.alpha * (log_p + self.cfg.sac.target_entropy).detach())

            # update critic
            with torch.no_grad():
                # TODO (Jasper): Should this be deterministic?
                a_pi_prime = self.pi(o_prime)
                q_target = torch.cat(self.q_target(o_prime, a_pi_prime), dim=1)
                q_target = torch.min(q_target, dim=1, keepdim=True).values
                # add entropy
                q_target = q_target - self.alpha * log_p.reshape(-1, 1)

                target = r + self.cfg.sac.gamma * (1 - te) * q_target

            q = torch.cat(self.q(o, a), dim=1)
            q_loss = torch.mean((q - target).pow(2))

            self.q_optim.zero_grad()
            q_loss.backward()
            self.q_optim.step()

            # update actor
            q = torch.cat(self.q(o, a_pi), dim=1)
            min_q = torch.min(q, dim=1)
            pi_loss = (self.alpha * log_p - min_q).mean()

            self.pi_optim.zero_grad()
            pi_loss.backward()
            self.pi_optim.step()

    def act(self, obs, deterministic: bool = False) -> torch.Tensor:
        return self.pi(obs, deterministic=deterministic)

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.q_optim, self.pi_optim, self.alpha_optim]

