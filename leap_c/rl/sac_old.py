import csv
import os
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from leap_c.nn.modules import CleanseAndReducePerSampleLoss
from leap_c.rl.replay_buffer import ReplayBuffer
from leap_c.rl.trainer import BaseTrainerConfig, Trainer
from leap_c.util import add_prefix_extend, tensor_to_numpy


class SACQNet(nn.Module):
    """Q-Network for SAC."""

    def __init__(
        self,
        state_embed: nn.Module,
        action_embed: nn.Module,
        embed_to_q: nn.Module,
        soft_update_factor: float,
    ):
        """
        Parameters:
            state_embed: Module that is used to embed the state.
            action_embed: Module that is used to embed the action.
            embed_to_q: Gets the concatenation of the two embeddings (in the last dimension!) as input and outputs a q value (also see forward).
                May also output a stats dictionary that will be logged.
            soft_update_factor: The soft update factor.
        """
        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed
        self.embed_to_q = embed_to_q

        self.soft_update_factor = soft_update_factor

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns the Q-Values for the given state and action and a dictionary containing statistics that should be logged.

        Parameters:
            state: The state for which the Q-Values should be determined.
            action: The action for which the Q-Values should be determined.
        """

        s_embed = self.state_embed(state)
        a_embed = self.action_embed(action)
        cat = torch.cat([s_embed, a_embed], dim=-1)
        out = self.embed_to_q(cat)
        if isinstance(out, tuple):
            return out
        else:
            return out, dict()

    def loss(
        self,
        target: torch.Tensor,
        mini_batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Calculates the loss with the given mini_batch and target, using a smooth L1 loss.

        Parameters:
            target: The target to train against.
            mini_batch: The mini_batch containing the states, actions, rewards, next_states and dones.

        Returns:
            The loss and a dictionary containing whatever else the internal module returns in the forward.
        """
        s, a, r, s_prime, done = mini_batch
        pred_qval, stats = self(s, a)
        loss = F.smooth_l1_loss(pred_qval, target)
        mean_loss = loss.mean()
        return mean_loss, stats

    def soft_update(self, net_target: "SACQNet"):
        """Update the target network parameters with the current network parameters using a soft update."""
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.soft_update_factor)
                + param.data * self.soft_update_factor
            )


class SACActor(nn.Module):
    obs_to_action_module: nn.Module
    log_entropy_scaling: torch.Tensor

    def __init__(
        self,
        obs_to_action_module: nn.Module,
        module_contains_mpc: bool,
        init_entropy_scaling: float,
        target_entropy: float,
        accept_n_nonconvergences_per_batch: int = 0,
        num_batch_dimensions: int = 1,
    ):
        """
        Parameters:
            obs_to_action_module: The module that is given the observation and predicts the action, etc. (see forward)
            module_contains_mpc: Whether the module contains an MPC and hence, returns a status of the MPC solution (see forward).
            init_entropy_scaling: The initial entropy bonus scaling.
            target_entropy: The target entropy for the automatic entropy bonus scaling (here referred to as alpha).
            accept_n_nonconvergences_per_batch: An option to adapt to nonconvergences that may appear while training with an MPC in the loop.
                If this number of nonconvergences is reached in a batch, the loss will be 0.
                If the number is not reached, the nonconvergend samples will still be removed from the batch before training.
            num_batch_dimensions: The number of batch dimensions in the input data. Necessary for cleansing the loss of nonconvergent samples.
        """
        super().__init__()
        self.obs_to_action_module = obs_to_action_module
        self.log_entropy_scaling = torch.tensor(np.log(init_entropy_scaling))  # type:ignore
        self.log_entropy_scaling.requires_grad = True
        self.module_contains_mpc = module_contains_mpc
        if module_contains_mpc:
            self.cleansed_mean = CleanseAndReducePerSampleLoss(
                reduction="mean",
                num_batch_dimensions=num_batch_dimensions,
                n_nonconvergences_allowed=accept_n_nonconvergences_per_batch,
                throw_exception_if_exceeded=True,  # Current loss calculation assumes it knows when to stop preliminary, so throw an error when it does not.
            )
        else:
            self.cleansed_mean = None

        self.target_entropy = target_entropy

    @abstractmethod
    def forward(
        self, obs: Any, deterministic: bool = False
    ) -> (
        tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """

        Parameters:
        obs: The observation for which the action should be determined.
        deterministic: If True, the action is drawn deterministically.

        Returns:
            The action and the log_prob of the action, and statistics
            or the action, the log_prob of the action, the status of the MPC and statistics, if the module contains an MPC.
        """
        out = self.obs_to_action_module.forward(obs, deterministic=deterministic)
        maybe_dict = out[-1]
        if isinstance(maybe_dict, dict):
            return out
        else:
            return *out, dict()

    def loss(
        self,
        q1: SACQNet,
        q2: SACQNet,
        mini_batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Calculates the loss for the actor and the automatic entropy bonus scaling.

        Parameters:
            q1: The first critic to train against.
            q2: The second critic to train against.
            mini_batch: The mini_batch containing the states, actions, rewards, next_states and dones.

        Returns:
            The loss for the actor, the loss for the automatic entropy bonus scaling, and a dictionary containing the following keys

                success: A boolean indicating whether the actor training was successful.

                actor_nonconvergencent_samples: The number of nonconvergences that appeared in the actor training.

                actor_mean_entropy_bonus: The mean entropy bonus of this batch.

                Whatever else self.module returns in the forward
        """
        s, _, _, _, _ = mini_batch
        stats = dict()
        if self.module_contains_mpc:
            a, log_prob, status, fwd_stats = self.forward(s)  # type:ignore
            status = status.unsqueeze(-1)  # type:ignore
            stats["status"] = status
            error_mask = status.to(dtype=torch.bool)  # type:ignore
            nonconvergences = torch.sum(error_mask).item()
            stats["nonconvergent_samples"] = nonconvergences
            if nonconvergences > self.cleansed_mean.n_nonconvergences_allowed:  # type:ignore
                stats["success"] = False
                return (
                    torch.zeros(1),
                    torch.zeros(1),
                    stats,
                )
            else:
                stats["success"] = True
        else:
            stats["success"] = True
            stats["nonconvergencent_samples"] = 0
            a, log_prob, fwd_stats = self.forward(s)  # type:ignore

        stats.update(fwd_stats)  # type:ignore
        stats["mean_log_prob"] = log_prob.mean().item()
        entropy = -self.log_entropy_scaling.exp() * log_prob

        stats["mean_entropy_bonus"] = entropy.mean().item()

        q1_val, q1_stats = q1(s, a)
        q2_val, q2_stats = q2(s, a)
        add_prefix_extend("q1_", stats, q1_stats)
        add_prefix_extend("q2_", stats, q2_stats)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  # for gradient ascent
        alpha_loss = -(
            self.log_entropy_scaling.exp() * (log_prob + self.target_entropy).detach()
        )
        if self.cleansed_mean is not None:
            loss, _ = self.cleansed_mean(loss, status)
            alpha_loss, _ = self.cleansed_mean(alpha_loss, status)
        else:
            loss = loss.mean()
            alpha_loss = alpha_loss.mean()

        return loss, alpha_loss, stats


@dataclass(kw_only=True)
class SACConfig(BaseTrainerConfig):
    # Trainer
    actor_lr: float
    critic_lr: float
    entropy_scaling_lr: float
    soft_update_frequency: int
    discount_factor: float
    batch_size: int
    device: str

    # Replay Buffer
    replay_buffer_size: int

    # Actor
    init_entropy_scaling: float
    target_entropy: (
        float | None
    )  # NOTE None is allowed here because this is often inferred as -a_dim.
    minimal_std: (
        float  # Will be used in the TanhNormal to avoid collapse of the distribution
    )
    param_factor: np.ndarray
    param_shift: np.ndarray

    # Critic
    soft_update_factor: float
    soft_update_frequency: int

    # Logging
    save_frequency: int
    moving_average_width: int


class SACTrainer(Trainer):
    """A class to handle training of the different components in the SAC algorithm.
    NOTE: You might want to override the method normalize_transitions_before_training which is applied
    to a batch before training with it."""

    def __init__(
        self,
        actor: SACActor,
        critic1: SACQNet,
        critic2: SACQNet,
        critic1_target: SACQNet,
        critic2_target: SACQNet,
        replay_buffer: ReplayBuffer,
        logger: NumberLogger,
        config: SACConfig,
    ):
        """
        Parameters:
            actor: The actor to train.
            critic1: The first critic to train.
            critic2: The second critic to train.
            critic1_target: The target network for the first critic.
            critic2_target: The target network for the second critic.
            replay_buffer: The replay buffer to sample from.
            config: The configuration for the trainer.
        """
        super().__init__()
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.critic1_target = critic1_target
        self.critic2_target = critic2_target

        self.device = config.device
        self.actor = self.actor.to(self.device)
        self.critic1 = self.critic1.to(self.device)
        self.critic2 = self.critic2.to(self.device)
        self.critic1_target = self.critic1_target.to(self.device)
        self.critic2_target = self.critic2_target.to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.obs_to_action_module.parameters(), lr=config.actor_lr
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.actor.log_entropy_scaling], lr=config.entropy_scaling_lr
        )
        self.critics_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.critic_lr,
        )

        self.replay_buffer = replay_buffer

        self.discount_factor = config.discount_factor
        self.batch_size = config.batch_size
        self.soft_update_frequency = config.soft_update_frequency
        self.soft_update_counter = 0

        self.logger = logger

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        # Standard assuming obs is just a numpy array.
        return tensor_to_numpy(
            self.actor.forward(torch.tensor(obs), deterministic=deterministic)[0]
        )

    def train(self):
        """
        Do a training step with the given mini_batch, updating the actor and the critics one time, respectively,
        also updating the target networks with a soft update every self.soft_update_frequency steps.

        NOTE: As it is implemented now, the Q updates are still done, even if the actor training fails
        (because we can still do them, since the gradient needs not be backpropagated through the q_target).
        """
        stats = dict()
        mini_batch = self.normalize_transitions_before_training(
            self.replay_buffer.sample(n=self.batch_size)
        )
        actor_loss, alpha_loss, actor_stats = self.actor.loss(
            self.critic1, self.critic2, mini_batch
        )
        actor_stats["mean_loss"] = actor_loss.item()
        actor_stats["alpha_loss"] = alpha_loss.item()
        actor_stats["alpha"] = self.actor.log_entropy_scaling.exp().item()
        add_prefix_extend("actor_", stats, actor_stats)
        actor_succeeded = stats["actor_success"]
        if actor_succeeded is None:
            raise ValueError("The actor training did not return a success status.")
        if actor_succeeded:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.soft_update_counter += 1
        # NOTE: For now no logging happens in target calculation (all stat dicts are ignored).
        q_target = self.calc_Q_target(mini_batch)
        stats["q_target_mean"] = q_target.mean().item()
        c1_loss, c1_stats = self.critic1.loss(q_target, mini_batch)
        c1_stats["mean_loss"] = c1_loss.item()
        add_prefix_extend("critic1_", stats, c1_stats)
        c2_loss, c2_stats = self.critic2.loss(q_target, mini_batch)
        c2_stats["mean_loss"] = c2_loss.item()
        add_prefix_extend("critic2_", stats, c2_stats)

        self.critics_optimizer.zero_grad()
        (c1_loss + c2_loss).backward()
        self.critics_optimizer.step()

        if self.soft_update_counter == self.soft_update_frequency:
            self.critic1.soft_update(self.critic1_target)
            self.critic2.soft_update(self.critic2_target)
            self.soft_update_counter = 0

        return stats

    def calc_Q_target(
        self,
        mini_batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        """NOTE: No logging happens here (all stat dicts are ignored)."""
        s, a, r, s_prime, done = mini_batch

        with torch.no_grad():
            actor_fwd = self.actor.forward(s_prime)
            a_prime, log_prob = actor_fwd[0], actor_fwd[1]
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            entropy = -self.actor.log_entropy_scaling.exp() * log_prob
            q1_val, q1_stats = self.critic1_target(s_prime, a_prime)
            q2_val, q2_stats = self.critic2_target(s_prime, a_prime)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = r + self.discount_factor * (1 - done) * (min_q + entropy)

        return target

    def normalize_transitions_before_training(
        self,
        transition_batch: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ):
        """Normalize the transition before training with it."""
        return transition_batch

    def save(self, save_directory: str):
        """Save the models in the given directory."""
        torch.save(self.actor.state_dict(), f"{save_directory}/actor.pth")
        torch.save(self.critic1.state_dict(), f"{save_directory}/critic1.pth")
        torch.save(self.critic2.state_dict(), f"{save_directory}/critic2.pth")
        torch.save(
            self.critic1_target.state_dict(), f"{save_directory}/critic1_target.pth"
        )
        torch.save(
            self.critic2_target.state_dict(), f"{save_directory}/critic2_target.pth"
        )

    def load(self, save_directory: str):
        """Load the models from the given directory. Is meant to be exactly compatible with save."""
        self.actor.load_state_dict(
            torch.load(f"{save_directory}/actor.pth", weights_only=True)
        )
        self.critic1.load_state_dict(
            torch.load(f"{save_directory}/critic1.pth", weights_only=True)
        )
        self.critic2.load_state_dict(
            torch.load(f"{save_directory}/critic2.pth", weights_only=True)
        )
        self.critic1_target.load_state_dict(
            torch.load(f"{save_directory}/critic1_target.pth", weights_only=True)
        )
        self.critic2_target.load_state_dict(
            torch.load(f"{save_directory}/critic2_target.pth", weights_only=True)
        )

    def log(self, stats: dict[str, Any], commit: bool):
        current_log = self.logger.log(stats, commit)
        return current_log  # Return the modified stats so you can do your own logging on top of this.
