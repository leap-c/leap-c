"""Provides a trainer for a SAC algorithm with a parameterized controller.

Trainer carries out episodic rollouts for a given parameter and updates the policy accordingly.
"""

from typing import Generator

import torch

from leap_c.torch.rl.mpc_actor import StochasticMPCActorOutput
from leap_c.torch.rl.sac_zop import SacZopTrainer
from leap_c.torch.rl.utils import soft_target_update
from leap_c.torch.utils.seed import mk_seed
from leap_c.utils.gym import seed_env


class SacZopTrainerEpisodic(SacZopTrainer):
    """A trainer that implements SAC with a controller in the policy network.

    The controller is used to compute actions but without differentiating through it (SAC-ZOP).
    Uses parameter noise and a parameter critic.

    Attributes:
        train_env: The training environment.
        q: The Q-function approximator (critic).
        q_target: The target Q-function approximator.
        q_optim: The optimizer for the Q-function.
        pi: The policy network containing the parameterized controller (the actor).
        pi_optim: The optimizer for the policy network.
        log_alpha: The log of the temperature parameter.
        alpha_optim: The optimizer for the temperature parameter.
            If `None`, the temperature is fixed.
        target_entropy: The target entropy for the policy.
            If `None`, the temperature is fixed.
        entropy_norm: The normalization factor for the entropy term.
            Normalizes the entropy based on the ratio of parameter and action dimensions.
        buffer: The replay buffer used to store transitions.
    """

    def train_loop(self) -> Generator[int, None, None]:
        is_terminated = is_truncated = True
        policy_ctx = None
        obs = None

        while True:
            if is_terminated or is_truncated:
                obs, _ = seed_env(self.train_env, mk_seed(self.rng), {"mode": "train"})
                policy_ctx = None
                is_terminated = is_truncated = False

            obs_batched = self.buffer.collate([obs])

            with torch.no_grad():
                pi_output: StochasticMPCActorOutput = self.pi(
                    obs_batched, policy_ctx, deterministic=False
                )
            assert pi_output.action is not None, "Expected action to be not `None`"
            action = pi_output.action.cpu().numpy()[0]
            param = pi_output.param.cpu().numpy()[0]

            self.report_stats("train_trajectory", {"action": action, "param": param}, verbose=True)
            self.report_stats("train_policy_rollout", pi_output.stats, verbose=True)

            obs_prime, ret, is_terminated, is_truncated, ep_info_list = self.train_env.step(
                action
            )  # step_rollout
            ep_length = len(ep_info_list.get("rewards", []))

            if "episode" in ep_info_list or "task" in ep_info_list:
                self.report_stats(
                    "train", ep_info_list.get("episode", {}) | ep_info_list.get("task", {})
                )

            data_keys = ["observations", "rewards", "dones", "truncated", "info"]
            for items in zip(*(ep_info_list[key] for key in data_keys)):
                obs_prime, reward, is_terminated, is_truncated, info = items
                self.buffer.put((obs, param, reward, obs_prime, is_terminated))
                obs = obs_prime

            obs = obs_prime
            policy_ctx = pi_output.ctx

            if (
                self.state.step >= self.cfg.train_start
                and len(self.buffer) >= self.cfg.batch_size
                and self.state.step % self.cfg.update_freq == 0
            ):
                for i in range(ep_length):
                    # sample batch
                    o, a, r, o_prime, te = self.buffer.sample(self.cfg.batch_size)

                    # sample action
                    pi_o = self.pi(o, None, only_param=True)
                    a_pi = pi_o.param
                    log_p = pi_o.log_prob / self.entropy_norm

                    # update temperature
                    if self.alpha_optim is not None:
                        alpha_loss = -torch.mean(
                            self.log_alpha.exp() * (log_p + self.target_entropy).detach()
                        )
                        self.alpha_optim.zero_grad()
                        alpha_loss.backward()
                        self.alpha_optim.step()

                    # update critic
                    alpha = self.log_alpha.exp().item()
                    with torch.no_grad():
                        pi_o_prime = self.pi(o_prime, None, only_param=True)
                        q_target = torch.cat(self.q_target(o_prime, pi_o_prime.param), dim=1)
                        q_target = torch.min(q_target, dim=1, keepdim=True).values

                        # add entropy
                        factor = self.cfg.entropy_reward_bonus / self.entropy_norm
                        q_target = q_target - alpha * pi_o_prime.log_prob * factor

                        target = r[:, None] + self.cfg.gamma * (1 - te[:, None]) * q_target

                    q = torch.cat(self.q(o, a), dim=1)
                    q_loss = torch.mean((q - target).pow(2))

                    self.q_optim.zero_grad()
                    q_loss.backward()
                    self.q_optim.step()

                    # update actor
                    q_pi = torch.cat(self.q(o, a_pi), dim=1)
                    min_q_pi = torch.min(q_pi, dim=1, keepdim=True).values
                    pi_loss = (alpha * log_p - min_q_pi).mean()

                    self.pi_optim.zero_grad()
                    pi_loss.backward()
                    self.pi_optim.step()

                    # soft updates
                    if i % self.cfg.soft_update_freq == 0:
                        soft_target_update(self.q, self.q_target, self.cfg.tau)

                    # report stats
                    loss_stats = {
                        "q_loss": q_loss.item(),
                        "pi_loss": pi_loss.item(),
                        "alpha": alpha,
                        "q": q.mean().item(),
                        "q_target": target.mean().item(),
                        "entropy": -log_p.mean().item(),
                    }
                    self.report_stats("loss", loss_stats, verbose=True)

            yield 1, float(ret)
