from typing import Iterator

import numpy as np
import torch

from leap_c.registry import register_trainer
from leap_c.rl.sac_fop import (
    SACFOPBaseConfig,
    SACFOPTrainer,
    update_mpc_stats_train_loss,
    update_mpc_stats_train_rollout,
)


@register_trainer("sac_fop_previous", SACFOPBaseConfig())
class PreviousInitStrategy(SACFOPTrainer):
    """This initialization strategy always uses the previous solution (like shift strategy but without shift).
    We do not expect this to perform well, but we want to use it as a baseline."""

    def train_loop(self) -> Iterator[int]:
        is_terminated = is_truncated = True
        episode_return = episode_length = np.inf
        policy_state = self.init_policy_state()
        mpc_stats_aggregated_rollout = {}
        state_previous = (
            self.pi.mpc.mpc.ocp_batch_solver.store_iterate_to_flat_obj()
        )  # Will be used in the losses

        while True:
            if is_terminated or is_truncated:
                obs, _ = self.train_env.reset()
                if (
                    episode_length < np.inf
                ):  # TODO: Add rollout-logging for the non-episodic case
                    stats = {}
                    for k, v in mpc_stats_aggregated_rollout.items():
                        if not ("max" in k or "min" in k):
                            stats["avg_" + k] = v / episode_length
                        else:
                            stats[k] = v

                    stats["episode_return"] = episode_return
                    stats["episode_length"] = episode_length
                    self.report_stats("train_rollout", stats, self.state.step)
                policy_state = (
                    self.init_policy_state()
                )  # NOTE: At rollout, this init strategy is the same as reload
                mpc_stats_aggregated_rollout = {}
                is_terminated = is_truncated = False
                episode_return = episode_length = 0
            action, policy_state_prime, param, status, mpc_stats = self.act(
                obs, state=policy_state
            )

            obs_prime, reward, is_terminated, is_truncated, _ = self.train_env.step(
                action
            )

            episode_return += float(reward)
            episode_length += 1
            update_mpc_stats_train_rollout(
                mpc_stats, mpc_stats_aggregated_rollout, status
            )

            self.buffer.put(
                (
                    obs,
                    action,
                    reward,
                    obs_prime,
                    is_terminated,
                    is_truncated,
                    policy_state,
                    policy_state_prime,
                    param,
                )
            )  # type: ignore

            obs = obs_prime
            policy_state = policy_state_prime

            if (
                self.state.step >= self.cfg.train.start
                and len(self.buffer) >= self.cfg.sac.batch_size
                and self.state.step % self.cfg.sac.update_freq == 0
            ):
                # sample batch
                o, a, r, o_prime, te, tr, ps, ps_prime, param = self.buffer.sample(
                    self.cfg.sac.batch_size
                )

                # sample action
                a_pi, log_p, status, state_sol, param, mpc_stats = self.pi(
                    o, state_previous
                )
                log_p = log_p.sum(dim=-1).unsqueeze(-1)

                # update temperature
                target_entropy = -np.prod(self.task.param_space.shape)  # type: ignore
                alpha_loss = -torch.mean(
                    self.log_alpha.exp() * (log_p + target_entropy).detach()
                )
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                # update critic
                alpha = self.log_alpha.exp().item()
                with torch.no_grad():
                    (
                        a_pi_prime,
                        log_p_prime,
                        status_prime,
                        state_sol_prime,
                        param_prime,
                        mpc_stats_prime,
                    ) = self.pi(o_prime, state_previous)
                    q_target = torch.cat(self.q_target(o_prime, a_pi_prime), dim=1)
                    q_target = torch.min(q_target, dim=1).values

                    # add entropy
                    q_target = q_target - alpha * log_p_prime[:, 0]

                    target = r + self.cfg.sac.gamma * (1 - te) * q_target

                q = torch.cat(self.q(o, a), dim=1)
                q_loss = torch.mean((q - target[:, None]).pow(2))

                self.q_optim.zero_grad()
                q_loss.backward()
                self.q_optim.step()

                # update actor
                mask_status = status == 0
                q_pi = torch.cat(self.q(o, a_pi), dim=1)
                min_q_pi = torch.min(q_pi, dim=1).values
                pi_loss = (alpha * log_p - min_q_pi)[mask_status].mean()

                self.pi_optim.zero_grad()
                pi_loss.backward()
                self.pi_optim.step()

                state_previous = state_sol

                # soft updates
                for q, q_target in zip(self.q.parameters(), self.q_target.parameters()):
                    q_target.data = (
                        self.cfg.sac.tau * q.data
                        + (1 - self.cfg.sac.tau) * q_target.data
                    )

                report_freq = self.cfg.sac.report_loss_freq * self.cfg.sac.update_freq

                if self.state.step % report_freq == 0:
                    loss_stats = {
                        "q_loss": q_loss.item(),
                        "pi_loss": pi_loss.item(),
                        "alpha": alpha,
                        "q": q.mean().item(),
                        "q_target": target.mean().item(),
                        "not_converged": (status != 0).float().mean().item(),
                    }
                    update_mpc_stats_train_loss(
                        mpc_stats=mpc_stats, loss_stats=loss_stats, actual_status=status
                    )
                    self.report_stats("train_loss", loss_stats, self.state.step + 1)

            yield 1
