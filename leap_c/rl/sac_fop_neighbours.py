from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate
from sklearn.neighbors import KNeighborsClassifier

from leap_c.mpc import MPCBatchedState
from leap_c.nn.mlp import MLPConfig
from leap_c.registry import register_trainer
from leap_c.rl.replay_buffer import ReplayBufferWriteback
from leap_c.rl.sac_fop import (
    MPCSACActor,
    MPCSACActorPrimal,
    SACFOPBaseConfig,
    SACFOPTrainer,
    update_mpc_stats_train_loss,
    update_mpc_stats_train_rollout,
)
from leap_c.task import Task
from leap_c.trainer import Trainer


class NNClassifierBuffer(ReplayBufferWriteback):
    """This is meant as an experiment to to see how well the nearest neighbour strategy works in the first place.
    It can probably be made much faster and more memory efficient.
    """

    def __init__(
        self,
        buffer_limit: int,
        device: str,
        normalization: Callable[[np.ndarray], np.ndarray],
        short_to_long_term_freq: int = 1000,
        tensor_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(buffer_limit, device, tensor_dtype)

        self.nn_long_term = KNeighborsClassifier(n_neighbors=1, p=1, n_jobs=8)
        self.long_term_data = None
        self.long_term_labels = None
        self.long_term_idx = 0

        self.nn_short_term = KNeighborsClassifier(n_neighbors=1, p=1, n_jobs=2)
        self.short_term_data = None
        self.short_term_labels = None

        self.short_to_long_term_freq = short_to_long_term_freq
        assert self.buffer.maxlen % short_to_long_term_freq == 0  # type:ignore
        self.normalization = normalization

    def predict(self, x: np.ndarray, param: np.ndarray):
        query = np.concatenate((x, param), axis=1)
        query = self.normalization(query)
        idxs_long = self.nn_long_term.predict(query)
        idxs_short = self.nn_short_term.predict(query)
        return_data = []
        for i in range(len(idxs_long)):
            idx_long = idxs_long[i]
            idx_short = idxs_short[i]
            lookup_long = self.lookup[idx_long % self.buffer.maxlen]  # type:ignore
            lookup_short = self.lookup[idx_short % self.buffer.maxlen]  # type:ignore
            query_long = self.data_to_vector(lookup_long)
            query_short = self.data_to_vector(lookup_short)
            dist_long = np.linalg.norm(query - query_long, ord=1)
            dist_short = np.linalg.norm(query - query_short, ord=1)
            if dist_long < dist_short:
                return_data.append(lookup_long[-3])
            else:
                return_data.append(lookup_short[-3])
        if len(return_data) > 1:
            return AcadosOcpFlattenedBatchIterate(
                x=np.stack([x.x for x in return_data], axis=0),
                u=np.stack([x.u for x in return_data], axis=0),
                z=np.stack([x.z for x in return_data], axis=0),
                sl=np.stack([x.sl for x in return_data], axis=0),
                su=np.stack([x.su for x in return_data], axis=0),
                pi=np.stack([x.pi for x in return_data], axis=0),
                lam=np.stack([x.lam for x in return_data], axis=0),
                N_batch=len(return_data),
            )
        else:
            return return_data[0]

    def data_to_vector(self, data: Sequence[Any]):
        return self.normalization(np.concatenate((data[0], data[-2]), axis=0))

    def update_short_term(self, data: Sequence[Any]):
        if self.short_term_data is None:
            self.short_term_data = self.data_to_vector(data).reshape(1, -1)
            self.short_term_labels = np.array([data[-1]])

        elif self.short_term_data.shape[0] < self.short_to_long_term_freq:
            self.short_term_data = np.concatenate(
                (self.short_term_data, self.data_to_vector(data)), axis=0
            )
            self.short_term_labels = np.concatenate(
                (self.short_term_labels, data[-1]),  # type:ignore
                axis=0,
            )
        else:
            self.short_term_data[self.short_to_long_term_counter] = self.data_to_vector(
                data
            )
            self.short_term_labels[self.short_to_long_term_counter] = data[-1]  # type:ignore
            self.nn_short_term.fit(self.short_term_data, self.short_term_labels)  # type:ignore

        self.short_to_long_term_counter += 1

    def update_long_term(self):
        if self.long_term_data is None:
            self.long_term_data = self.short_term_data
            self.long_term_labels = self.short_term_labels
        elif self.long_term_data.shape[0] < self.buffer.maxlen:  # type:ignore
            self.long_term_data = np.concatenate(
                (self.long_term_data, self.short_term_data),  # type:ignore
                axis=0,
            )
            self.long_term_labels = np.concatenate(
                (self.long_term_labels, self.short_term_labels),  # type:ignore
                axis=0,
            )
        else:
            start = self.long_term_idx
            end = start + self.short_to_long_term_freq
            self.long_term_data[start:end, :] = self.short_term_data
            self.long_term_labels[start:end, :] = self.short_term_labels  # type:ignore

            self.nn_long_term.fit(self.long_term_data, self.long_term_labels)  # type:ignore

        self.long_term_idx += self.short_to_long_term_freq
        self.long_term_idx %= self.buffer.maxlen  # type:ignore

    def put(self, data: Sequence[Any]):
        super().put(data)

        self.update_short_term(data)
        if self.short_to_long_term_counter == self.short_to_long_term_freq:
            self.short_to_long_term_counter = 0
            self.update_long_term()


class MpcSacActorMpcStatePredictor(MPCSACActor):
    """The same as the MpcSacActor, but it always ignores the input mpc_state and instead
    asks the predictor what the MPC state should be, dependend on the current state obs
    and the params from the neural network."""

    def __init__(
        self,
        task: Task,
        trainer: Trainer,
        mlp_cfg: MLPConfig,
        predictor: Callable[[np.ndarray, np.ndarray], MPCBatchedState],
    ):
        self.predictor = predictor

        def prepare_mpc_state(
            x: torch.Tensor, p: torch.Tensor, mpc_state: MPCBatchedState
        ) -> MPCBatchedState:
            state = x.detach().cpu().numpy()
            param = p.detach().cpu().numpy()
            return self.predictor(state, param)

        super().__init__(
            task=task,
            trainer=trainer,
            mlp_cfg=mlp_cfg,
            prepare_mpc_state=prepare_mpc_state,
        )


@register_trainer("sac_fop_neighbours", SACFOPBaseConfig())
class NearestNeighbourInitStrategy(SACFOPTrainer):
    """This initialization strategy computes the nearest neighbour for the stored vectors of form (x, param)
    and uses the corresponding stored solutions."""

    def __init__(
        self, task: Task, output_path: str | Path, device: str, cfg: SACFOPBaseConfig
    ):
        super().__init__(task, output_path, device, cfg)

        param_space = self.task.param_space
        state_space = self.task.train_env.observation_space
        if len(state_space.shape) > 1000:  # type:ignore
            raise ValueError(
                "Note that the observation space is very large. Better use the initial state of the MPC instead (has yet to be implemented)."
            )
        if not param_space.is_bounded() or not state_space.is_bounded():  # type:ignore
            raise ValueError(
                "The parameter space and the observation space must be bounded for the nearest neighbour initialization strategy, as it is used for normalization."
            )
        loc_param = (param_space.high + param_space.low) / 2.0  # type: ignore
        scale_param = (param_space.high - param_space.low) / 2.0  # type: ignore
        loc_state = (state_space.high + state_space.low) / 2.0  # type: ignore
        scale_state = (state_space.high - state_space.low) / 2.0  # type: ignore
        loc = np.concatenate((loc_state, loc_param), axis=0)
        scale = np.concatenate((scale_state, scale_param), axis=0)

        def normalization(data: np.ndarray) -> np.ndarray:
            return (data - loc) / scale

        self.buffer = NNClassifierBuffer(
            buffer_limit=cfg.sac.buffer_size,
            device=device,
            normalization=normalization,
            short_to_long_term_freq=1000,
        )

    def train_loop(self) -> Iterator[int]:
        is_terminated = is_truncated = True
        episode_return = episode_length = np.inf
        policy_state = self.init_policy_state()
        mpc_stats_aggregated_rollout = {}

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
                a_pi, log_p, status, state_sol, param, mpc_stats = self.pi(o, ps_prime)
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
                    ) = self.pi(o_prime, ps_prime)
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


@register_trainer("sac_fop_neighbours_primal", SACFOPBaseConfig())
class NearestNeighbourInitStrategyPrimal(NearestNeighbourInitStrategy):
    """The same as NearestNeighbourInitStrategy, but only the primal variables are being used, the duals are always set to zero."""

    def __init__(
        self, task: Task, output_path: str | Path, device: str, cfg: SACFOPBaseConfig
    ):
        super().__init__(task, output_path, device, cfg)

        self.pi = MPCSACActorPrimal(task, self, cfg.sac.actor_mlp).to(device)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.sac.lr_pi)
