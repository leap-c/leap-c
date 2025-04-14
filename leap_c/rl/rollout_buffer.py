from typing import Tuple, Callable

import torch
from torch.utils.data._utils.collate import collate

from leap_c.collate import create_collate_fn_map, pytree_tensor_to


class RolloutBuffer:
    def __init__(
            self,
            buffer_limit: int,
            obs_shape: Tuple[int, ...],
            action_shape: Tuple[int, ...],
            device: str,
            tensor_dtype: torch.dtype = torch.float32,
            collate_fn_map: dict[type, Callable] | None = None,
    ):
        """
        Rollout buffer for storing transitions.

        A group of tensors to store the transitions, it allows for random access on both read and write. This is not intended
        to be used as a replay buffer, since it does not support sampling. Also, it's optimized as a reusable buffer, and not
        a persistent one.

        Args:
            buffer_limit: The maximum number of transitions that can be stored in the buffer.
            obs_shape: The shape of the observation.
            action_shape: The shape of the action.
            device: The device to which all tensors will be cast.
            tensor_dtype: The data type to which the tensors in the observation will be cast.
            collate_fn_map: The collate function map that informs the buffer how to form batches.
        """
        self.observations = torch.zeros((buffer_limit,) + obs_shape)
        self.actions = torch.zeros((buffer_limit,) + action_shape)
        self.log_probs = torch.zeros(buffer_limit)
        self.rewards = torch.zeros(buffer_limit)
        self.dones = torch.zeros(buffer_limit)
        self.values = torch.zeros(buffer_limit)

        self.device = device
        self.tensor_dtype = tensor_dtype

        self.collate_fn_map = collate_fn_map if collate_fn_map is not None else create_collate_fn_map()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            mini_batch = [(
                self.observations[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.rewards[idx],
                self.dones[idx],
                self.values[idx]
            )]
        else:
            indices = list(range(*idx.indices(self.observations.size(0))))
            mini_batch = [(
                self.observations[i],
                self.actions[i],
                self.log_probs[i],
                self.rewards[i],
                self.dones[i],
                self.values[i]
            ) for i in indices]

        return pytree_tensor_to(
            collate(mini_batch, collate_fn_map=self.collate_fn_map),
            device=self.device,
            tensor_dtype=self.tensor_dtype,
        )

    def __setitem__(self, idx, data: tuple) -> None:
        obs, action, log_prob, reward, done, value = data
        self.observations[idx] = obs
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
