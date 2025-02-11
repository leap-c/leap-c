import collections
import random
from typing import Any

import torch
from leap_c.collate import create_collate_fn_map, pytree_tensor_to
from torch.utils.data._utils.collate import collate


class ReplayBuffer:
    def __init__(
        self,
        buffer_limit: int,
        device: str,
        tensor_dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            buffer_limit: The maximum number of transitions that can be stored in the buffer.
                If the buffer is full, the oldest transitions are discarded when putting in a new one.
            device: The device to which all sampled tensors will be cast.
            collate_fn_map: The collate function map that informs the buffer how to form batches.
            tensor_dtype: The data type to which the tensors in the observation will be cast.
        """
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device
        self.tensor_dtype = tensor_dtype

        self.collate_fn_map = create_collate_fn_map()

    def put(self, data: Any):
        """Put the data into the replay buffer. If the buffer is full, the oldest data is discarded.

        Parameters:
            data: The data to put into the buffer.
                It should be collatable according to the custom_collate function.
        """
        self.buffer.append(data)

    def sample(self, n: int) -> Any:
        """
        Sample a mini-batch from the replay buffer,
        collate the mini-batch according to self.custom_collate_map
        and cast all tensors in the collated mini-batch (must be a pytree structure)
        to the device and dtype of the buffer.

        Parameters:
            n: The number of samples to draw.
        """
        mini_batch = random.sample(self.buffer, n)
        return pytree_tensor_to(
            collate(mini_batch, collate_fn_map=self.collate_fn_map),
            device=self.device,
            tensor_dtype=self.tensor_dtype,
        )

    def __len__(self):
        return len(self.buffer)
