import collections
import random
from typing import Any, Callable, Iterable

import torch
from torch.utils.data._utils.collate import collate

from leap_c.collate import create_collate_fn_map, pytree_tensor_to


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
            input_transformation: A function that transforms the data before it is put into the buffer.
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


class ReplayBufferWriteback(ReplayBuffer):
    """A ReplayBuffer where the data can be updated, e.g., when some information has changed at the time it was sampled."""

    def __init__(
        self, buffer_limit: int, device: str, tensor_dtype: torch.dtype = torch.float32
    ):
        super().__init__(buffer_limit, device, tensor_dtype)
        self.id = 0
        self.lookup: dict[int, Any] = (
            dict()
        )  # Keep a lookup table for the writeback instead of iterating through the deque

    def put(self, data: Iterable[Any]):
        """Almost the same as the put of the usual ReplayBuffer, but
        1. The input has to be an iterable.
        2. The input is appended with an id that can be used for the writeback.
        """
        entry = list(data)
        entry.append(self.id)
        self.id += 1

        self.buffer.append(data)
        self.lookup[self.id % self.buffer.maxlen] = entry  # type:ignore

    def writeback(self, id: int, inplace_modification: Callable[[list[Any]]]):
        """Apply the inplace_modification function to the data with the given id."""
        data_old = self.lookup[id % self.buffer.maxlen]  # type:ignore
        inplace_modification(data_old)
