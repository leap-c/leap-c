import collections
import random
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data._utils.collate import collate

from leap_c.collate import create_collate_fn_map, pytree_tensor_to


class ReplayBuffer(nn.Module):
    """Replay buffer for storing transitions.

    The replay buffer is a deque that stores transitions in a FIFO manner. The buffer has
    a maximum size, and when the buffer is full, the oldest transitions are discarded
    when putting in a new one.

    Attributes:
        buffer: A deque that stores the transitions.
        device: The device to which all sampled tensors will be cast.
        collate_fn_map: The collate function map that informs the buffer how to form batches.
        tensor_dtype: The data type to which the tensors in the observation will be cast.
    """

    def __init__(
        self,
        buffer_limit: int,
        device: str,
        tensor_dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the replay buffer.

        Args:
            buffer_limit: The maximum number of transitions that can be stored in the buffer.
                If the buffer is full, the oldest transitions are discarded when putting in a new one.
            device: The device to which all sampled tensors will be cast.
            collate_fn_map: The collate function map that informs the buffer how to form batches.
            tensor_dtype: The data type to which the tensors in the observation will be cast.
        """
        super().__init__()
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device
        self.tensor_dtype = tensor_dtype

        # TODO (Jasper): This should be derived from task.
        self.collate_fn_map = create_collate_fn_map()

    def put(self, data: Any):
        """Put the data into the replay buffer. If the buffer is full, the oldest data is discarded.

        Args:
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

        Args:
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

    def get_extra_state(self) -> dict:
        """State of the replay buffer.

        This interface is used by state_dict and load_state_dict of nn.Module.
        """
        return {"buffer": self.buffer}

    def set_extra_state(self, state: dict):
        """Set the state dict of the replay buffer.

        This interface is used by state_dict and load_state_dict of nn.Module.

        Args:
            state: The state dict to set.
        """
        buffer = state["buffer"]
        self.buffer = collections.deque(buffer, maxlen=self.buffer.maxlen)
