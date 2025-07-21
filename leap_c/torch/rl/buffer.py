import collections
import random
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torch.utils._pytree import tree_map_only


def pytree_tensor_to(pytree: Any, device: str, tensor_dtype: torch.dtype) -> Any:
    """Convert tensors in the pytree to tensor_dtype and move them to device."""
    return tree_map_only(
        torch.Tensor,
        lambda t: t.to(device=device, dtype=tensor_dtype),
        pytree,
    )


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
        collate_fn_map: Optional[dict[Union[tuple, tuple[type, ...]], Callable]] = None,
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

        if collate_fn_map is None:
            self.collate_fn_map = default_collate_fn_map
        else:
            self.collate_fn_map = {**default_collate_fn_map, **collate_fn_map}

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
        return self.collate(mini_batch)

    def collate(self, batch: Any) -> Any:
        """Collate a batch of data according to the collate function map.

        Args:
            batch: The batch of data to collate.

        Returns:
            The collated batch.
        """
        return pytree_tensor_to(
            collate(batch, collate_fn_map=self.collate_fn_map),  # type: ignore
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
