import collections
import random
from typing import Any, Callable

import torch
from tensordict import TensorDict
from torch.utils._pytree import tree_map_only
from torch.utils.data._utils.collate import collate, default_collate_fn_map


def pytree_tensor_to(
    pytree: Any, device: int | str | torch.device, tensor_dtype: torch.dtype
) -> Any:
    """Convert tensors in the pytree to tensor_dtype and move them to device."""
    return tree_map_only(
        (torch.Tensor, TensorDict), lambda t: t.to(device=device, dtype=tensor_dtype), pytree
    )


def collate_dict_to_tensordict(batch: list, *, collate_fn_map=None) -> TensorDict:
    """Collate a batch of dicts into a `TensorDict`.

    This custom collation handler converts `dict` observations to `TensorDict` automatically during
    batch collation, enabling proper batching of `dict` observations in the replay buffer.

    Args:
        batch: A list of dictionaries to collate.
        collate_fn_map: The collate function map (ignored, kept for compatibility).

    Returns:
        A `TensorDict` with collated data and `batch_size=[len(batch)]`.
    """
    elem = batch[0]
    collated = {
        key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem
    }
    return TensorDict(collated, batch_size=[len(batch)])


class ReplayBuffer(torch.nn.Module):
    """Replay buffer for storing transitions.

    The replay buffer is a `deque` that stores transitions in a FIFO manner. The buffer has a
    maximum size, and when the buffer is full, the oldest transitions are discarded when appending a
    new one.

    Attributes:
        buffer: A deque that stores the transitions.
        device: The device to which all sampled tensors will be cast.
        collate_fn_map: The collate function map that informs the buffer how to form batches.
            For more information, please refer to the official pytorch documentation, e.g.,
            https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.default_collate .
        tensor_dtype: The data type to which the tensors will be cast.
    """

    buffer: collections.deque
    device: torch.device
    collate_fn_map: dict[tuple | tuple[type, ...], Callable]
    tensor_dtype: torch.dtype

    def __init__(
        self,
        buffer_limit: int,
        device: int | str | torch.device | None = None,
        tensor_dtype: torch.dtype | None = None,
        collate_fn_map: dict[tuple | tuple[type, ...], Callable] | None = None,
    ) -> None:
        """Initialize the replay buffer.

        Args:
            buffer_limit: The maximum number of transitions that can be stored in the buffer.
                If the buffer is full, the oldest transition is discarded when appending a new one.
            device: The device to which all sampled tensors will be cast. If `None`, the default
                PyTorch device is used.
            tensor_dtype: The data type to which the sampled tensors will be cast. If `None`, the
                default PyTorch type is used.
            collate_fn_map: The collate function map that informs the buffer how to form batches.
                If given, extends the default collate function map of PyTorch.
        """
        super().__init__()
        self.buffer = collections.deque(maxlen=buffer_limit)

        self.device = torch.get_default_device() if device is None else torch.device(device)
        self.tensor_dtype = torch.get_default_dtype() if tensor_dtype is None else tensor_dtype

        self.collate_fn_map = {**default_collate_fn_map, dict: collate_dict_to_tensordict}
        if collate_fn_map is not None:
            self.collate_fn_map.update(collate_fn_map)

    def put(self, data: Any) -> None:
        """Put the data into the replay buffer. If the buffer is full, the oldest data is discarded.

        Args:
            data: The data to put into the buffer.
                It should be collatable according to the `collate` function.
        """
        self.buffer.append(data)

    def sample(self, n: int) -> Any:
        """Sample a mini-batch from the replay buffer, and collate it.

        The collate is according to the `collate` function of this class.

        Args:
            n: The number of samples to draw.
        """
        mini_batch = random.sample(self.buffer, n)
        return self.collate(mini_batch)

    def collate(self, batch: Any) -> Any:
        """Collate a batch of data according to the collate function map.

        After collating, move and cast all tensors in the
        collated batch (must be a pytree structure).

        Args:
            batch: The batch of data to collate.

        Returns:
            The collated batch.
        """
        return pytree_tensor_to(
            collate(batch, collate_fn_map=self.collate_fn_map), self.device, self.tensor_dtype
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def get_extra_state(self) -> dict:
        """State of the replay buffer.

        This interface is used by `state_dict` and `load_state_dict` of `nn.Module`.
        """
        return {"buffer": self.buffer}

    def set_extra_state(self, state: dict) -> None:
        """Set the state dict of the replay buffer.

        This interface is used by `state_dict` and `load_state_dict` of `nn.Module`.

        Args:
            state: The state dict to set.
        """
        buffer = state["buffer"]
        self.buffer = collections.deque(buffer, maxlen=self.buffer.maxlen)
