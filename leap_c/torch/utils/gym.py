from typing import Any

import gymnasium as gym
import torch
from gymnasium.spaces import utils as gym_utils


def flatten_space(space: gym.Space) -> gym.spaces.Box:
    """Return the flattened Gymnasium space."""
    flat_space = gym_utils.flatten_space(space)
    if not isinstance(flat_space, gym.spaces.Box):
        raise NotImplementedError(f"Cannot flatten space of type {type(space)} into a Box.")
    return flat_space


def flatten(space: gym.Space, x: Any):
    """Flatten a value from a Gymnasium space using torch operations."""
    if isinstance(space, gym.spaces.Box):
        x = torch.as_tensor(x)
        return x.reshape(*x.shape[: -len(space.shape)], gym_utils.flatdim(space))

    if isinstance(space, gym.spaces.Dict):
        return torch.cat(
            [flatten(subspace, x[key]) for key, subspace in space.spaces.items()], dim=-1
        )

    if isinstance(space, gym.spaces.Tuple):
        return torch.cat([flatten(subspace, x_i) for subspace, x_i in zip(space.spaces, x)], dim=-1)

    raise NotImplementedError(f"flatten does not support space type {type(space)}.")


def unflatten(space: gym.Space, x: Any):
    """Unflatten a flat tensor into a value matching a Gymnasium space."""
    x = torch.as_tensor(x)

    if isinstance(space, gym.spaces.Box):
        return x.reshape(*x.shape[:-1], *space.shape)

    if isinstance(space, gym.spaces.Dict):
        out = {}
        offset = 0
        for key, subspace in space.spaces.items():
            width = gym_utils.flatdim(subspace)
            out[key] = unflatten(subspace, x[..., offset : offset + width])
            offset += width
        return out

    if isinstance(space, gym.spaces.Tuple):
        out = []
        offset = 0
        for subspace in space.spaces:
            width = gym_utils.flatdim(subspace)
            out.append(unflatten(subspace, x[..., offset : offset + width]))
            offset += width
        return tuple(out)

    raise NotImplementedError(f"unflatten does not support space type {type(space)}.")
