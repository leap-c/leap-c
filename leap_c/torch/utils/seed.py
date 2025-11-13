import random
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch

RngType: TypeAlias = (
    int | Sequence[int] | np.random.SeedSequence | np.random.BitGenerator | np.random.Generator
)
MAX_SEED = np.iinfo(np.uint32).max + 1


def set_seed(seed: int) -> np.random.Generator:
    """Set the seed for all random number generators.

    Args:
        seed: The seed to use.

    Returns:
        np.random.Generator: A numpy random number generator initialized with the given seed.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa:NPY002
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)


def mk_seed(rng: np.random.Generator) -> int:
    """Generates a random seed compatible with `gymnasium.Env.reset`.

    Args:
        rng: A `numpy.random.Generator` instance.

    Returns:
        int: A random integer in the range [0, 2**32).
    """
    return int(rng.integers(MAX_SEED))
