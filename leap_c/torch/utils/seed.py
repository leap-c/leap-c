import random

import numpy as np
import torch


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
