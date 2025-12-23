from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


def raise_value_zero_exception(field_name):
    raise ValueError(
        f"This kind of randomization is uneffective for parameter '{field_name}' "
        "containing zero values. "
        "Consider skipping it or using a different randomization method."
    )


def randomize_normal(
    params: Any,
    rng: np.random.Generator,
    noise_scale: float = 0.3,
    skip_names: list[str] | None = None,
    override_noise_scale: dict[str, float] | None = None,
) -> Any:
    """Generate a new instance of the parameter dataclass with randomized values.

    The randomization is done by sampling from a normal distribution centered
    at the original parameter value with a standard deviation noise_scale*|value|.
    Raises an error when |value| is zero, in which case the parameter is probably unsuited to be
    randomized this way.

    Args:
        params: Instance of a dataclass containing parameters to randomize.
        rng: NumPy random generator for reproducibility.
        noise_scale: Scale for parameter randomization (default: 0.3).
        skip_names: List of parameter names to skip randomization.
        override_noise_scale: Dictionary mapping parameter field names to custom noise scales.

    Returns:
        New instance with randomized values.
    """
    if not is_dataclass(params):
        raise ValueError("params must be a dataclass instance")
    randomized_params = {}
    if override_noise_scale is None:
        override_noise_scale = {}
    if skip_names is None:
        skip_names = []
    for field_name in fields(params):
        if field_name.name in skip_names:
            randomized_params[field_name.name] = getattr(params, field_name.name)
            continue
        value = getattr(params, field_name.name)
        scale = override_noise_scale.get(field_name.name, noise_scale)
        if isinstance(value, (int, float)):
            if value == 0:
                raise_value_zero_exception(field_name.name)
            randomized_params[field_name.name] = rng.normal(loc=value, scale=scale * np.abs(value))
        elif isinstance(value, list):
            new_list = []
            for v in value:
                if v == 0:
                    raise_value_zero_exception(field_name.name)
                new_v = rng.normal(loc=v, scale=scale * np.abs(v))
                new_list.append(new_v)
            randomized_params[field_name.name] = new_list
        elif isinstance(value, np.ndarray):
            if np.any(value == 0):
                raise_value_zero_exception(field_name.name)
            randomized_params[field_name.name] = rng.normal(loc=value, scale=scale * np.abs(value))
        else:
            raise Exception(
                "Only list and np.ndarray fields are currently supported "
                f"for randomization, but got {type(value)}."
            )
    return type(params)(**randomized_params)
