from typing import NamedTuple, Literal
import numpy as np
import gymnasium as gym


class Parameter(NamedTuple):
    """
    High-level parameter class for flexible parameter configuration.

    This class provides a user-friendly interface for defining parameter sets without
    requiring knowledge of internal tools. It supports
    configurable properties for default values, bounds, and parameter behavior.

    Attributes:
        name: The name identifier for the parameter.
        value: The parameter's numerical value(s).
        bounds: A gym.spaces.Box defining the valid range for the parameter values.
            Only used for learnable parameters. Defaults to None (unbounded).
        interface: The interface type for the parameter.
            Can be "fix", "learnable", or "non-learnable". Defaults to "fix".
    """

    name: str
    default: np.ndarray
    space: gym.spaces.Box | None = None
    interface: Literal["fix", "learnable", "non-learnable"] = "fix"


class ParameterManager:
    """Manager for parameters."""

    parameters: dict[str, Parameter]
    learnable_parameters: dict[str, dict[str, int | np.ndarray]]
    non_learnable_parameters: dict[str, dict[str, int | np.ndarray]]

    def __init__(
        self,
        parameters: list[Parameter],
    ) -> None:
        self.parameters = {param.name: param for param in parameters}

        # Build map for learnable parameters and create flattened array
        self.learnable_parameters = {}
        learnable_arrays = []
        current_index = 0

        for name, parameter in self.parameters.items():
            if parameter.interface == "learnable":
                param_size = parameter.default.size
                self.learnable_parameters[name] = {
                    "start_idx": current_index,
                    "end_idx": current_index + param_size,
                    "shape": parameter.default.shape,
                }
                learnable_arrays.append(parameter.default.flatten())
                current_index += param_size

        self.learnable_array = (
            np.concatenate(learnable_arrays) if learnable_arrays else np.array([])
        )

    def get(self, name: str) -> Parameter:
        """Get a parameter by name."""
        if name not in self.parameters:
            raise KeyError(f"Parameter '{name}' not found.")
        return self.parameters[name]

    def learnable_params_default(self) -> np.ndarray:
        """
        Return the default values for all learnable parameters.

        Returns:
            np.ndarray: Flattened array of default values for learnable parameters.
        """
        return self.learnable_array.copy()

    def get_param_space(self) -> gym.Space:
        """
        Return a Gymnasium Box space for the learnable parameters.

        Returns:
            gym.spaces.Box: Flattened Box space with lower and upper bounds for learnable parameters.
        """

        learnable_spaces = []

        for name, param in self.parameters.items():
            if param.interface == "learnable":
                if param.space is not None:
                    learnable_spaces.append(param.space)
                else:
                    # Create unbounded space for parameters without bounds
                    param_shape = param.default.shape
                    unbounded_space = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=param_shape,
                        dtype=np.float32,
                    )
                    learnable_spaces.append(unbounded_space)

        if not learnable_spaces:
            # No learnable parameters - return empty box space
            return gym.spaces.Box(low=np.array([]), high=np.array([]), dtype=np.float32)
        elif len(learnable_spaces) == 1:
            # Single space - flatten it to ensure consistent return type
            space = learnable_spaces[0]
            return gym.spaces.Box(
                low=space.low.flatten(), high=space.high.flatten(), dtype=space.dtype
            )
        else:
            # Multiple spaces
            tuple_space = gym.spaces.Tuple(learnable_spaces)
            return gym.spaces.utils.flatten_space(tuple_space)
