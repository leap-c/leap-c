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
        lower_bound: Lower bounds for the parameter values. Only used for learnable.
            Defaults to None (unbounded).
        upper_bound: Upper bounds for the parameter values. Only used for learnable.
            Defaults to None (unbounded).
        interface: The interface type for the parameter.
            Can be "fix", "learnable", or "non-learnable". Defaults to "fix".
    """

    name: str
    default: np.ndarray
    lower_bound: np.ndarray | None = None
    upper_bound: np.ndarray | None = None
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

        # Build map for non-learnable parameters and create flattened array
        self.non_learnable_parameters = {}
        non_learnable_arrays = []
        current_index = 0

        for name, parameter in self.parameters.items():
            if parameter.interface == "non-learnable":
                param_size = parameter.default.size
                self.non_learnable_parameters[name] = {
                    "start_idx": current_index,
                    "end_idx": current_index + param_size,
                    "shape": parameter.default.shape,
                }
                non_learnable_arrays.append(parameter.default.flatten())
                current_index += param_size

        self.non_learnable_array = (
            np.concatenate(non_learnable_arrays)
            if non_learnable_arrays
            else np.array([])
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

    def _learnable_params_lower_bound(self) -> np.ndarray:
        """
        Return the lower bounds for all learnable parameters.

        Returns:
            np.ndarray: Flattened array of lower bounds for learnable parameters,
                    in the same order as learnable_array.
        """
        lower_bounds = []

        for name, param in self.parameters.items():
            if param.interface == "learnable":
                if param.lower_bound is not None:
                    lower_bounds.append(param.lower_bound.flatten())
                else:
                    # Use -infinity for unbounded parameters
                    param_size = param.default.size
                    lower_bounds.append(np.full(param_size, -np.inf))

        return np.concatenate(lower_bounds) if lower_bounds else np.array([])

    def _learnable_params_upper_bound(self) -> np.ndarray:
        """
        Return the upper bounds for all learnable parameters.

        Returns:
            np.ndarray: Flattened array of upper bounds for learnable parameters,
                    in the same order as learnable_array.
        """
        upper_bounds = []

        for name, param in self.parameters.items():
            if param.interface == "learnable":
                if param.upper_bound is not None:
                    upper_bounds.append(param.upper_bound.flatten())
                else:
                    # Use +infinity for unbounded parameters
                    param_size = param.default.size
                    upper_bounds.append(np.full(param_size, np.inf))

        return np.concatenate(upper_bounds) if upper_bounds else np.array([])

    def get_param_space(self) -> gym.Space:
        """
        Return a Gymnasium Box space for the learnable parameters.

        Returns:
            gym.spaces.Box: Box space with lower and upper bounds for learnable parameters.
        """

        low = self._learnable_params_lower_bound()
        high = self._learnable_params_upper_bound()

        if low.size == 0:
            # No learnable parameters - return empty box space
            return gym.spaces.Box(low=np.array([]), high=np.array([]), dtype=np.float32)

        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
