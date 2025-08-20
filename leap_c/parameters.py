from typing import NamedTuple, Literal
import numpy as np


class Parameter(NamedTuple):
    """
    High-level parameter class for flexible parameter configuration.

    This class provides a user-friendly interface for defining parameter sets without
    requiring knowledge of internal tools. It supports
    configurable properties for default values, bounds, and parameter behavior.

    Attributes:
        name: The name identifier for the parameter.
        value: The parameter's numerical value(s).
        lower_bound: Lower bounds for the parameter values.
            Defaults to None (unbounded).
        upper_bound: Upper bounds for the parameter values.
            Defaults to None (unbounded).
        interface: The interface type for the parameter.
            Can be "fix", "learnable", or "non-learnable". Defaults to "fix".

    Note:
        TODO: Check about infinity bounds implementation in lower_bound and upper_bound.
    """

    name: str
    default: np.ndarray
    lower_bound: np.ndarray | None = None
    upper_bound: np.ndarray | None = None
    interface: Literal["fix", "learnable", "non-learnable"] = "fix"


class ParameterManager:
    """Manager for parameters."""

    parameters: dict[str, Parameter] = {}

    def __init__(
        self,
        params: list[Parameter],
    ) -> None:
        self.parameters = {param.name: param for param in params}

        # Build map for learnable parameters and create flattened array
        self.learnable_params = {}
        learnable_arrays = []
        current_index = 0

        for name, param in self.parameters.items():
            if param.interface == "learnable":
                param_size = param.default.size
                self.learnable_params[name] = {
                    "start_idx": current_index,
                    "end_idx": current_index + param_size,
                    "shape": param.default.shape,
                }
                learnable_arrays.append(param.default.flatten())
                current_index += param_size

        self.learnable_array = (
            np.concatenate(learnable_arrays) if learnable_arrays else np.array([])
        )

    def get(self, name: str) -> Parameter:
        """Get a parameter by name."""
        if name not in self.parameters:
            raise KeyError(f"Parameter '{name}' not found.")
        return self.parameters[name]

    def combine_parameter_values(
        self,
        batch_size: int | None = None,
        **overwrite: np.ndarray,
    ) -> np.ndarray:
        """
        Combine all parameters into a single numpy array.

        Args:
            batch_size: The batch size for the parameters.
            Not needed if overwrite is provided.
            **overwrite: Overwrite values for specific parameters.
                values need to be np.ndarray with shape (batch_size, ...).

        Returns:
            np.ndarray: shape (batch_size, np). with np being the number of learnable parameters.
        """
        # Infer batch size from overwrite if not provided.
        # Resolve to 1 if empty, will result in one batch sample of default values.
        batch_size = (
            next(iter(overwrite.values())).shape[0] if overwrite else batch_size or 1
        )

        # Create a copy of the learnable array as default values
        batch_parameter_values = np.tile(
            self.learnable_array.copy().reshape(1, -1),
            (batch_size, 1),
        )

        # Overwrite the values in the batch for specified parameters
        for key, val in overwrite.items():
            if key not in self.learnable_params:
                raise KeyError(f"Parameter '{key}' is not learnable or not found.")

            param_info = self.learnable_params[key]
            start_idx = param_info["start_idx"]
            end_idx = param_info["end_idx"]

            # Reshape the input values to match the parameter size
            val_reshaped = val.reshape(batch_size, -1)
            if val_reshaped.shape[1] != (end_idx - start_idx):
                raise ValueError(
                    f"Shape mismatch for parameter '{key}': expected "
                    f"{end_idx - start_idx} values, got {val_reshaped.shape[1]}"
                )

            batch_parameter_values[:, start_idx:end_idx] = val_reshaped

        expected_shape = (batch_size, len(self.learnable_array))
        assert batch_parameter_values.shape == expected_shape, (
            f"batch_parameter_values should have shape {expected_shape}, "
            f"got {batch_parameter_values.shape}."
        )

        return batch_parameter_values

    def learnable_params_lower_bound(self) -> np.ndarray:
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

    def learnable_params_upper_bound(self) -> np.ndarray:
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
