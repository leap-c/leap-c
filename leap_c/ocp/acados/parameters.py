from dataclasses import dataclass, field
from typing import Any, Collection, Iterable, Literal
from warnings import warn

import casadi as ca
import gymnasium as gym
import numpy as np
import torch
from acados_template import AcadosOcp
from casadi.tools import entry, struct, struct_symMX, struct_symSX


@dataclass
class AcadosParameter:
    """High-level parameter class for flexible optimization configuration with acados extensions.

    It provides an interface for defining parameter sets without requiring knowledge of
    internal CasADi tools or acados interface details.

    Attributes:
        name: The name identifier for the parameter.
        default: The parameter's default numerical value(s).
        space: A gym.spaces.Space defining the valid parameter space.
            Only used for learnable parameters. Defaults to `None` (unbounded).
        interface: Parameter interface type.
            Either `"fix"` (fixed values, unchangeable after creation of the solver),
            `"non-learnable"` (not exposed to the learning interface, but will be changeable
            parameters also after creation of the solver), or `"learnable"` (parameters directly
            exposed to the learning interface, in particular supporting sensitivities). Defaults to
            `"fix"`.
        end_stages: Sorted list (ascending order) of stages after which the parameter varies.
            Only used for the `"learnable"` interface. If `None`, the parameter remains constant
            across all stages. Defaults to `None`.
            Example: If the horizon has `9` stages (`0` to `9`, including the terminal stage),
            and `end_stages = [4, 9]`, then the parameter will have one value for stages `0` to `4`,
            and a different value for stages `5` to `9`.
    """

    # Fields from base Parameter class
    name: str
    default: np.ndarray
    space: gym.spaces.Space | None = None
    interface: Literal["fix", "learnable", "non-learnable"] = "fix"
    # Additional acados-specific field
    end_stages: list[int] = field(default_factory=list)


class AcadosParameterManager:
    """Manager class for handling acados parameters according to their specifications.

    In particular, this handles stage-varying learnable parameters, which is not available
    out-of-the-box in acados.

    Basic usage inside the AcadosOcp formulation:
        Step 1: Create a list of AcadosParameter objects that define the Parameters you want to use.
        Step 2: Use `param = param_manager.get(name)` to retrieve parameters with a certain name.
        Use these variables in the ocp formulation to define costs, dynamics, etc.
        Step 3: Use `param_manager.assign_to_ocp(ocp)` to assign the parameters to the ocp object.

    Attributes:
        parameters: Dictionary of parameter names to AcadosParameter instances.
        learnable_parameters: CasADi struct of learnable parameters.
        learnable_parameters_default: Default values for learnable parameters.
        learnable_parameters_lb: Lower bounds for learnable parameters.
        learnable_parameters_ub: Upper bounds for learnable parameters.
        non_learnable_parameters: CasADi struct of non-learnable parameters.
        non_learnable_parameters_default: Default values for non-learnable parameters.
        N_horizon: The horizon length for the ocp.
        need_indicator: Whether indicator variables exist (for controlling stage-varying learnable
            parameters).
    """

    parameters: dict[str, AcadosParameter]
    learnable_parameters: struct_symSX | struct_symMX
    learnable_parameters_default: struct
    learnable_parameters_lb: struct
    learnable_parameters_ub: struct
    non_learnable_parameters: struct_symSX | struct_symMX
    non_learnable_parameters_default: struct
    N_horizon: int
    need_indicator: bool

    def __init__(
        self,
        parameters: Collection[AcadosParameter],
        N_horizon: int,
        casadi_type: Literal["SX", "MX"] = "SX",
    ) -> None:
        # add parameters to the manager
        if not parameters:
            warn(
                "Empty parameter list provided to AcadosParamManager. "
                "Consider adding parameters for building a parametric AcadosOcp.",
                UserWarning,
                stacklevel=2,
            )
        # validate parameter dimensions before storing
        for param in parameters:
            if param.default.ndim > 2:
                raise ValueError(
                    f"Parameter '{param.name}' has {param.default.ndim} dimensions, "
                    f"but CasADi only supports arrays up to 2 dimensions. "
                    f"Parameter shape: {param.default.shape}"
                )
            if isinstance(param.space, gym.spaces.Box):
                if len(param.space.shape) > 2:
                    raise ValueError(
                        f"Parameter '{param.name}' space has {len(param.space.shape)} dimensions, "
                        f"but CasADi only supports arrays up to 2 dimensions. "
                        f"Space shape: {param.space.shape}"
                    )
            elif param.space is None:
                pass
            else:
                raise NotImplementedError(
                    f"Parameter '{param.name}' has space of type {type(param.space)}, "
                    "but currently only gym.spaces.Box is supported."
                )

            # Check end_stages convention
            if param.end_stages and param.end_stages[-1] not in [N_horizon - 1, N_horizon]:
                raise ValueError(
                    f"Parameter '{param.name}' has end_stages {param.end_stages} "
                    f"but the last element must be either {N_horizon - 1} or {N_horizon}."
                )
        self.parameters = {param.name: param for param in parameters}

        self.N_horizon = N_horizon

        entries = {"learnable": [], "non-learnable": []}

        def _add_learnable_parameter_entries(name: str, parameter: AcadosParameter) -> None:
            interface_type = "learnable"
            if parameter.end_stages:
                self.need_indicator = True
                starts, ends = _define_starts_and_ends(
                    end_stages=parameter.end_stages, N_horizon=self.N_horizon
                )
                for start, end in zip(starts, ends):
                    # Build symbolic expressions for each stage
                    # following the template {name}_{first_stage}_{last_stage}
                    # e.g. price_0_10, price_11_20, etc.
                    entries[interface_type].append(
                        entry(
                            f"{name}_{start}_{end}",
                            shape=parameter.default.shape,
                        )
                    )
            else:
                entries[interface_type].append(entry(name, shape=parameter.default.shape))

        def _add_non_learnable_parameter_entries(name: str, parameter: AcadosParameter) -> None:
            interface_type = "non-learnable"
            # Non-learnable parameters are by construction for each stage
            entries[interface_type].append(entry(name, shape=parameter.default.shape))

        self.need_indicator = False
        for name, parameter in self.parameters.items():
            if parameter.interface == "learnable":
                _add_learnable_parameter_entries(name, parameter)
            if parameter.interface == "non-learnable":
                _add_non_learnable_parameter_entries(name, parameter)

        if self.need_indicator:
            entries["non-learnable"].append(entry("indicator", shape=(self.N_horizon + 1,)))

        if casadi_type == "SX":
            self.learnable_parameters = struct_symSX(entries["learnable"])
            self.non_learnable_parameters = struct_symSX(entries["non-learnable"])
        elif casadi_type == "MX":
            self.learnable_parameters = struct_symMX(entries["learnable"])
            self.non_learnable_parameters = struct_symMX(entries["non-learnable"])

        # Now build the lower and upper bound
        self.learnable_parameters_default = self.learnable_parameters(0)
        self.learnable_parameters_lb = self.learnable_parameters(0)
        self.learnable_parameters_ub = self.learnable_parameters(0)

        def _extract_parameter_name(key: str) -> str:
            """Extract the original parameter name from the template {name}_{start}_{end}."""
            if "_" in key and key.count("_") >= 2:
                # Split from the right to handle names that contain underscores
                parts = key.rsplit("_", 2)
                return parts[0]
            else:
                # For parameters without stage variations
                return key

        def _fill_learnable_parameter_values(
            struct_dict: dict[str, struct], keys: Iterable[str]
        ) -> None:
            """Fill parameter values and optionally bounds for a parameter structure."""
            for key in keys:
                # First check if the key exists directly in parameters (no staging)
                if key in self.parameters:
                    name = key
                else:
                    # Try to extract the original parameter name from staged key
                    name = _extract_parameter_name(key)

                if name in self.parameters:
                    param = self.parameters[name]
                    struct_dict["default"][key] = param.default
                    if param.space is not None and hasattr(param.space, "low"):
                        struct_dict["lb"][key] = param.space.low
                    else:
                        struct_dict["lb"][key] = -np.inf
                    if param.space is not None and hasattr(param.space, "high"):
                        struct_dict["ub"][key] = param.space.high
                    else:
                        struct_dict["ub"][key] = np.inf

        # Fill in the values for learnable parameters (with bounds)
        _fill_learnable_parameter_values(
            {
                "default": self.learnable_parameters_default,
                "lb": self.learnable_parameters_lb,
                "ub": self.learnable_parameters_ub,
            },
            self.learnable_parameters.keys(),
        )

        # Fill the values for the non-learnable parameters.
        # Indicators are set at combine_parameter_values.
        self.non_learnable_parameters_default = self.non_learnable_parameters(0)
        for key in self.parameters.keys():
            if self.parameters[key].interface == "non-learnable":
                self.non_learnable_parameters_default[key] = self.parameters[key].default

    def combine_default_learnable_parameter_values(
        self, batch_size: int | None = None, **overwrites: np.ndarray
    ) -> np.ndarray:
        """Combine all learnable parameters and provided overwrites into a single numpy array.

        This can be used to create a batch of default learnable parameter values. An example would
        be to load forecasts into the ocp, which depend on the current observation.

        Args:
            batch_size: The batch size for the parameters.
                Not needed if overwrites is provided.
            **overwrites: Overwrite values for specific parameters.
                The keys should correspond to the parameter names to overwrite.
                For stage-varying parameters (those with end_stages), the values need to be
                np.ndarray with shape `(batch_size, N_horizon + 1)` or
                `(batch_size, N_horizon + 1, pdim)`.
                For non-stage-varying parameters, shape `(batch_size,)` or `(batch_size, pdim)`.

        Returns:
            np.ndarray: shape `(batch_size, N_learnable)` with `N_learnable` being the total
            number of learnable parameter values.
        """
        # Infer batch size from overwrites if not provided
        inferred_batch_size = (
            next(iter(overwrites.values())).shape[0] if overwrites else None
        )
        
        # Validate batch_size consistency
        if batch_size is not None and inferred_batch_size is not None:
            if batch_size != inferred_batch_size:
                raise ValueError(
                    f"Provided batch_size={batch_size} does not match "
                    f"inferred batch_size={inferred_batch_size} from overwrites."
                )
        
        batch_size = inferred_batch_size if inferred_batch_size is not None else batch_size or 1

        # Get default parameter array and tile for batch
        default_flat = self.learnable_parameters_default.cat.full().flatten()
        batch_param = np.tile(default_flat, (batch_size, 1))

        if not overwrites:
            return batch_param

        # Build index mappings for efficient vectorized assignment
        for param_name, values in overwrites.items():
            if param_name not in self.parameters:
                raise ValueError(
                    f"Parameter '{param_name}' not found. "
                    f"Available parameters: {list(self.parameters.keys())}"
                )

            param = self.parameters[param_name]

            if param.interface != "learnable":
                raise ValueError(
                    f"Parameter '{param_name}' has interface '{param.interface}', "
                    "but only 'learnable' parameters can be used in this method."
                )

            # Ensure values has correct batch dimension
            if values.shape[0] != batch_size:
                raise ValueError(
                    f"Parameter '{param_name}' values have batch size {values.shape[0]}, "
                    f"but expected {batch_size}."
                )

            if param.end_stages:
                # Stage-varying parameter
                Np1 = self.N_horizon + 1
                starts, ends = _define_starts_and_ends(
                    end_stages=param.end_stages, N_horizon=self.N_horizon
                )

                # Expected shape: (batch_size, N_horizon + 1) or (batch_size, N_horizon + 1, pdim)
                if values.shape[1] != Np1:
                    raise ValueError(
                        f"Parameter '{param_name}' is stage-varying and requires shape "
                        f"(batch_size, {Np1}, ...), but got shape {values.shape}."
                    )

                # Reshape to (batch_size, N_horizon + 1, -1) for consistent handling
                values_reshaped = values.reshape(batch_size, Np1, -1)

                # Assign each block separately (blocks share the same parameter value)
                for start, end in zip(starts, ends):
                    param_key = f"{param_name}_{start}_{end}"
                    try:
                        param_idx = self.learnable_parameters.f[param_key]
                    except KeyError as e:
                        raise KeyError(
                            f"Learnable parameter '{param_key}' not found."
                        ) from e

                    # All stages in this block use the value from the start stage
                    block_value = values_reshaped[:, start, :]
                    # Handle scalar vs vector parameters
                    if isinstance(param_idx, (list, np.ndarray)):
                        # Vector parameter
                        batch_param[:, param_idx] = block_value
                    else:
                        # Scalar parameter
                        if block_value.shape[-1] == 1:
                            batch_param[:, param_idx] = block_value.squeeze(-1)
                        else:
                            batch_param[:, param_idx] = block_value

            else:
                # Non-stage-varying parameter - single value per batch
                param_key = param_name
                try:
                    param_idx = self.learnable_parameters.f[param_key]
                except KeyError as e:
                    raise KeyError(f"Learnable parameter '{param_key}' not found.") from e

                # Reshape to handle both scalar and vector parameters
                values_reshaped = values.reshape(batch_size, -1)
                # Handle the parameter index which might be scalar or vector
                if isinstance(param_idx, (list, np.ndarray)):
                    # Vector parameter
                    batch_param[:, param_idx] = values_reshaped
                else:
                    # Scalar parameter - need to flatten
                    if values_reshaped.shape[-1] == 1:
                        batch_param[:, param_idx] = values_reshaped.squeeze(-1)
                    else:
                        batch_param[:, param_idx] = values_reshaped

        return batch_param

    def combine_non_learnable_parameter_values(
        self, batch_size: int | None = None, **overwrite: np.ndarray
    ) -> np.ndarray:
        """Combine all non-learnable parameters and provided overwrites into a single numpy array.

        Args:
            batch_size: The batch size for the parameters.
                Not needed if overwrite is provided.
            **overwrite: Overwrite values for specific parameters.
                The keys should correspond to the parameter names to overwrite.
                The values need to be np.ndarray with shape `(batch_size, N_horizon, pdim)`,
                where `pdim` is the number of dimensions of the parameter to overwrite.

        Returns:
            np.ndarray: shape `(batch_size, N_horizon, np)`. with `np` being the number of
            `parameter_values`.
        """
        # Infer batch size from overwrite if not provided.
        # Resolve to 1 if empty, will result in one batch sample of default values.
        batch_size = next(iter(overwrite.values())).shape[0] if overwrite else batch_size or 1

        # Create a batch of parameter values
        Np1 = self.N_horizon + 1
        batch_parameter_values = np.tile(
            self.non_learnable_parameters_default.cat.full().reshape(1, -1),
            (batch_size, Np1, 1),
        )

        # Set indicator for each stage
        if self.need_indicator:
            batch_parameter_values[:, :, -Np1:] = np.eye(Np1)

        # Overwrite the values in the batch
        # TODO: Make sure indexing is consistent.
        # Issue is the difference between casadi (row major) and numpy (column major)
        # when using matrix values.
        # NOTE: Can use numpy.reshape with order='C' or order='F'
        # to specify column / row major.
        # NOTE: First check the order, using something like a.flags.f_contiguous,
        # see https://numpy.org/doc/2.1/reference/generated/numpy.isfortran.html
        # and reshape if needed or raise an error.
        for key, val in overwrite.items():
            batch_parameter_values[:, :, self.non_learnable_parameters.f[key]] = val.reshape(
                batch_size, Np1, -1
            )

        expected_shape = (batch_size, Np1, self.non_learnable_parameters.cat.shape[0])
        assert batch_parameter_values.shape == expected_shape, (
            f"batch_parameter_values should have shape {expected_shape}, "
            f"got {batch_parameter_values.shape}."
        )

        return batch_parameter_values

    def get_param_space(
        self, dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32
    ) -> gym.Space:
        """Return the combined Gym space for the learnable parameters.

        If the parameters do not provide a space themselves, an unbounded Box space with type
        `dtype` will be filled in for them.

        For parameters with stage variations (end_stages), the space is duplicated according
        to the number of stage variations.

        Args:
            dtype: The desired data type for the filled-in spaces.
        """
        learnable_spaces = []

        for param in self.parameters.values():
            if param.interface == "learnable":
                # Determine the base space for this parameter
                if param.space is not None:
                    base_space = param.space
                else:
                    # Create unbounded space for parameters without bounds
                    param_shape = param.default.shape
                    base_space = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=param_shape,
                        dtype=dtype,
                    )

                # If parameter has stage variations, duplicate the space for each variation
                if param.end_stages:
                    starts, ends = _define_starts_and_ends(
                        end_stages=param.end_stages, N_horizon=self.N_horizon
                    )
                    # Add one space for each stage variation
                    for _ in zip(starts, ends):
                        learnable_spaces.append(base_space)
                else:
                    # No stage variations, add single space
                    learnable_spaces.append(base_space)

        if not learnable_spaces:
            # No learnable parameters - return empty box space
            return gym.spaces.Box(low=np.empty(0, dtype), high=np.empty(0, dtype), dtype=dtype)
        elif len(learnable_spaces) == 1:
            # Single space - flatten it to ensure consistent return type
            return learnable_spaces[0]
        else:
            # Multiple spaces
            tuple_space = gym.spaces.Tuple(learnable_spaces)
            return gym.spaces.utils.flatten_space(tuple_space)
        

    def get(self, name: str) -> ca.SX | ca.MX | np.ndarray:
        """Get the variable of a given name.

        Args:
            name: The name of the parameter to retrieve.

        Returns:
            A casadi variable for the parameter, or its default value if fixed.
        """
        if name not in self.parameters:
            raise ValueError(f"Unknown name: {name}. Available names: {', '.join(self.parameters)}")

        if self.parameters[name].interface == "fix":
            return self.parameters[name].default

        if self.parameters[name].interface == "learnable" and self.parameters[name].end_stages:
            starts, ends = _define_starts_and_ends(
                end_stages=self.parameters[name].end_stages, N_horizon=self.N_horizon
            )
            indicators = []
            variables = []
            for start, end in zip(starts, ends):
                indicators.append(
                    ca.sum(self.non_learnable_parameters["indicator"][start : end + 1])
                )
                variables.append(self.learnable_parameters[f"{name}_{start}_{end}"])

            terms = []
            for indicator, variable in zip(indicators, variables):
                terms.append(indicator * variable)
            return sum(terms)

        if self.parameters[name].interface == "learnable":
            return self.learnable_parameters[name]

        if self.parameters[name].interface == "non-learnable":
            return self.non_learnable_parameters[name]
        else:
            raise ValueError(
                f"Unknown interface type for field '{name}': {self.parameters[name].interface}"
            )

    def assign_to_ocp(self, ocp: AcadosOcp) -> None:
        """Assign the parameters to the acados ocp object.

        NOTE: Overwrites any existing parameter definitions in the ocp object.
        """
        if self.learnable_parameters is not None:
            ocp.model.p_global = self.learnable_parameters.cat
            ocp.p_global_values = (
                self.learnable_parameters_default.cat.full().flatten()
                if self.learnable_parameters_default
                else np.array([])
            )

        if self.non_learnable_parameters is not None:
            ocp.model.p = self.non_learnable_parameters.cat
            ocp.parameter_values = (
                self.non_learnable_parameters_default.cat.full().flatten()
                if self.non_learnable_parameters_default
                else np.array([])
            )

    def recreate_dataclass(self, cls):
        """Recreate a dataclass instance of type cls with current parameter values.

        Args:
            cls: The dataclass type to recreate.

        Returns:
            An instance of cls with fields populated from the current parameter values.
        """
        field_values = {}
        for field_name in cls.__dataclass_fields__.keys():
            if field_name in self.parameters:
                field_values[field_name] = self.get(field_name)
            else:
                raise ValueError(f"Parameter '{field_name}' not found in parameter manager.")
        return cls(**field_values)

    def has_learnable_param_pattern(self, pattern: str) -> bool:
        """Check if any parameter names match the given pattern.

        Supports glob-style wildcards where '*' matches any characters.
        For example, 'temperature_*_*' matches 'temperature_0_0', 'temperature_1_1', etc.

        Args:
            pattern: Pattern string with wildcards (*) to match against parameter names.

        Returns:
            True if any learnable parameter names match the pattern, False otherwise.

        Example:
            >>> planner.has_param_pattern('temperature_*_*')
            True  # if parameters like temperature_0_0, temperature_1_1, etc. exist
            >>> planner.has_param_pattern('nonexistent_*')
            False
        """
        import fnmatch

        learnable_param_names = self.learnable_parameters.keys()
        return any(fnmatch.fnmatch(name, pattern) for name in learnable_param_names)

    def get_labeled_learnable_parameters(
        self,
        param_values: np.ndarray | torch.Tensor,
        label: str,
    ) -> np.ndarray | torch.Tensor:
        """Get a structured representation of the learnable parameters from flat values.

        Args:
            param_values: Flat numpy array of learnable parameter values.
            label: Substring to filter parameters by name.

        Returns:
            A numpy array or torch Tensor corresponding to the parameters
            matching the label.
        """
        if label is None:
            raise ValueError("Label must be provided to filter learnable parameters.")

        keys = [key for key in self.learnable_parameters.keys() if label in key]

        if keys == []:
            raise ValueError(f"No learnable parameters found with label '{label}'.")

        idx = [self.learnable_parameters.f[key] for key in keys]
        return param_values[..., idx].reshape(-1, len(keys))


def _define_starts_and_ends(end_stages: list[int], N_horizon: int) -> tuple[list[int], list[int]]:
    """Define the start and end indices for stage-varying parameters."""
    ends = end_stages
    starts = [0] + [v + 1 for v in ends if v + 1 <= N_horizon]
    return starts, ends
