from dataclasses import dataclass, field
from typing import Literal
import warnings

import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from casadi.tools import entry, struct, struct_symSX
import gymnasium as gym
from leap_c.parameters import Parameter as BaseParameter


@dataclass
class AcadosParameter:
    """
    High-level parameter class for flexible optimization parameter configuration with acados extensions.

    This class extends the base Parameter functionality with acados-specific features like vary_stages.
    It provides a user-friendly interface for defining parameter sets without
    requiring knowledge of internal CasADi tools or acados interface details. It supports
    configurable properties for bounds, differentiability, and parameter behavior.

    Attributes:
        name: The name identifier for the parameter.
        default: The parameter's default numerical value(s).
        space: A gym.spaces.Space defining the valid parameter space.
            Only used for learnable parameters. Defaults to None (unbounded).
        interface: Parameter interface type. Either "fix" (fixed values),
            "learnable" (optimizable parameters), or "non-learnable"
            (non-optimizable but changeable). Defaults to "fix".
        vary_stages: List of stages after which the parameter varies.
            Only used for "learnable" interface. If None, parameter
            remains constant across all stages. Defaults to None.
    """

    # Fields from base Parameter class
    name: str
    default: np.ndarray
    space: gym.spaces.Space | None = None
    interface: Literal["fix", "learnable", "non-learnable"] = "fix"
    # Additional acados-specific field
    vary_stages: list[int] = field(default_factory=list)

    @classmethod
    def from_base_parameter(
        cls, base_param: BaseParameter, vary_stages: list[int] = None
    ):
        """Create an acados Parameter from a base Parameter."""
        return cls(
            name=base_param.name,
            default=base_param.default,
            space=base_param.space,
            interface=base_param.interface,
            vary_stages=vary_stages or [],
        )

    def to_base_parameter(self) -> BaseParameter:
        """Convert to base Parameter (loses vary_stages information)."""
        return BaseParameter(
            name=self.name,
            default=self.default,
            space=self.space,
            interface=self.interface,
        )


class AcadosParameterManager:
    """Manager for acados parameters."""

    parameters: dict[str, AcadosParameter]
    learnable_parameters: struct_symSX
    learnable_parameters_default: struct
    non_learnable_parameters: struct_symSX
    non_learnable_parameters_default: list[struct]

    def __init__(
        self,
        parameters: list[AcadosParameter],
        N_horizon: int,
    ) -> None:
        if not parameters:
            warnings.warn(
                "Empty parameter list provided to AcadosParamManager. "
                "Consider adding parameters for building a parametric AcadosOcp.",
                UserWarning,
                stacklevel=2,
            )

        # Validate parameter dimensions before storing
        for param in parameters:
            if param.default.ndim > 2:
                raise ValueError(
                    f"Parameter '{param.name}' has {param.default.ndim} dimensions, "
                    f"but CasADi only supports arrays up to 2 dimensions. "
                    f"Parameter shape: {param.default.shape}"
                )
            if (
                param.space is not None
                and hasattr(param.space, "low")
                and param.space.low.ndim > 2
            ):
                raise ValueError(
                    f"Parameter '{param.name}' space.low has {param.space.low.ndim} dimensions, "
                    f"but CasADi only supports arrays up to 2 dimensions. "
                    f"Lower bound shape: {param.space.low.shape}"
                )
            if (
                param.space is not None
                and hasattr(param.space, "high")
                and param.space.high.ndim > 2
            ):
                raise ValueError(
                    f"Parameter '{param.name}' space.high has {param.space.high.ndim} dimensions, "
                    f"but CasADi only supports arrays up to 2 dimensions. "
                    f"Upper bound shape: {param.space.high.shape}"
                )
            if param.vary_stages and param.vary_stages[-1] > N_horizon:
                raise ValueError(
                    f"Parameter '{param.name}' has vary_stages {param.vary_stages} "
                    f"which exceed the horizon length {N_horizon}."
                )

        self.parameters = {param.name: param for param in parameters}

        self.N_horizon = N_horizon

        entries = {
            "learnable": [],
            "non-learnable": [],
        }

        def _add_learnable_parameter_entries(name: str, parameter: AcadosParameter):
            interface_type = "learnable"
            if parameter.vary_stages:
                self.need_indicator = True
                # Clip vary_stages to the horizon
                vary_stages = parameter.vary_stages
                starts = [0] + vary_stages
                ends = np.array(vary_stages + [self.N_horizon + 1]) - 1
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
                entries[interface_type].append(
                    entry(name, shape=parameter.default.shape)
                )

        def _add_non_learnable_parameter_entries(name: str, parameter: AcadosParameter):
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
            entries["non-learnable"].append(
                entry("indicator", shape=(self.N_horizon + 1,))
            )

        self.learnable_parameters = struct_symSX(entries["learnable"])
        self.non_learnable_parameters = struct_symSX(entries["non-learnable"])

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

        def _fill_learnable_parameter_values(struct_dict, keys):
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

        # Fill the values for the non-learnable parameters. Indicators are set at combine_parameter_values.
        self.non_learnable_parameters_default = self.non_learnable_parameters(0)
        for key in self.parameters.keys():
            if self.parameters[key].interface == "non-learnable":
                self.non_learnable_parameters_default[key] = self.parameters[
                    key
                ].default

    def combine_non_learnable_parameter_values(
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
                values need to be np.ndarray with shape (batch_size, N_horizon, ...).

        Returns:
            np.ndarray: shape (batch_size, N_horizon, np). with np being the number of
            parameter_values.
        """
        # Infer batch size from overwrite if not provided.
        # Resolve to 1 if empty, will result in one batch sample of default values.
        batch_size = (
            next(iter(overwrite.values())).shape[0] if overwrite else batch_size or 1
        )

        # Create a batch of parameter values
        batch_parameter_values = np.tile(
            self.non_learnable_parameters_default.cat.full().reshape(1, -1),
            (batch_size, self.N_horizon + 1, 1),
        )

        # Set indicator for each stage
        if self.need_indicator:
            batch_parameter_values[:, :, -(self.N_horizon + 1) :] = np.tile(
                np.eye(self.N_horizon + 1),
                (batch_size, 1, 1),
            )

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
            batch_parameter_values[:, :, self.non_learnable_parameters.f[key]] = (
                val.reshape(batch_size, self.N_horizon + 1, -1)
            )

        expected_shape = (
            batch_size,
            self.N_horizon + 1,
            self.non_learnable_parameters.cat.shape[0],
        )
        assert batch_parameter_values.shape == expected_shape, (
            f"batch_parameter_values should have shape {expected_shape}, "
            f"got {batch_parameter_values.shape}."
        )

        return batch_parameter_values

    def get_param_space(self, dtype: np.dtype = np.float32) -> gym.Space:
        """Get the Gym space for the parameters."""
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
                        dtype=dtype,
                    )
                    learnable_spaces.append(unbounded_space)

        if not learnable_spaces:
            # No learnable parameters - return empty box space
            return gym.spaces.Box(low=np.array([]), high=np.array([]), dtype=dtype)
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

    def get(self, field: str) -> ca.SX | ca.MX | np.ndarray:
        """Get the variable for a given field."""
        if field not in self.parameters:
            raise ValueError(
                f"Unknown field: {field}. Available fields: {list(self.parameters.keys())}"
            )

        if self.parameters[field].interface == "fix":
            return self.parameters[field].default

        if (
            self.parameters[field].interface == "learnable"
            and self.parameters[field].vary_stages
        ):
            starts = (
                [0] if 0 not in self.parameters[field].vary_stages else []
            ) + self.parameters[field].vary_stages
            ends = (
                np.array(self.parameters[field].vary_stages + [self.N_horizon + 1]) - 1
            )
            indicators = []
            variables = []
            for start, end in zip(starts, ends):
                indicators.append(
                    ca.sum(self.non_learnable_parameters["indicator"][start : end + 1])
                )
                variables.append(self.learnable_parameters[f"{field}_{start}_{end}"])

            terms = []
            for indicator, variable in zip(indicators, variables):
                terms.append(indicator * variable)
            return sum(terms)

        if self.parameters[field].interface == "learnable":
            return self.learnable_parameters[field]

        if self.parameters[field].interface == "non-learnable":
            return self.non_learnable_parameters[field]

    def assign_to_ocp(self, ocp: AcadosOcp) -> None:
        """Assign the parameters to the OCP model."""
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
