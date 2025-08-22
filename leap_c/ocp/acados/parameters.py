from typing import NamedTuple, Literal

import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from casadi.tools import entry, struct, struct_symSX


class Parameter(NamedTuple):
    """
    High-level parameter class for flexible optimization parameter configuration.

    This class provides a user-friendly interface for defining parameter sets without
    requiring knowledge of internal CasADi tools or acados interface details. It supports
    configurable properties for bounds, differentiability, and parameter behavior.

    Attributes:
        name: The name identifier for the parameter.
        value: The parameter's numerical value(s).
        lower_bound: Lower bounds for the parameter values.
            Defaults to None (unbounded).
        upper_bound: Upper bounds for the parameter values.
            Defaults to None (unbounded).
        fix: Flag indicating if this is a fixed value rather than a
            settable parameter. Defaults to True.
        differentiable: Flag indicating if the parameter should be
            treated as differentiable in optimization. Defaults to False.
        stagewise: Flag indicating if the parameter varies across
            optimization stages. Defaults to False.

    Note:
        TODO: Check about infinity bounds implementation in lower_bound and upper_bound.
    """

    name: str
    value: np.ndarray
    lower_bound: np.ndarray | None = None
    upper_bound: np.ndarray | None = None
    interface: Literal["fix", "learnable", "non-learnable"] = "fix"
    # Indicates the stages after which the parameter varies. Only used for "learnable".
    vary_stages: list[int] = []


class AcadosParamManager:
    """Manager for acados parameters."""

    parameters: dict[str, Parameter] = {}
    learnable_parameters: struct_symSX | None = None
    learnable_parameter_values: struct | None = None
    non_learnable_parameters: struct_symSX | None = None
    non_learnable_parameter_values: list[struct] | None = None

    def __init__(
        self,
        parameters: list[Parameter],
        N_horizon: int,
    ) -> None:
        self.parameters = {param.name: param for param in parameters}

        self.N_horizon = N_horizon

        need_indicator = False
        entries = {
            "learnable": [],
            "non-learnable": [],
        }

        def _add_parameter_entries(
            name: str, parameter: Parameter, interface_type: str
        ):
            nonlocal need_indicator
            if parameter.vary_stages:
                need_indicator = True
                # Clip vary_stages to the horizon
                vary_stages = [
                    stage for stage in parameter.vary_stages if stage <= self.N_horizon
                ]
                starts = [0] + vary_stages
                ends = np.array(vary_stages + [self.N_horizon + 1]) - 1
                for start, end in zip(starts, ends):
                    # Build symbolic expressions for each stage
                    # following the template {name}_{first_stage}_{last_stage}
                    # e.g. price_0_10, price_11_20, etc.
                    entries[interface_type].append(
                        entry(
                            f"{name}_{start}_{end}",
                            shape=parameter.value.shape,
                        )
                    )
            else:
                entries[interface_type].append(entry(name, shape=parameter.value.shape))

        for name, parameter in self.parameters.items():
            if parameter.interface == "learnable":
                _add_parameter_entries(name, parameter, "learnable")
            if parameter.interface == "non-learnable":
                _add_parameter_entries(name, parameter, "non-learnable")

        if need_indicator:
            entries["non-learnable"].append(
                entry("indicator", shape=(self.N_horizon + 1,))
            )

        self.learnable_parameters = struct_symSX(entries["learnable"])
        self.non_learnable_parameters = struct_symSX(entries["non-learnable"])

        # Now build the lower and upper bound
        self.learnable_parameters_default = self.learnable_parameters(0)
        self.learnable_parameters_lb = self.learnable_parameters(0)
        self.learnable_parameters_ub = self.learnable_parameters(0)
        self.non_learnable_parameters_default = self.non_learnable_parameters(0)

        def _extract_parameter_name(key: str) -> str:
            """Extract the original parameter name from the template {name}_{start}_{end}."""
            if "_" in key and key.count("_") >= 2:
                # Split from the right to handle names that contain underscores
                parts = key.rsplit("_", 2)
                return parts[0]
            else:
                # For parameters without stage variations
                return key

        def _fill_parameter_values(struct_dict, keys, include_bounds: bool = False):
            """Fill parameter values and optionally bounds for a parameter structure."""
            for key in keys:
                # First check if the key exists directly in parameters (no staging)
                if key in self.parameters:
                    name = key
                else:
                    # Try to extract the original parameter name from staged key
                    name = _extract_parameter_name(key)
                else:
                    name = key
                if name in self.parameters:
                    param = self.parameters[name]
                    struct_dict["default"][key] = param.value
                    if include_bounds:
                        if param.lower_bound is not None:
                            struct_dict["lb"][key] = param.lower_bound
                        if param.upper_bound is not None:
                            struct_dict["ub"][key] = param.upper_bound

        # Fill in the values for learnable parameters (with bounds)
        _fill_parameter_values(
            {
                "default": self.learnable_parameters_default,
                "lb": self.learnable_parameters_lb,
                "ub": self.learnable_parameters_ub,
            },
            self.learnable_parameters.keys(),
            include_bounds=True,
        )

        # Fill in the values for non-learnable parameters (no bounds)
        _fill_parameter_values(
            {"default": self.non_learnable_parameters_default},
            self.non_learnable_parameters.keys(),
            include_bounds=False,
        )

    # This is for non_learnable_parameters.
    # TODO: Modify name after PR from example cleanup
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
            self.non_learnable_parameter_values.cat.full().reshape(1, -1),
            (batch_size, self.N_horizon + 1, 1),
        )

        # Set indicator for each stage
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

    # This is for learnable_parameters.
    # TODO: Modify name after PR from example cleanup
    def get_p_global_bounds(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Get the lower bound for p_global parameters."""
        if self.learnable_parameters is None:
            return None, None

        # return self.lb, self.ub
        return (
            self.learnable_parameters_lb.cat.full(),
            self.learnable_parameters_ub.cat.full(),
        )

    def get(
        self,
        field: str,
        stage: int | None = None,
    ) -> ca.SX | ca.MX | np.ndarray:
        """Get the variable for a given field at a specific stage."""
        if field not in self.parameters:
            raise ValueError(
                f"Unknown field: {field}. Available fields: {list(self.parameters.keys())}"
            )

        if self.parameters[field].interface == "fix":
            return self.parameters[field].value

        if (
            self.parameters[field].interface in ["learnable", "non-learnable"]
            and self.parameters[field].vary_stages
        ):
            starts = [0] + self.parameters[field].vary_stages
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

        if stage is not None and field == "indicator":
            return self.non_learnable_parameters[field][stage]

        if self.parameters[field].interface == "non-learnable":
            return self.non_learnable_parameters[field]

    def assign_to_ocp(self, ocp: AcadosOcp) -> None:
        """Assign the parameters to the OCP model."""
        if self.learnable_parameters is not None:
            ocp.model.p_global = self.learnable_parameters.cat
            ocp.p_global_values = (
                self.learnable_parameter_values.cat.full().flatten()
                if self.learnable_parameter_values
                else np.array([])
            )

        if self.non_learnable_parameters is not None:
            ocp.model.p = self.non_learnable_parameters.cat
            ocp.parameter_values = (
                self.non_learnable_parameter_values.cat.full().flatten()
                if self.non_learnable_parameter_values
                else np.array([])
            )
