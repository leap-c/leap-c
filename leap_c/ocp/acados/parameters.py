import contextlib
from typing import ClassVar, NamedTuple

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi.tools import entry, struct, struct_symSX


class Parameter(NamedTuple):
    name: str
    value: np.ndarray
    lower_bound: np.ndarray | None
    upper_bound: np.ndarray | None
    differentiable: bool
    varying: bool


class AcadosParamManager:
    """Manager for Acados parameters."""

    parameters: ClassVar[dict[str, Parameter]] = {}
    p_global: struct_symSX | None = None
    p_global_values: struct | None = None
    p: struct_symSX | None = None
    parameter_values: list[struct] | None = None

    def __init__(self, N_horizon: int) -> None:
        self.N_horizon = N_horizon

    def reset(self) -> None:
        """Reset the parameter manager."""
        self.parameters = {}
        self.p_global = None
        self.p_global_values = None
        self.p = None
        self.parameter_values = None

    def add(
        self,
        parameter: Parameter,
    ) -> None:
        """Add a parameter to the manager."""
        assert isinstance(parameter, Parameter), (
            "Parameter must be an instance of the Parameter NamedTuple."
        )
        self.parameters[parameter.name] = parameter

        # TODO: Consider not updating p and p_global every time a parameter is added.
        self.update_p()
        self.update_p_global()

    def initialize_parameter_values(self) -> None:
        self.parameter_values = [self.p(0) for _ in range(self.N_horizon)]

        indicators = np.eye(self.N_horizon)

        for stage in range(self.N_horizon):
            for key, value in self.get_nondifferentiable_parameters().items():
                self.parameter_values[stage][key] = value

            if "indicator" in self.parameter_values[stage].keys():  # noqa: SIM118
                self.parameter_values[stage]["indicator"] = indicators[stage, :]

    def initialize_p_global_values(self) -> None:
        self.p_global_values = self.p_global(0)

        # Set the values for the symbolic structure
        for key, value in self.get_differentiable_constant_parameters().items():
            self.p_global_values[key] = value

        for key, value in self.get_differentiable_varying_parameters().items():
            for stage in range(self.N_horizon):
                self.p_global_values[key, stage] = value

    def set_parameter_value(self, stage_: int, field_: str, value_: np.ndarray) -> None:
        """Set the value for a given field at a specific stage."""
        self.parameter_values[stage_][field_] = value_

    def set_p_global_value(
        self, stage_: int | None, field_: str, value_: np.ndarray
    ) -> None:
        """Set the global value for a given field at a specific stage."""
        self.p_global_values[field_, stage_] = value_

    def get_parameter_symbol(self, field_: str, stage_: int) -> ca.SX:
        """Get the symbolic variable for a given field at a specific stage."""
        return self.parameter_values(stage_)[field_]

    def get_parameter_value(self, stage_: int, field_: str) -> np.ndarray:
        """Get the value for a given field at a specific stage."""
        raise NotImplementedError

    def get_p_global_symbol(self, field_: str, stage_: int | None = None) -> ca.SX:
        """Get the symbolic variable for a given field at a specific stage."""
        raise NotImplementedError

    def get_p_global_value(self, field_: str, stage_: int | None = None) -> np.ndarray:
        """Get the global value for a given field at a specific stage."""
        raise NotImplementedError

    def set(
        self,
        field_: str,
        value_: np.ndarray,
        stage_: int | None = None,
    ) -> None:
        """Set the value for a given field at a specific stage."""
        # TODO: Add error handling for invalid combinations of field_, _stage_, and value_
        if field_ in self.p_global_values.keys():  # noqa: SIM118
            if stage_ is not None:
                self.p_global_values[field_, stage_] = value_
            else:
                self.p_global_values[field_] = value_

        if field_ in self.parameter_values[stage_].keys():  # noqa: SIM118
            self.parameter_values[stage_][field_] = value_

    def get(
        self,
        field_: str,
        stage_: int | None = None,
    ) -> np.ndarray:
        """Get the value for a given field at a specific stage."""
        # TODO: Add error handling for invalid combinations of field_, _stage_, and value_
        if field_ in self.p_global_values.keys():  # noqa: SIM118
            if stage_ is not None:
                return self.p_global_values[field_, stage_]
            return self.p_global_values[field_]

        if field_ in self.parameter_values[stage_].keys():  # noqa: SIM118
            return self.parameter_values[stage_][field_, stage_]

        available_fields = list(self.p_global_values.keys()) + list(
            self.parameter_values.keys()
        )
        error_message = f"Unknown field: {field_}. Available fields: {available_fields}"
        raise ValueError(error_message)

    def get_sym(
        self,
        field_: str,
        stage_: int | None = None,
    ) -> ca.SX:
        """Get the symbolic variable for a given field at a specific stage."""
        if field_ in self.p_global.keys():  # noqa: SIM118
            if stage_ is not None:
                return self.p_global[field_, stage_]
            return self.p_global[field_]

        if field_ in self.p.keys():  # noqa: SIM118
            if stage_ is not None and field_ == "indicator":
                return self.p[field_][stage_]
            return self.p[field_]

        available_fields = list(self.p_global.keys()) + list(self.p.keys())
        error_message = f"Unknown field: {field_}. Available fields: {available_fields}"
        raise ValueError(error_message)

    def update_p(self) -> struct_symSX:
        """Update the structured parameter p."""
        entries = []
        for key, value in self.get_nondifferentiable_parameters().items():
            entries.append(entry(key, shape=value.shape))

        if self.get_differentiable_varying_parameters():
            entries.append(entry("indicator", shape=(self.N_horizon,)))

        self.p = struct_symSX(entries)

    def update_p_global(self) -> struct_symSX:
        """Update the structured parameter p_global."""
        entries = []
        for key, value in self.get_differentiable_constant_parameters().items():
            entries.append(entry(key, shape=value.shape))

        for key, value in self.get_differentiable_varying_parameters().items():
            entries.append(entry(key, shape=value.shape, repeat=self.N_horizon))

        self.p_global = struct_symSX(entries)

    def get_parameter_by_name(self, name: str) -> Parameter | None:
        """Get a parameter by its name."""
        return self.parameters.get(name)

    def get_differentiable_constant_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """Get all differentiable constant parameters."""
        return {
            key: value.value
            for key, value in self.parameters.items()
            if value.differentiable and not value.varying
        }

    def get_differentiable_varying_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """Get all differentiable varying parameters."""
        return {
            key: value.value
            for key, value in self.parameters.items()
            if value.differentiable and value.varying
        }

    def get_nondifferentiable_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """Get all nondifferentiable parameters."""
        return {
            key: value.value
            for key, value in self.parameters.items()
            if not value.differentiable
        }

    def get_default_param(self, field_: str) -> np.ndarray:
        if field_ not in ["p_global", "p"]:
            error_msg = f"Unknown field: {field_}. Available fields: p_global, p."
            raise ValueError(error_msg)

        val = None

        if field_ == "p_global":
            val = self.p_global(0)
            # Set the values for the symbolic structure
            for key, value in self.get_differentiable_constant_parameters().items():
                val[key] = value

            for key, value in self.get_differentiable_varying_parameters().items():
                for stage in range(self.N_horizon):
                    val[key, stage] = value

        if field_ == "p":
            val = self.p(0)
            # Set the values for the symbolic structure
            for key, value in self.get_nondifferentiable_parameters().items():
                val[key] = value

        return val.cat.full().flatten() if val is not None else np.array([])

    def map_dense_to_structured(
        self, field_: str, values_: np.ndarray, stage_: int | None = None
    ) -> struct:
        # TODO: Add error handling when field_ and stage_ and values_ are not compatible

        available_fields = ["p_global", "p"]

        if stage_ is None:
            stage_ = 0

        if field_ == "p_global":
            self.p_global_values = self.p_global(values_)
            return self.p_global_values

        if field_ == "p":
            self.parameter_values[stage_] = self.p(values_)
            return self.parameter_values[stage_]

        error_msg = f"Unknown field: {field_}. Available fields: {available_fields}."
        raise ValueError(error_msg)

    def get_dense(self, field_: str) -> np.ndarray:
        """Get the dense values of the structured parameter."""
        if field_ == "p_global":
            return self.get_p_global_values()

        if field_ == "p":
            return self.get_parameter_values()

        error_msg = f"Unknown field: {field_}. Available fields: p_global, p."
        raise ValueError(error_msg)

    def get_p_global_values(self) -> np.ndarray:
        return self.p_global_values.cat.full().flatten() if self.p_global_values else []

    def get_parameter_values(self, stage_: int | None = None) -> np.ndarray:
        """Get the symbolic variable p."""
        if stage_ is None:
            stage_ = 0
        return self.parameter_values[stage_].cat.full().flatten()

        # if stage_ is not None:

    def get_flat(self, field_: str) -> ca.SX | list:
        """Get the flat symbolic variable."""
        if field_ == "p_global":
            return self.p_global.cat if self.p_global is not None else []

        if field_ == "p":
            return self.p.cat if self.p is not None else []

        error_msg = f"Unknown field: {field_}. Available fields: p_global, p."
        raise ValueError(error_msg)

    def assign_to_ocp(self, ocp: AcadosOcp) -> None:
        """Assign the parameters to the OCP model."""
        if self.parameter_values is None:
            msg = "parameter_values is not set. Using nominal values."
            print(msg)
            self.initialize_parameter_values()

        if self.p_global_values is None:
            msg = "p_global_values is not set. Using nominal values."
            print(msg)
            self.initialize_p_global_values()

        if self.p_global is not None:
            # TODO: Refactor code to not rely on struct_symSX, then assign self.p_global.cat
            ocp.model.p_global = self.p_global
            ocp.p_global_values = self.get_p_global_values()

        if self.p is not None:
            # TODO: Refactor code to not rely on struct_symSX, then assign self.p.cat
            ocp.model.p = self.p
            ocp.parameter_values = self.get_parameter_values()


# TODO: Remove this function and use the AcadosParamManager instead.
def _process_params(
    params: list[str], nominal_param: dict[str, np.ndarray]
) -> tuple[list, list]:
    entries = []
    values = []
    for param in params:
        try:
            entries.append(entry(param, shape=nominal_param[param].shape))
            values.append(nominal_param[param].T.reshape(-1, 1))
        except AttributeError:
            entries.append(entry(param, shape=(1, 1)))
            values.append(np.array([nominal_param[param]]).reshape(-1, 1))
    return entries, values


# TODO: Remove this function and use the AcadosParamManager instead.
def find_param_in_p_or_p_global(
    param_name: list[str], model: AcadosModel
) -> dict[str, ca.SX]:
    if model.p == []:
        return {key: model.p_global[key] for key in param_name}
    if model.p_global is None:
        return {key: model.p[key] for key in param_name}
    return {
        key: (model.p[key] if key in model.p.keys() else model.p_global[key])  # noqa: SIM118
        for key in param_name
    }


# TODO: Remove this function and use the AcadosParamManager instead.
def translate_learnable_param_to_p_global(
    nominal_param: dict[str, np.ndarray],
    learnable_param: list[str],
    ocp: AcadosOcp,
    verbosity: int = 0,
) -> AcadosOcp:
    if learnable_param:
        entries, values = _process_params(learnable_param, nominal_param)
        ocp.model.p_global = struct_symSX(entries)
        ocp.p_global_values = np.concatenate(values).flatten()

    non_learnable_params = [key for key in nominal_param if key not in learnable_param]
    if non_learnable_params:
        entries, values = _process_params(non_learnable_params, nominal_param)
        ocp.model.p = struct_symSX(entries)
        ocp.parameter_values = np.concatenate(values).flatten()

    if verbosity:
        print("learnable_params", learnable_param)
        print("non_learnable_params", non_learnable_params)
    return ocp


# TODO: Remove this function and use the AcadosParamManager instead.
def is_stagewise_varying(param_key: str) -> bool:
    """
    Determine if a parameter is stage-wise varying based on its key pattern.

    Stage-wise varying parameters typically have keys in the format "label_stage_index",
    for example: "xref_0", "uref_5", "Q_2", etc.

    Args:
        param_key: The parameter key to check

    Returns:
        True if the parameter is stage-wise varying, False otherwise
    """
    # If there's no underscore, it's definitely not stage-wise
    if "_" not in param_key:
        return False

    # Split by underscore
    parts = param_key.split("_")

    # Check if the last part is a numeric stage index
    try:
        # Try to convert the last part to an integer
        int(parts[-1])

        # Make sure the base_name is not empty
        base_name = "_".join(parts[:-1])
        return bool(base_name)
    except ValueError:
        # The last part is not a numeric stage index
        return False


# TODO: Remove this function and use the AcadosParamManager instead.
def categorize_parameters(
    nominal_params: dict[str, np.ndarray], nonlearnable_keys: set[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Categorize parameters into learnable/nonlearnable and constant/varying."""
    learnable_keys = set(nominal_params.keys()) - nonlearnable_keys

    nonlearnable = {
        key: value for key, value in nominal_params.items() if key in nonlearnable_keys
    }

    learnable = {
        key: value for key, value in nominal_params.items() if key in learnable_keys
    }

    varying_learnable = {
        key: value for key, value in learnable.items() if is_stagewise_varying(key)
    }

    constant_learnable = {
        key: value for key, value in learnable.items() if key not in varying_learnable
    }

    return {
        "nonlearnable": nonlearnable,
        "learnable": {
            "constant": constant_learnable,
            "varying": varying_learnable,
        },
    }


# TODO: Remove this function and use the AcadosParamManager instead.
def create_p_global_entries(
    params: dict[str, np.ndarray], N_horizon: int
) -> tuple[dict[str, list]]:
    """Create casadi struct entries for parameters."""
    labels = {
        "constant": list(params["constant"].keys()),
        "varying": list({"_".join(key.split("_")[:-1]) for key in params["varying"]}),
    }

    entries = []

    # Add constant parameter entries
    for label in labels["constant"]:
        param_shape = params["constant"][label].shape
        entries.append(entry(label, shape=param_shape))

    # Add varying parameter entries with shape consistency check
    for label in labels["varying"]:
        # Get all shapes for this parameter across stages
        shapes = {
            params["varying"][f"{label}_{stage}"].shape
            for stage in range(N_horizon)
            if f"{label}_{stage}" in params["varying"]
        }

        if not shapes:
            continue  # Skip if no matching parameters

        if len(shapes) > 1:
            msg = f"Parameter '{label}' has inconsistent shapes across stages: {shapes}"
            raise ValueError(msg)

        entries.append(entry(label, shape=shapes.pop(), repeat=N_horizon))

    return entries


# TODO: Remove this function and use the AcadosParamManager instead.
def fill_p_global_values(
    params: dict[str, np.ndarray],
    p_global_values: struct_symSX,
) -> np.ndarray:
    """Fill parameter values in the CasADi structure."""
    # Fill constant values
    for key, value in params["constant"].items():
        p_global_values[key] = value

    # Fill varying values
    for key, value in params["varying"].items():
        label, stage = key.rsplit("_", 1)
        p_global_values[label, int(stage)] = value

    return p_global_values.cat.full().flatten()


# TODO: Remove this function and use the AcadosParamManager instead.
def create_p_entries(
    params: dict[str, np.ndarray],
    N_horizon: int,
) -> tuple[dict[str, list], dict[str, list]]:
    """Create casadi struct entries for parameters."""
    entries = []

    # Add non-learnable parameter entries
    for key, value in params.items():
        entries.append(entry(key, shape=value.shape))

    # Add indicator entry
    entries.append(entry("indicator", shape=(1,), repeat=N_horizon))

    return entries


# TODO: Remove this function and use the AcadosParamManager instead.
def fill_p_values(
    params: dict[str, np.ndarray],
    p_values: struct_symSX,
) -> np.ndarray:
    """Fill parameter values in the CasADi structure."""
    # Fill non-learnable values
    for key, value in params.items():
        p_values[key] = value

    return p_values.cat.full().flatten()
