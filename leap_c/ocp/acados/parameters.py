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
    p: struct_symSX | None = None
    p_global_val: np.ndarray | None = None
    p_val: np.ndarray | None = None

    def __init__(self, N_horizon: int) -> None:
        self.N_horizon = N_horizon

    def reset(self) -> None:
        """Reset the parameter manager."""
        self.parameters = {}
        self.p_global = None
        self.p = None
        self.p_global_val = None
        self.p_val = None

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

    def update_p(self) -> struct_symSX:
        """Update the structured parameter p."""
        entries = []
        for key, value in self.get_nondifferentiable_parameters().items():
            entries.append(entry(key, shape=value.shape))

        # If self.get_differentiable_varying_parameters() is not empty, we need
        # indicator variables
        if self.get_differentiable_varying_parameters():
            entries.append(
                entry(
                    "indicator",
                    shape=(1,),
                    repeat=self.N_horizon,
                )
            )

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

    def get_sym(self, field_: str, stage_: int | None = None) -> ca.SX:
        """Get the symbolic variable for a given field."""
        if field_ in self.p_global.keys():  # noqa: SIM118
            if stage_ is not None:
                return self.p_global[field_, stage_]
            return self.p_global[field_]

        if field_ in self.p.keys():  # noqa: SIM118
            if stage_ is not None:
                return self.p[field_, stage_]
            return self.p[field_]

        available_fields = list(self.p_global.keys()) + list(self.p.keys())
        error_message = f"Unknown field: {field_}. Available fields: {available_fields}"
        raise ValueError(error_message)

    def set_val(self, field_: str, value: ca.SX, stage_: int | None = None) -> None:
        """Set the value for a given field."""
        pass

    def get_nominal_values(self, field_: str) -> struct:
        if field_ not in ["p_global", "p"]:
            error_msg = f"Unknown field: {field_}. Available fields: p_global, p."
            raise ValueError(error_msg)

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

            # NB: The indicator variables are all zero by default. They need to handled
            # together with the OCP solver.

        return val

    def map_dense_to_structured(self, field_: str, values_: np.ndarray) -> struct:
        if field_ == "p_global":
            self.p_global_val = self.p_global(values_)
            return self.p_global_val
        if field_ == "p":
            self.p_val = self.p(values_)
            return self.p_val

        error_msg = f"Unknown field: {field_}. Available fields: p_global, p."
        raise ValueError(error_msg)

    def get_dense(self, field_: str) -> np.ndarray:
        """Get the dense values of the structured parameter."""
        if field_ == "p_global":
            if self.p_global_val is None:
                msg = "p_global_val is not set. Using nominal values."
                print(msg)
                self.p_global_val = self.get_nominal_values("p_global")
            return self.p_global_val.cat.full().flatten()

        if field_ == "p":
            if self.p_val is None:
                msg = "p_val is not set. Using nominal values."
                print(msg)
                self.p_val = self.get_nominal_values("p")
            return self.p_val.cat.full().flatten()

        error_msg = f"Unknown field: {field_}. Available fields: p_global, p."
        raise ValueError(error_msg)

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
        if self.p_global is not None:
            ocp.model.p_global = self.p_global
            ocp.p_global_values = self.get_dense("p_global")

        if self.p is not None:
            ocp.model.p = self.p
            ocp.parameter_values = self.get_dense("p")


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


def fill_p_values(
    params: dict[str, np.ndarray],
    p_values: struct_symSX,
) -> np.ndarray:
    """Fill parameter values in the CasADi structure."""
    # Fill non-learnable values
    for key, value in params.items():
        p_values[key] = value

    return p_values.cat.full().flatten()
