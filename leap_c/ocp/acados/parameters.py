from typing import NamedTuple

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
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

    parameters: dict[str, Parameter] = {}
    p_global: struct_symSX | None = None
    p_global_values: struct | None = None
    p: struct_symSX | None = None
    parameter_values: list[struct] | None = None

    def __init__(self, N_horizon: int) -> None:
        self.N_horizon = N_horizon

    def add(
        self,
        parameter: Parameter,
    ) -> None:
        """
        Add a parameter to the manager.

        Args:
            parameter (Parameter): The parameter to add.
        """
        assert isinstance(parameter, Parameter), (
            "Parameter must be an instance of the Parameter NamedTuple."
        )
        self.parameters[parameter.name] = parameter

        # TODO: Consider not initializing p and p_global every time a parameter is added.
        self.initialize_p()
        self.initialize_p_global()

    def initialize_p(self) -> struct_symSX:
        """Update the structured parameter p."""
        entries = []
        for key, value in self.get_nondifferentiable_parameters().items():
            entries.append(entry(key, shape=value.shape))

        if self.get_differentiable_varying_parameters():
            entries.append(entry("indicator", shape=(self.N_horizon,)))

        self.p = struct_symSX(entries)

    def initialize_p_global(self) -> struct_symSX:
        """Update the structured parameter p_global."""
        entries = []
        for key, value in self.get_differentiable_constant_parameters().items():
            entries.append(entry(key, shape=value.shape))

        for key, value in self.get_differentiable_varying_parameters().items():
            entries.append(entry(key, shape=value.shape, repeat=self.N_horizon))

        self.p_global = struct_symSX(entries)

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

        if stage_ is None:
            stage_ = 0

        if field_ in self.p_global_values.keys():  # noqa: SIM118
            if stage_ is not None:
                return self.p_global_values[field_, stage_]
            return self.p_global_values[field_]

        if field_ in self.parameter_values[stage_].keys():  # noqa: SIM118
            return self.parameter_values[stage_][field_, stage_]

        available_fields = list(self.p_global_values.keys()) + list(
            self.parameter_values[stage_].keys()
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
        """
        Retrieve the default parameter array for a specified field.

        Args:
            field_ (str): The name of the parameter field to retrieve. Must be one
                "p_global" or "p".

        Returns:
            np.ndarray: The default parameter array corresponding to the requested
                field. Returns an empty array if the value is None.

        Raises:
            ValueError: If the provided field_ is not recognized.
        """
        available_fields = ["p_global", "p"]
        val = None

        if field_ == "p_global":
            val = self.p_global(0)
            for key, value in self.get_differentiable_constant_parameters().items():
                val[key] = value

            for key, value in self.get_differentiable_varying_parameters().items():
                for stage in range(self.N_horizon):
                    val[key, stage] = value

            return val.cat.full().flatten() if val is not None else np.array([])

        if field_ == "p":
            val = self.p(0)
            for key, value in self.get_nondifferentiable_parameters().items():
                val[key] = value

            return val.cat.full().flatten() if val is not None else np.array([])

        error_msg = f"Unknown field: {field_}. Available fields: {available_fields}."
        raise ValueError(error_msg)

    def map_dense_to_structured(
        self, field_: str, values_: np.ndarray, stage_: int | None = None
    ) -> struct:
        """
        Map a dense 1D numpy array of parameter values to the appropriate structured field
        within the object, either globally or for a specific stage.

        Args:
            field_ (str): The name of the parameter field to map values to. Must be one
                of "p_global" or "p".
            values_ (np.ndarray): A 1D numpy array containing the parameter values to be
                mapped.
            stage_ (int or None, optional): The stage index for stage-wise parameters.
                Must be a non-negative integer less than `self.N_horizon` if specified.
                Should be None for global parameters.

        Returns:
            struct: The structured parameter object after mapping the dense values.

        Raises:
            ValueError: If `field_` is unknown, if `stage_` is out of bounds, if `values_`
                is not 1D, or if an unhandled combination of arguments is provided.
            TypeError: If `stage_` is not an integer when specified, or if `values_` is
                not a numpy array.
        """
        available_fields = ["p_global", "p"]
        if field_ not in available_fields:
            error_msg = (
                f"Unknown field: {field_}. Available fields: {available_fields}."
            )
            raise ValueError(error_msg)

        if field_ == "p_global" and stage_ is not None:
            error_msg = "dense vector can not be mapped to be p_global stage-wise."
            raise ValueError(error_msg)

        if stage_ is not None and not isinstance(stage_, int):
            error_msg = f"stage_ must be an integer, got {type(stage_)}."
            raise TypeError(error_msg)

        if stage_ is not None and stage_ < 0:
            error_msg = f"stage_ must be non-negative, got {stage_}."
            raise ValueError(error_msg)

        if stage_ is not None and stage_ >= self.N_horizon:
            error_msg = (
                f"stage_ must be less than N_horizon ({self.N_horizon}), got {stage_}."
            )
            raise ValueError(error_msg)

        if values_ is None or not isinstance(values_, np.ndarray):
            error_msg = f"values_ must be a numpy array, got {type(values_)}."
            raise TypeError(error_msg)

        if values_.ndim != 1:
            error_msg = f"values_ must be a 1D numpy array, got {values_.ndim}D."
            raise ValueError(error_msg)

        if stage_ is None:
            stage_ = 0

        if field_ == "p_global":
            self.p_global_values = self.p_global(values_)
            return self.p_global_values

        if field_ == "p":
            self.parameter_values[stage_] = self.p(values_)
            return self.parameter_values[stage_]

        error_msg = (
            f"Unhandled combination of field: {field_} and stage_: {stage_}"
            f" and values_: {values_}."
        )
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
        """Get the p_global values."""
        return (
            self.p_global_values.cat.full().flatten()
            if self.p_global_values
            else np.array([])
        )

    def get_parameter_values(self, stage_: int | None = None) -> np.ndarray:
        """Get the parameter values for a specific stage."""
        stage_ = stage_ or 0
        return (
            self.parameter_values[stage_].cat.full().flatten()
            if self.parameter_values
            else np.array([])
        )

    def get_flat(self, field_: str) -> ca.SX | list:
        """Get the flat symbolic variable to use in an AcadosModel."""
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
            # TODO: Refactor code to not rely on struct_symSX, then assign get_flat("p_global")
            ocp.model.p_global = self.p_global
            ocp.p_global_values = self.get_p_global_values()

        if self.p is not None:
            # TODO: Refactor code to not rely on struct_symSX, then assign get_flat("p")
            ocp.model.p = self.p
            ocp.parameter_values = self.get_parameter_values()

    def set_params(self, acados_ocp_solver: AcadosOcpSolver) -> None:
        """Set_p_global and parameter_values in the AcadosOcpSolver."""
        raise NotImplementedError


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
