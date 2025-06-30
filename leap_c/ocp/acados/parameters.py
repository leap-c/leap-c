from typing import NamedTuple

import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from casadi.tools import entry, struct, struct_symSX


class Parameter(NamedTuple):
    name: str
    value: np.ndarray
    # OJ: what are those bounds used for?
    lower_bound: np.ndarray | None
    upper_bound: np.ndarray | None
    differentiable: bool
    varying: bool  # TODO: rename to global? or to stage_wise and use negated form, to not have the same word (global).


class AcadosParamManager:
    """Manager for acados parameters."""

    parameters: dict[str, Parameter] = {}
    p_global: struct_symSX | None = None
    p_global_values: struct | None = None
    p: struct_symSX | None = None
    parameter_values: list[struct] | None = None

    def __init__(
        self,
        params: list[Parameter],
        ocp: AcadosOcp,
    ) -> None:
        self.parameters = {param.name: param for param in params}

        self.N_horizon = ocp.solver_options.N_horizon

        self._build_p()
        self._build_p_global()
        self._build_p_global_bounds()

        self.assign_to_ocp(ocp)

    def _build_p(self) -> None:
        # Create symbolic structures for parameters
        entries = []
        for key, value in self._get_nondifferentiable_parameters().items():
            entries.append(entry(key, shape=value.shape))

        # TODO: N+1
        if self._get_differentiable_varying_parameters():
            entries.append(entry("indicator", shape=(self.N_horizon,)))

        self.p = struct_symSX(entries)

        # Initialize parameter values for each stage
        parameter_values = self.p(0)

        # for stage in range(self.N_horizon):
        for key, value in self._get_nondifferentiable_parameters().items():
            parameter_values[key] = value

        self.parameter_values = parameter_values

    def _build_p_global(self) -> None:
        # Create symbolic structure for global parameters
        entries = []
        for key, value in self._get_differentiable_constant_parameters().items():
            entries.append(entry(key, shape=value.shape))

        for key, value in self._get_differentiable_varying_parameters().items():
            entries.append(entry(key, shape=value.shape, repeat=self.N_horizon))

        self.p_global = struct_symSX(entries)

        # Initialize global parameter values
        self.p_global_values = self.p_global(0)

        for key, value in self._get_differentiable_constant_parameters().items():
            self.p_global_values[key] = value

        for key, value in self._get_differentiable_varying_parameters().items():
            for stage in range(self.N_horizon):
                self.p_global_values[key, stage] = value

    def _build_p_global_bounds(self) -> None:
        # Build bounds for p_global parameters
        lb = self.p_global(0)
        ub = self.p_global(0)
        for key in self.p_global.keys():  # noqa: SIM118
            if self.parameters[key].varying:
                for stage in range(self.N_horizon):
                    lb[key, stage] = self.parameters[key].lower_bound
                    ub[key, stage] = self.parameters[key].upper_bound
            else:
                lb[key] = self.parameters[key].lower_bound
                ub[key] = self.parameters[key].upper_bound

        self.lb = lb.cat.full().flatten()
        self.ub = ub.cat.full().flatten()

    # TODO: global instead of constant? as they are varied via learning.
    def _get_differentiable_constant_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """Get all differentiable constant parameters."""
        return {
            key: value.value
            for key, value in self.parameters.items()
            if value.differentiable and not value.varying
        }

    def _get_differentiable_varying_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """Get all differentiable varying parameters."""
        return {
            key: value.value
            for key, value in self.parameters.items()
            if value.differentiable and value.varying
        }

    def _get_nondifferentiable_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """Get all nondifferentiable parameters."""
        return {
            key: value.value
            for key, value in self.parameters.items()
            if not value.differentiable
        }

    def combine_parameter_values(
        self,
        batch_size: int | None = None,
        **overwrite: np.ndarray,
    ) -> np.ndarray:
        """
        Combine all parameters into a single numpy array.

        Args:
            batch_size (int, optional): The batch size for the parameters.
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
            self.parameter_values.cat.full().reshape(1, -1),
            (batch_size, self.N_horizon, 1),
        )

        # Set indicator for each stage
        batch_parameter_values[:, :, -self.N_horizon :] = np.tile(
            np.eye(self.N_horizon),
            (batch_size, 1, 1),
        )

        # Overwrite the values in the batch
        # TODO: Make sure indexing is consistent.
        # Issue is the difference between casadi (row major) and numpy (column major)
        # when using matrix values.
        for key, val in overwrite.items():
            batch_parameter_values[:, :, self.p.f[key]] = val.reshape(
                batch_size, self.N_horizon, -1
            )

        expected_shape = (batch_size, self.N_horizon, self.p.cat.shape[0])
        assert batch_parameter_values.shape == expected_shape, (
            f"batch_parameter_values should have shape {expected_shape}, "
            f"got {batch_parameter_values.shape}."
        )

        return batch_parameter_values

    def get_p_global_bounds(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Get the lower bound for p_global parameters."""
        if self.p_global is None:
            return None, None

        return self.lb, self.ub

    def get(
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

        # if field_ in self.differentiable_varying:
        #   return sum([self.indicator[i] * self.p_global[field_][stage_] for i in range(N+1)])

        available_fields = list(self.p_global.keys()) + list(self.p.keys())
        error_message = f"Unknown field: {field_}. Available fields: {available_fields}"
        raise ValueError(error_message)

    def assign_to_ocp(self, ocp: AcadosOcp) -> None:
        """Assign the parameters to the OCP model."""
        if self.p_global is not None:
            ocp.model.p_global = self.p_global.cat
            ocp.p_global_values = (
                self.p_global_values.cat.full().flatten()
                if self.p_global_values
                else np.array([])
            )

        if self.p is not None:
            ocp.model.p = self.p.cat
            ocp.parameter_values = (
                self.parameter_values.cat.full().flatten()
                if self.parameter_values
                else np.array([])
            )
