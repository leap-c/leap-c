from typing import NamedTuple

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
    fix: bool = True
    differentiable: bool = False
    stagewise: bool = False


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
        N_horizon: int,
    ) -> None:
        self.parameters = {param.name: param for param in params}

        # Check that no parameter has more than one dimension.
        for key, value in self.parameters.items():
            if value.value.ndim > 1:
                raise ValueError(
                    f"Parameter '{key}' has more than one dimension, "
                    "which is not supported. Use a single-dimensional array."
                )

        # Check tha no parameter has None as lower_bound or upper_bound.
        # for key, value in self.parameters.items():
        #     if value.fix:
        #         continue
        #     if value.lower_bound is None:
        #         raise ValueError(
        #             f"Parameter '{key}' has no lower bound. This is not supported."
        #         )
        #     if value.upper_bound is None:
        #         raise ValueError(
        #             f"Parameter '{key}' has no upper bound. This is not supported."
        #         )

        self.N_horizon = N_horizon

        self._build_p()
        self._build_p_global()
        self._build_p_global_bounds()

    def _build_p(self) -> None:
        # Create symbolic structures for parameters
        entries = []
        for key, value in self._get_nondifferentiable_parameters().items():
            entries.append(entry(key, shape=value.shape))

        for key, value in self._get_nondifferentiable_stagewise_parameters().items():
            entries.append(entry(key, shape=value.shape, repeat=self.N_horizon + 1))

        # if (
        #     self._get_differentiable_stagewise_parameters()
        #     or self._get_nondifferentiable_stagewise_parameters()
        # ):
        #     entries.append(entry("indicator", shape=(self.N_horizon + 1,)))
        #    # Add indicator for stagewise parameters
        # TODO (Jasper): Fix this, currently in the overwrite function we always try
        #  to set the indicator, even if there are no stagewise parameters.
        entries.append(entry("indicator", shape=(self.N_horizon + 1,)))

        self.p = struct_symSX(entries)

        # Initialize parameter values for each stage
        parameter_values = self.p(0)

        # for stage in range(self.N_horizon):
        for key, value in self._get_nondifferentiable_parameters().items():
            parameter_values[key] = value

        for key, value in self._get_nondifferentiable_stagewise_parameters().items():
            for stage in range(self.N_horizon + 1):
                parameter_values[key, stage] = value

        self.parameter_values = parameter_values

    def _build_p_global(self) -> None:
        # Create symbolic structure for global parameters
        entries = []
        for key, value in self._get_differentiable_global_parameters().items():
            entries.append(entry(key, shape=value.shape))

        for key, value in self._get_differentiable_stagewise_parameters().items():
            entries.append(entry(key, shape=value.shape, repeat=self.N_horizon + 1))

        self.p_global = struct_symSX(entries)

        # Initialize global parameter values
        self.p_global_values = self.p_global(0)

        for key, value in self._get_differentiable_global_parameters().items():
            self.p_global_values[key] = value

        for key, value in self._get_differentiable_stagewise_parameters().items():
            for stage in range(self.N_horizon + 1):
                self.p_global_values[key, stage] = value

    def _build_p_global_bounds(self) -> None:
        # Build bounds for p_global parameters
        lb = self.p_global(0)
        ub = self.p_global(0)
        for key in self.p_global.keys():
            if self.parameters[key].stagewise:
                for stage in range(self.N_horizon + 1):
                    if self.parameters[key].lower_bound is None:
                        # TODO: Needs test if this is correct.
                        # lb[key, stage] = -ca.inf
                        raise ValueError(
                            f"Lower bound for stagewise parameter '{key}' is None. "
                            "This is not supported."
                        )
                    else:
                        lb[key, stage] = self.parameters[key].lower_bound

                    if self.parameters[key].upper_bound is None:
                        # TODO: Needs test if this is correct.
                        # ub[key, stage] = ca.inf
                        raise ValueError(
                            f"Upper bound for stagewise parameter '{key}' is None. "
                            "This is not supported."
                        )
                    else:
                        ub[key, stage] = self.parameters[key].upper_bound
            else:
                if self.parameters[key].lower_bound is None:
                    raise ValueError(
                        f"Lower bound for global parameter '{key}' is None. "
                        "This is not supported."
                    )
                else:
                    lb[key] = self.parameters[key].lower_bound
                if self.parameters[key].upper_bound is None:
                    raise ValueError(
                        f"Upper bound for global parameter '{key}' is None. "
                        "This is not supported."
                    )
                else:
                    ub[key] = self.parameters[key].upper_bound

        self.lb = lb.cat.full().flatten()
        self.ub = ub.cat.full().flatten()

    def _get_differentiable_global_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """Get all differentiable global parameters."""
        return {
            key: value.value
            for key, value in self.parameters.items()
            if value.differentiable and not value.stagewise
        }

    def _get_differentiable_stagewise_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """Get all differentiable stage-wise parameters."""
        return {
            key: value.value
            for key, value in self.parameters.items()
            if value.differentiable and value.stagewise
        }

    def _get_nondifferentiable_stagewise_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """Get all differentiable stage-wise parameters."""
        return {
            key: value.value
            for key, value in self.parameters.items()
            if not value.differentiable and value.stagewise
        }

    def _get_nondifferentiable_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """Get all nondifferentiable parameters."""
        return {
            key: value.value
            for key, value in self.parameters.items()
            if not value.differentiable and not value.stagewise
        }

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
            self.parameter_values.cat.full().reshape(1, -1),
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
            batch_parameter_values[:, :, self.p.f[key]] = val.reshape(
                batch_size, self.N_horizon + 1, -1
            )

        expected_shape = (batch_size, self.N_horizon + 1, self.p.cat.shape[0])
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
        field: str,
        stage: int | None = None,
    ) -> ca.SX | ca.MX | np.ndarray:
        """Get the variable for a given field at a specific stage."""
        if field in self.parameters and self.parameters[field].fix:
            return self.parameters[field].value

        if field in self._get_differentiable_stagewise_parameters():
            return sum(
                [
                    self.p["indicator"][stage_] * self.p_global[field, stage_]
                    for stage_ in range(self.N_horizon + 1)
                ]
            )

        if field in self._get_nondifferentiable_stagewise_parameters():
            return sum(
                [
                    self.p["indicator"][stage_] * self.p[field, stage_]
                    for stage_ in range(self.N_horizon + 1)
                ]
            )

        if field in self.p_global.keys():
            if stage is not None:
                return self.p_global[field, stage]
            return self.p_global[field]

        if field in self.p.keys():
            if stage is not None and field == "indicator":
                return self.p[field][stage]
            return self.p[field]

        available_fields = list(self.p_global.keys()) + list(self.p.keys())
        error_message = f"Unknown field: {field}. Available fields: {available_fields}"
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
