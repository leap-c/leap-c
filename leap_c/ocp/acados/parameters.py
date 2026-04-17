from dataclasses import dataclass, field
from typing import Any, Collection, Literal
from warnings import warn

import casadi as ca
import gymnasium as gym
import numpy as np
import torch
from acados_template import AcadosOcp


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
            Only used for the `"learnable"` interface. If empty, the parameter remains constant
            across all stages. Defaults to an empty list.
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

    def __post_init__(self):
        if self.default.ndim > 2:
            raise ValueError(
                f"Parameter '{self.name}' has {self.default.ndim} dimensions, "
                f"but CasADi only supports arrays up to 2 dimensions. "
                f"Parameter shape: {self.default.shape}"
            )

        if self.interface == "learnable":
            if isinstance(self.space, gym.spaces.Box):
                if len(self.space.shape) > 2:
                    raise ValueError(
                        f"Parameter '{self.name}' space has {len(self.space.shape)} dimensions, "
                        f"but CasADi only supports arrays up to 2 dimensions. "
                        f"Space shape: {self.space.shape}"
                    )
            elif self.space is None:
                self.space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=np.shape(self.default),
                )
            else:
                raise NotImplementedError(
                    f"Parameter '{self.name}' has space of type {type(self.space)}, "
                    "but currently only gym.spaces.Box is supported."
                )

            if self.end_stages and sorted(self.end_stages) != self.end_stages:
                raise ValueError(
                    f"Parameter '{self.name}' has end_stages {self.end_stages} which are not "
                    "in sorted ascending order."
                )
        else:
            if self.space is not None:
                warn(
                    f"Parameter '{self.name}' with interface '{self.interface}' defines space."
                    " The space will be ignored as only 'learnable' parameters supports it.",
                    UserWarning,
                    stacklevel=2,
                )
                self.space = None
            if self.end_stages:
                warn(
                    f"Parameter '{self.name}' with interface '{self.interface}' defines end_stages."
                    " The end_stages will be ignored as only 'learnable' parameters supports it.",
                    UserWarning,
                    stacklevel=2,
                )
                self.end_stages = []


class AcadosParameterManager:
    """Manager class for handling acados parameters according to their specifications.

    In particular, this handles stage-varying learnable parameters, which is not available
    out-of-the-box in acados.

    Basic usage inside the AcadosOcp formulation:

    .. code-block:: python

        # Step 1: define parameters
        params = [
            AcadosParameter("price", default=np.array([1.0]), interface="learnable",
                            end_stages=[4, 9]),
            AcadosParameter("outdoor_temp", default=np.array([20.0]), interface="non-learnable"),
        ]
        manager = AcadosParameterManager(params, N_horizon=10)

        # Step 2: use symbolic variables in the OCP formulation
        price = manager.get("price")           # CasADi expression, stage-aware
        outdoor_temp = manager.get("outdoor_temp")  # CasADi symbolic variable

        # Step 3: wire into acados
        manager.assign_to_ocp(ocp)

    **Stage-varying learnable parameters** (``end_stages`` non-empty) are implemented via a
    one-hot *indicator* vector that is appended to the non-learnable parameters.  At stage ``k``
    only ``indicator[k]`` is 1; ``get()`` returns a weighted sum over all stage blocks so the
    same symbolic expression evaluates to the correct block value at every stage.

    .. warning::
        If you forget to set the indicator correctly in
        ``combine_non_learnable_parameter_values()``, every stage will silently evaluate to zero
        for all stage-varying learnable parameters.

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
    _learnable_symbols: dict[str, ca.SX | ca.MX]
    _learnable_indices: dict[str, tuple[int, int]]
    _learnable_size: int
    _learnable_parameters_default: dict[str, np.ndarray]
    _learnable_parameters_lb: dict[str, np.ndarray]
    _learnable_parameters_ub: dict[str, np.ndarray]
    _non_learnable_symbols: dict[str, ca.SX | ca.MX]
    _non_learnable_indices: dict[str, tuple[int, int]]
    _non_learnable_size: int
    _non_learnable_parameters_default: dict[str, np.ndarray]
    N_horizon: int
    need_indicator: bool

    @property
    def learnable_default_flat(self) -> np.ndarray:
        return (
            np.concatenate(
                [arr.reshape(-1, order="F") for arr in self._learnable_parameters_default.values()]
            )
            if self._learnable_parameters_default
            else np.array([])
        )

    @property
    def non_learnable_default_flat(self) -> np.ndarray:
        return (
            np.concatenate(
                [
                    arr.reshape(-1, order="F")
                    for arr in self._non_learnable_parameters_default.values()
                ]
            )
            if self._non_learnable_parameters_default
            else np.array([])
        )

    @property
    def p_global(self) -> ca.SX | ca.MX:
        return ca.vertcat(*list(self._learnable_symbols.values()))

    @property
    def p(self) -> ca.SX | ca.MX:
        return ca.vertcat(*list(self._non_learnable_symbols.values()))

    @property
    def p_full(self) -> ca.SX | ca.MX:
        return ca.vertcat(self.p_global, self.p)

    def __init__(
        self,
        parameters: Collection[AcadosParameter],
        N_horizon: int,
        casadi_type: Literal["SX", "MX"] = "SX",
    ) -> None:
        """Initialize the parameter manager and build CasADi symbolic structs.

        Validates all parameters, then constructs two CasADi symbolic structs:

        - ``learnable_parameters``: one entry per learnable param; stage-varying params are
          split into named blocks, e.g. ``price_0_4`` and ``price_5_9``.
        - ``non_learnable_parameters``: one entry per non-learnable param, plus an
          ``indicator`` vector of length ``N_horizon + 1`` if any stage-varying learnable
          params exist.

        Args:
            parameters: Collection of :class:`AcadosParameter` instances describing all
                parameters for the OCP.
            N_horizon: Horizon length ``N``.  Stages are indexed ``0`` to ``N`` (inclusive),
                giving ``N + 1`` stages in total.
            casadi_type: CasADi symbolic type to use, either ``"SX"`` (default, faster for
                small problems) or ``"MX"`` (required when parameters appear inside CasADi
                ``Function`` objects that are evaluated multiple times).

        Raises:
            ValueError: If any parameter has more than 2 dimensions, uses a non-Box space,
                or has ``end_stages`` whose last element is not ``N_horizon - 1`` or
                ``N_horizon``.
        """
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
            # Check end_stages convention
            if param.end_stages and param.end_stages[-1] not in [N_horizon - 1, N_horizon]:
                raise ValueError(
                    f"Parameter '{param.name}' has end_stages {param.end_stages} "
                    f"but the last element must be either {N_horizon - 1} or {N_horizon}."
                )
        self.parameters = {param.name: param for param in parameters}

        self._learnable_symbols = {}
        self._learnable_indices = {}
        self._learnable_size = 0
        self._learnable_parameters_default = {}
        self._learnable_parameters_lb = {}
        self._learnable_parameters_ub = {}
        self._non_learnable_symbols = {}
        self._non_learnable_indices = {}
        self._non_learnable_size = 0
        self._non_learnable_parameters_default = {}

        self.N_horizon = N_horizon

        def _add_learnable_parameter_entries(name: str, parameter: AcadosParameter) -> None:
            if parameter.end_stages:
                self.need_indicator = True
                starts, ends = _define_starts_and_ends(
                    end_stages=parameter.end_stages, N_horizon=self.N_horizon
                )
                for start, end in zip(starts, ends):
                    # Build symbolic expressions for each stage
                    # following the template {name}_{first_stage}_{last_stage}
                    # e.g. price_0_10, price_11_20, etc.
                    pname = f"{name}_{start}_{end}"
                    if casadi_type == "SX":
                        self._learnable_symbols[pname] = ca.SX.sym(pname, parameter.default.size, 1)
                    elif casadi_type == "MX":
                        self._learnable_symbols[pname] = ca.MX.sym(pname, parameter.default.size, 1)
                    self._learnable_indices[pname] = (
                        self._learnable_size,
                        self._learnable_size + parameter.default.size,
                    )
                    self._learnable_size += parameter.default.size
                    self._learnable_parameters_default[pname] = parameter.default
                    self._learnable_parameters_lb[pname] = parameter.space.low
                    self._learnable_parameters_ub[pname] = parameter.space.high
            else:
                if casadi_type == "SX":
                    self._learnable_symbols[name] = ca.SX.sym(name, parameter.default.size, 1)
                elif casadi_type == "MX":
                    self._learnable_symbols[name] = ca.MX.sym(name, parameter.default.size, 1)
                self._learnable_indices[name] = (
                    self._learnable_size,
                    self._learnable_size + parameter.default.size,
                )
                self._learnable_size += parameter.default.size
                self._learnable_parameters_default[name] = parameter.default
                self._learnable_parameters_lb[name] = parameter.space.low
                self._learnable_parameters_ub[name] = parameter.space.high

        def _add_non_learnable_parameter_entries(name: str, parameter: AcadosParameter) -> None:
            # Non-learnable parameters are by construction for each stage
            if casadi_type == "SX":
                self._non_learnable_symbols[name] = ca.SX.sym(name, parameter.default.size, 1)
            elif casadi_type == "MX":
                self._non_learnable_symbols[name] = ca.MX.sym(name, parameter.default.size, 1)
            self._non_learnable_indices[name] = (
                self._non_learnable_size,
                self._non_learnable_size + parameter.default.size,
            )
            self._non_learnable_size += parameter.default.size
            self._non_learnable_parameters_default[name] = parameter.default

        self.need_indicator = False
        for name, parameter in self.parameters.items():
            if parameter.interface == "learnable":
                _add_learnable_parameter_entries(name, parameter)
            if parameter.interface == "non-learnable":
                _add_non_learnable_parameter_entries(name, parameter)

        if self.need_indicator:
            if casadi_type == "SX":
                self._non_learnable_symbols["indicator"] = ca.SX.sym(
                    "indicator", self.N_horizon + 1, 1
                )
            elif casadi_type == "MX":
                self._non_learnable_symbols["indicator"] = ca.MX.sym(
                    "indicator", self.N_horizon + 1, 1
                )
            self._non_learnable_indices["indicator"] = (
                self._non_learnable_size,
                self._non_learnable_size + self.N_horizon + 1,
            )
            self._non_learnable_size += self.N_horizon + 1
            self._non_learnable_parameters_default["indicator"] = np.zeros(self.N_horizon + 1)

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
        inferred_batch_size = next(iter(overwrites.values())).shape[0] if overwrites else None

        # Validate batch_size consistency
        if batch_size is not None and inferred_batch_size is not None:
            if batch_size != inferred_batch_size:
                raise ValueError(
                    f"Provided batch_size={batch_size} does not match "
                    f"inferred batch_size={inferred_batch_size} from overwrites."
                )

        batch_size = inferred_batch_size if inferred_batch_size is not None else batch_size or 1

        # Get default parameter array and replicate it along the batch dimension - if no overwrites
        # are passed, just return a broadcasted view to avoid unnecessary memory allocation;
        # otherwise, create a tiled array (is writeable, so overwrites can be applied afterwards)
        default_flat = self.learnable_default_flat
        if not overwrites:
            return np.broadcast_to(default_flat, (batch_size, default_flat.size))
        batch_param = np.tile(default_flat, (batch_size, 1))

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
                        s, e = self._learnable_indices[param_key]
                        param_idx = slice(s, e)
                    except KeyError as e:
                        raise KeyError(f"Learnable parameter '{param_key}' not found.") from e

                    # All stages in this block use the value from the start stage
                    batch_param[:, param_idx] = values_reshaped[:, start, :]
            else:
                # Non-stage-varying parameter - single value per batch
                param_key = param_name
                try:
                    s, e = self._learnable_indices[param_key]
                    param_idx = slice(s, e)
                except KeyError as e:
                    raise KeyError(f"Learnable parameter '{param_key}' not found.") from e

                # Reshape to handle both scalar and vector parameters
                values_reshaped = values.reshape(batch_size, -1)

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
        Np1 = self.N_horizon + 1
        expected_shape = (batch_size, Np1, self._non_learnable_size)

        # Create a batch of parameter values - if indicators are not needed and no overwrites are
        # passed, just return a broadcasted view to avoid unnecessary memory allocation; otherwise,
        # create a tiled array (is writeable, so indicators and overwrites can be applied afterward)
        nonlearn_param_default_flat = self.non_learnable_default_flat
        if not (self.need_indicator or overwrite):
            return np.broadcast_to(nonlearn_param_default_flat, expected_shape)
        batch_parameter_values = np.tile(nonlearn_param_default_flat, (batch_size, Np1, 1))

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
            s, e = self._non_learnable_indices[key]
            batch_parameter_values[:, :, s:e] = val.reshape(batch_size, Np1, -1)

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
        """Get the symbolic variable (or fixed value) for a parameter.

        For stage-varying learnable parameters (those with ``end_stages``), the returned
        expression is a weighted sum over all stage blocks, gated by the ``indicator`` vector
        in the non-learnable parameters.  The expression evaluates to the correct block value
        at each stage, but **only if the indicator is set correctly** via
        ``combine_non_learnable_parameter_values()``.  If the indicator is all-zero (e.g. the
        default), every stage silently evaluates to zero for these parameters.

        Args:
            name: The name of the parameter to retrieve.

        Returns:
            - ``np.ndarray`` for ``"fix"`` parameters (the default value).
            - CasADi ``SX``/``MX`` expression for ``"learnable"`` and ``"non-learnable"``
              parameters.

        Raises:
            ValueError: If ``name`` is not registered with this manager.
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
                indicators.append(ca.sum(self._non_learnable_symbols["indicator"][start : end + 1]))
                variables.append(self._learnable_symbols[f"{name}_{start}_{end}"])

            terms = []
            for indicator, variable in zip(indicators, variables):
                terms.append(indicator * variable)
            return sum(terms)

        if self.parameters[name].interface == "learnable":
            return self._learnable_symbols[name]

        if self.parameters[name].interface == "non-learnable":
            return self._non_learnable_symbols[name]
        else:
            raise ValueError(
                f"Unknown interface type for field '{name}': {self.parameters[name].interface}"
            )

    def assign_to_ocp(self, ocp: AcadosOcp) -> None:
        """Assign the parameters to the acados OCP object.

        Wires ``learnable_parameters`` into ``ocp.model.p_global`` (shared across all stages)
        and ``non_learnable_parameters`` into ``ocp.model.p`` (per-stage).  Default values are
        set on ``ocp.p_global_values`` and ``ocp.parameter_values`` respectively.

        Args:
            ocp: The :class:`acados_template.AcadosOcp` instance to configure.
                Any existing ``p_global`` / ``p`` definitions are overwritten.
        """
        if len(self._learnable_symbols) > 0:
            ocp.model.p_global = self.p_global
            ocp.p_global_values = self.learnable_default_flat

        if len(self._non_learnable_symbols) > 0:
            ocp.model.p = self.p
            ocp.parameter_values = self.non_learnable_default_flat

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

        learnable_param_names = self._learnable_symbols.keys()
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

        keys = [key for key in self._learnable_symbols.keys() if label in key]

        if keys == []:
            raise ValueError(f"No learnable parameters found with label '{label}'.")

        idx = [self._learnable_indices[key] for key in keys]
        idx = [slice(s, e) for s, e in idx]
        return param_values[..., np.r_[*idx]].reshape(-1, len(keys))


def _define_starts_and_ends(end_stages: list[int], N_horizon: int) -> tuple[list[int], list[int]]:
    """Define the start and end indices for stage-varying parameters."""
    ends = end_stages
    starts = [0] + [v + 1 for v in ends if v + 1 <= N_horizon]
    return starts, ends
