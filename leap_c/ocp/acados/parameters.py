from dataclasses import dataclass, field
from typing import Any, Collection, Literal, Self
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
            if self.end_stages:
                warn(
                    f"Parameter '{self.name}' with interface '{self.interface}' defines end_stages."
                    " The end_stages will be ignored as only 'learnable' parameters supports it.",
                    UserWarning,
                    stacklevel=2,
                )


@dataclass
class _ParameterStore:
    """Helper class for storing parameter information and symbolic variables.

    This is used internally by AcadosParameterManager to manage the parameters.
    Allows adding symbols, defaults, and bounds. Handles the indexing and sizing
    automatically.
    """

    symbols: dict[str, ca.SX | ca.MX] = field(default_factory=dict)
    indices: dict[str, tuple[int, int]] = field(default_factory=dict)
    defaults: dict[str, np.ndarray] = field(default_factory=dict)
    lb: dict[str, np.ndarray] = field(default_factory=dict)
    ub: dict[str, np.ndarray] = field(default_factory=dict)
    _size: int = field(default=0, init=False)

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        name: str,
        symbol: ca.SX | ca.MX,
        default: np.ndarray,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
    ) -> None:
        n = default.size
        self.symbols[name] = symbol
        self.indices[name] = (self._size, self._size + n)
        self.defaults[name] = default
        if lb is not None:
            self.lb[name] = lb
        if ub is not None:
            self.ub[name] = ub
        self._size += n

    def get_values(self) -> np.ndarray:
        return (
            np.concatenate([v.reshape(-1, order="F") for v in self.defaults.values()])
            if self.defaults
            else np.array([])
        )

    def get_symbols(self) -> ca.SX | ca.MX:
        return ca.vertcat(*list(self.symbols.values()))


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
        N_horizon: The horizon length for the ocp.
        casadi_type: The CasADi symbolic type used for the parameters, either "SX" or "MX".
        learnable_default_flat: Learnable parameters' default values as a flattened NDArray.
        non_learnable_default_flat: Non-learnable parameters' default values as a flattened NDArray.
        p_global: CasADi SX/MX expression for the learnable parameters.
        p: CasADi SX/MX expression for the non-learnable parameters.
        p_full: CasADi SX/MX expression for both learnable and non-learnable parameters.
    """

    parameters: dict[str, AcadosParameter]
    N_horizon: int
    casadi_type: Literal["SX", "MX"]
    _learnable_parameter_store: _ParameterStore
    _non_learnable_parameter_store: _ParameterStore
    _need_indicator: bool

    @property
    def learnable_default_flat(self) -> np.ndarray:
        return self._learnable_parameter_store.get_values()

    @property
    def non_learnable_default_flat(self) -> np.ndarray:
        return self._non_learnable_parameter_store.get_values()

    @property
    def p_global(self) -> ca.SX | ca.MX:
        return self._learnable_parameter_store.get_symbols()

    @property
    def p(self) -> ca.SX | ca.MX:
        return self._non_learnable_parameter_store.get_symbols()

    @property
    def p_full(self) -> ca.SX | ca.MX:
        return ca.vertcat(self.p_global, self.p)

    @staticmethod
    def _create_symbol(name: str, size: int, casadi_type: Literal["SX", "MX"]) -> ca.SX | ca.MX:
        if casadi_type == "SX":
            return ca.SX.sym(name, size, 1)
        elif casadi_type == "MX":
            return ca.MX.sym(name, size, 1)
        else:
            raise ValueError(f"Unsupported casadi_type: {casadi_type}")

    def _store_learnable_parameter(self, parameter: AcadosParameter) -> None:
        if parameter.end_stages:
            self._need_indicator = True
            if "indicator" not in self._non_learnable_parameter_store.symbols:
                indicator = AcadosParameter(
                    name="indicator",
                    default=np.zeros(self.N_horizon + 1),
                    interface="non-learnable",
                )
                self._store_non_learnable_parameter(indicator)
            starts, ends = _define_starts_and_ends(
                end_stages=parameter.end_stages, N_horizon=self.N_horizon
            )
            for start, end in zip(starts, ends):
                # Build symbolic expressions for each stage
                # following the template {name}_{first_stage}_{last_stage}
                # e.g. price_0_10, price_11_20, etc.
                p_name = f"{parameter.name}_{start}_{end}"
                symbol = self._create_symbol(p_name, parameter.default.size, self.casadi_type)
                self._learnable_parameter_store.add(
                    p_name, symbol, parameter.default, parameter.space.low, parameter.space.high
                )
        else:
            symbol = self._create_symbol(parameter.name, parameter.default.size, self.casadi_type)
            self._learnable_parameter_store.add(
                parameter.name, symbol, parameter.default, parameter.space.low, parameter.space.high
            )

    def _store_non_learnable_parameter(self, parameter: AcadosParameter) -> None:
        symbol = self._create_symbol(parameter.name, parameter.default.size, self.casadi_type)
        self._non_learnable_parameter_store.add(parameter.name, symbol, parameter.default)

    def __init__(
        self,
        parameters: Collection[AcadosParameter],
        N_horizon: int,
        casadi_type: Literal["SX", "MX"] = "SX",
    ) -> None:
        """Initialize the parameter manager by building the symbols and the parameter stores.

        Validates the provided input, then constructs two parameter stores:

        - ``learnable_parameter_store``: one entry per learnable param; stage-varying params are
          split into named blocks, e.g. ``price_0_4`` and ``price_5_9``.
        - ``non_learnable_parameter_store``: one entry per non-learnable param, plus an
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
            ValueError: If any parameter has ``end_stages`` whose last element is not
            ``N_horizon - 1`` or ``N_horizon``.
        """
        # add parameters to the manager
        # TODO: since parameters are being added incrementally, we should remove this warning, and
        # in the future make the parameters optional and default to an empty list on construction.
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

        self.N_horizon = N_horizon
        self.casadi_type = casadi_type
        self._learnable_parameter_store = _ParameterStore()
        self._non_learnable_parameter_store = _ParameterStore()

        self._need_indicator = False
        for _, parameter in self.parameters.items():
            if parameter.interface == "learnable":
                self._store_learnable_parameter(parameter)
            if parameter.interface == "non-learnable":
                self._store_non_learnable_parameter(parameter)

    def add_parameter(self, parameter: AcadosParameter) -> Self:
        """Adds a new parameter to the manager.

        This is a helper method for incrementally building the parameter manager, e.g. when
        parameters are defined in different parts of the code.

        Args:
            parameter: The AcadosParameter to add.

        Returns:
            The same parameter manager, returned to allow method chaining.
        """
        if parameter.name in self.parameters:
            raise ValueError(
                f"Parameter '{parameter.name}' already exists in the manager. "
                "Use a different name or modify the existing parameter instead."
            )
        if parameter.end_stages and parameter.end_stages[-1] not in [
            self.N_horizon - 1,
            self.N_horizon,
        ]:
            raise ValueError(
                f"Parameter '{parameter.name}' has end_stages {parameter.end_stages} "
                f"but the last element must be either {self.N_horizon - 1} or {self.N_horizon}."
            )
        self.parameters[parameter.name] = parameter
        if parameter.interface == "learnable":
            self._store_learnable_parameter(parameter)
        if parameter.interface == "non-learnable":
            self._store_non_learnable_parameter(parameter)
        return self

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
        default_flat = self._learnable_parameter_store.get_values()
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
                        s, e = self._learnable_parameter_store.indices[param_key]
                        param_idx = slice(s, e)
                    except KeyError as e:
                        raise KeyError(f"Learnable parameter '{param_key}' not found.") from e

                    # All stages in this block use the value from the start stage
                    batch_param[:, param_idx] = values_reshaped[:, start, :]
            else:
                # Non-stage-varying parameter - single value per batch
                param_key = param_name
                try:
                    s, e = self._learnable_parameter_store.indices[param_key]
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
        expected_shape = (batch_size, Np1, self._non_learnable_parameter_store.size)

        # Create a batch of parameter values - if indicators are not needed and no overwrites are
        # passed, just return a broadcasted view to avoid unnecessary memory allocation; otherwise,
        # create a tiled array (is writeable, so indicators and overwrites can be applied afterward)
        nonlearn_param_default_flat = self._non_learnable_parameter_store.get_values()
        if not (self._need_indicator or overwrite):
            return np.broadcast_to(nonlearn_param_default_flat, expected_shape)
        batch_parameter_values = np.tile(nonlearn_param_default_flat, (batch_size, Np1, 1))

        # Set indicator for each stage
        if self._need_indicator:
            s, e = self._non_learnable_parameter_store.indices["indicator"]
            batch_parameter_values[:, :, s:e] = np.eye(Np1)

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
            s, e = self._non_learnable_parameter_store.indices[key]
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

        Args:
            dtype: The desired data type for the spaces.
        """
        store = self._learnable_parameter_store
        learnable_spaces = [
            gym.spaces.Box(
                low=store.lb[name].reshape(store.defaults[name].shape),
                high=store.ub[name].reshape(store.defaults[name].shape),
                dtype=dtype,
            )
            for name in store.symbols
        ]

        if not learnable_spaces:
            return gym.spaces.Box(low=np.empty(0, dtype), high=np.empty(0, dtype), dtype=dtype)
        elif len(learnable_spaces) == 1:
            return learnable_spaces[0]
        else:
            return gym.spaces.utils.flatten_space(gym.spaces.Tuple(learnable_spaces))

    def has_parameter(self, name: str, interface: str | None = None) -> bool:
        """Return True if a parameter with the given name (and optionally interface) exists.

        Args:
            name: Parameter name to look up.
            interface: If provided, also check that the parameter's interface matches
                (e.g. ``"non-learnable"``, ``"learnable"``, ``"fix"``).
        """
        if name not in self.parameters:
            return False
        if interface is not None:
            return self.parameters[name].interface == interface
        return True

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
                indicators.append(
                    ca.sum(
                        self._non_learnable_parameter_store.symbols["indicator"][start : end + 1]
                    )
                )
                variables.append(self._learnable_parameter_store.symbols[f"{name}_{start}_{end}"])

            terms = []
            for indicator, variable in zip(indicators, variables):
                terms.append(indicator * variable)
            return sum(terms)

        if self.parameters[name].interface == "learnable":
            return self._learnable_parameter_store.symbols[name]

        if self.parameters[name].interface == "non-learnable":
            return self._non_learnable_parameter_store.symbols[name]
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
        if len(self._learnable_parameter_store.symbols) > 0:
            ocp.model.p_global = self._learnable_parameter_store.get_symbols()
            ocp.p_global_values = self._learnable_parameter_store.get_values()

        if len(self._non_learnable_parameter_store.symbols) > 0:
            ocp.model.p = self._non_learnable_parameter_store.get_symbols()
            ocp.parameter_values = self._non_learnable_parameter_store.get_values()

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

        return any(
            fnmatch.fnmatch(name, pattern) for name in self._learnable_parameter_store.symbols
        )

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

        keys = [key for key in self._learnable_parameter_store.symbols if label in key]

        if keys == []:
            raise ValueError(f"No learnable parameters found with label '{label}'.")

        idx = [self._learnable_parameter_store.indices[key] for key in keys]
        idx = [slice(s, e) for s, e in idx]
        return param_values[..., np.r_[*idx]].reshape(-1, len(keys))


def _define_starts_and_ends(end_stages: list[int], N_horizon: int) -> tuple[list[int], list[int]]:
    """Define the start and end indices for stage-varying parameters."""
    ends = end_stages
    starts = [0] + [v + 1 for v in ends if v + 1 <= N_horizon]
    return starts, ends
