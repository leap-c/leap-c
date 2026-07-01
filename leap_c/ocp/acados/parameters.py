from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.utils.dependencies import require_jax, require_torch
from leap_c.utils.parameters import ParamSplits

if TYPE_CHECKING:
    import torch


@dataclass
class AcadosParameter:
    """Parameter container for flexible parameter management with acados.

    It provides an interface for defining parameter sets without requiring knowledge of
    internal CasADi tools or acados interface details.

    Attributes:
        name: The name identifier for the parameter.
        default: The parameter's default numerical value(s).
        space: A gym.spaces.Space defining the valid parameter space.
            Only used for learnable parameters. Defaults to `None` (unbounded).
        interface: Parameter interface type.
            Either `"non-learnable"` (not exposed to the learning interface, but will be changeable
            parameters also after creation of the solver), or `"learnable"` (parameters directly
            exposed to the learning interface, in particular supporting sensitivities). Defaults to
            `"learnable"`.
        splits: Defines how the parameter varies across stages. Only used for the `"learnable"`
            interface. Accepts:
            - `list[int]`: Sorted (ascending) stage boundaries. The parameter takes one value per
            resulting segment. Example: with `9` stages (`0` to `9`) and `splits = [4, 9]`, the
            parameter has one value for stages `0` to `4` and another for stages `5` to `9`.
            - `int`: Number of equal-sized splits.
            - `"stagewise"`: One value per stage. Equivalent to `list(range(N_horizon + 1))`.
            - `"global"`: A single value across all stages. Equivalent to `[N_horizon]`.
            Defaults to `"global"`.
    """

    # Fields from base Parameter class
    name: str
    default: np.ndarray
    space: gym.spaces.Space | None = None
    interface: Literal["learnable", "non-learnable"] = "learnable"

    # Additional acados-specific field
    splits: ParamSplits = "global"

    @property
    def is_stage_varying(self) -> bool:
        return self.splits != "global"

    def __post_init__(self):
        if isinstance(self.splits, list) and not self.splits:
            raise ValueError(
                f"Parameter '{self.name}' has empty splits list. Hint: if you meant to define a"
                " global parameter, please set splits='global' instead."
            )

        if isinstance(self.splits, list) and self.splits != sorted(self.splits):
            raise ValueError(
                f"Parameter '{self.name}' splits {self.splits} are not sorted in ascending order."
            )

        if isinstance(self.splits, int) and self.splits <= 1:
            hint = (
                " If you meant to define a global parameter, please set splits='global'"
                if self.splits == 1
                else ""
            )
            raise ValueError(
                f"Parameter '{self.name}' has invalid splits value {self.splits}, number of splits"
                f" must be `>1`.{hint}"
            )

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
        else:
            if self.space is not None:
                warn(
                    f"Parameter '{self.name}' with interface '{self.interface}' defines space."
                    " The space will be ignored as only 'learnable' parameters supports it.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.splits != "global":
                warn(
                    f"Parameter '{self.name}' with interface '{self.interface}' defines splits."
                    " The splits will be ignored as only 'learnable' parameters supports it.",
                    UserWarning,
                    stacklevel=2,
                )

    def overwrite_shape(self, N_horizon: int) -> tuple:
        if self.splits == "global":
            return self.default.shape
        _, ends = _define_starts_and_ends(self.splits, N_horizon)
        return (len(ends), *self.default.shape)

    def broadcasted_default(self, N_horizon: int) -> np.ndarray:
        if self.splits == "global":
            return self.default
        _, ends = _define_starts_and_ends(self.splits, N_horizon)
        n_segments = len(ends)
        return np.tile(self.default, (n_segments, *([1] * self.default.ndim)))


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
    """acados parameter management.

    Handles parameter registration, validation, CasADi symbol creation, and provides default
    numpy implementations for combining parameters. Framework-specific methods
    (e.g. :meth:`combine_learnable_parameters_torch`) preserve differentiability through the
    composition.

    **Stage-varying learnable parameters** (``splits`` are not ``"global"``) are implemented via a
    one-hot *indicator* vector that is appended to the non-learnable parameters.  At stage ``k``
    only ``indicator[k]`` is 1; :meth:`get` returns a weighted sum over all stage blocks so the
    same symbolic expression evaluates to the correct block value at every stage.

    .. warning::
        If you forget to set the indicator correctly in
        :meth:`combine_non_learnable_parameters`, every stage will silently evaluate to zero
        for all stage-varying learnable parameters.

    Attributes:
        parameters: Dictionary of parameter names to AcadosParameter instances.
        N_horizon: The horizon length for the ocp.
        casadi_type: The CasADi symbolic type used for the parameters, either "SX" or "MX".
        learnable_default_flat: Learnable parameters' default values as a flattened NDArray.
        non_learnable_default_flat: Non-learnable parameters' default values as a flattened NDArray.
        learnable_symbols: CasADi SX/MX expression for the learnable parameters.
        non_learnable_symbols: CasADi SX/MX expression for the non-learnable parameters.
    """

    parameters: dict[str, AcadosParameter]
    N_horizon: int
    casadi_type: Literal["SX", "MX"]
    _learnable_parameter_store: _ParameterStore
    _non_learnable_parameter_store: _ParameterStore
    _need_indicator: bool
    _finalized: bool

    @property
    def learnable_default_flat(self) -> np.ndarray:
        return self._learnable_parameter_store.get_values()

    @property
    def non_learnable_default_flat(self) -> np.ndarray:
        return self._non_learnable_parameter_store.get_values()

    @property
    def learnable_parameter_names(self) -> list[str]:
        return [name for name, param in self.parameters.items() if param.interface == "learnable"]

    @property
    def non_learnable_parameter_names(self) -> list[str]:
        return [
            name for name, param in self.parameters.items() if param.interface == "non-learnable"
        ]

    @property
    def learnable_symbols(self) -> ca.SX | ca.MX:
        return self._learnable_parameter_store.get_symbols()

    @property
    def non_learnable_symbols(self) -> ca.SX | ca.MX:
        return self._non_learnable_parameter_store.get_symbols()

    @staticmethod
    def _create_symbol(name: str, size: int, casadi_type: Literal["SX", "MX"]) -> ca.SX | ca.MX:
        if casadi_type == "SX":
            return ca.SX.sym(name, size, 1)
        elif casadi_type == "MX":
            return ca.MX.sym(name, size, 1)
        else:
            raise ValueError(f"Unsupported casadi_type: {casadi_type}")

    def _store_learnable_parameter(self, parameter: AcadosParameter) -> None:
        if parameter.is_stage_varying:
            self._need_indicator = True
            if "indicator" not in self._non_learnable_parameter_store.symbols:
                indicator = AcadosParameter(
                    name="indicator",
                    default=np.zeros(self.N_horizon + 1),
                    interface="non-learnable",
                )
                self._store_non_learnable_parameter(indicator)
            starts, ends = _define_starts_and_ends(
                splits=parameter.splits, N_horizon=self.N_horizon
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
        N_horizon: int,
        casadi_type: Literal["SX", "MX"] = "SX",
    ) -> None:
        """Initialize the parameter manager.

        Args:
            N_horizon: Horizon length ``N``.  Stages are indexed ``0`` to ``N`` (inclusive),
                giving ``N + 1`` stages in total.
            casadi_type: CasADi symbolic type to use, either ``"SX"`` (default, faster for
                small problems) or ``"MX"`` (required when parameters appear inside CasADi
                ``Function`` objects that are evaluated multiple times).
        """
        self.parameters = {}
        self.N_horizon = N_horizon
        self.casadi_type = casadi_type
        self._learnable_parameter_store = _ParameterStore()
        self._non_learnable_parameter_store = _ParameterStore()
        self._need_indicator = False
        self._finalized = False

    def register_parameter(
        self,
        name: str,
        default: np.ndarray,
        space: gym.spaces.Space | None = None,
        differentiable: bool = False,
        splits: ParamSplits = "global",
    ) -> ca.SX | ca.MX:
        """Register a parameter and return a CasADi symbolic for immediate use.

        The returned symbolic is a real CasADi SX (or MX) expression (not a placeholder).
        It can be used directly in cost, dynamics, and constraint expressions.

        Args:
            name: The name of the parameter.
            default: The default value(s) for the parameter.
            space: A gymnasium space defining valid parameter values (for learnable params).
            differentiable: If True, the parameter supports sensitivities (learnable).
                If False, the parameter is changeable at runtime but not differentiable
                (non-learnable). Defaults to ``False``.
            splits: Defines how the parameter varies across stages. See
                :class:`AcadosParameter` for details. Defaults to ``"global"``.

        Returns:
            A CasADi symbolic expression for the parameter.
        """
        if self._finalized:
            raise ValueError("Cannot register parameters after assigning to OCP")

        parameter = AcadosParameter(
            name=name,
            default=default,
            space=space,
            interface="learnable" if differentiable else "non-learnable",
            splits=splits,
        )
        if isinstance(parameter.splits, list):
            if parameter.splits[-1] not in [
                self.N_horizon - 1,
                self.N_horizon,
            ]:
                raise ValueError(
                    f"Parameter '{parameter.name}' has splits {parameter.splits} "
                    f"but the last element must be either {self.N_horizon - 1} or {self.N_horizon}."
                )
        if isinstance(parameter.splits, int):
            if parameter.splits > self.N_horizon + 1:
                raise ValueError(
                    f"Parameter '{parameter.name}' has {parameter.splits} splits, which exceeds the"
                    f" number of stages {self.N_horizon + 1}."
                )
        self.parameters[parameter.name] = parameter
        if parameter.interface == "learnable":
            self._store_learnable_parameter(parameter)
        if parameter.interface == "non-learnable":
            self._store_non_learnable_parameter(parameter)
        return self.get(parameter.name)

    def combine_learnable_parameters_torch(
        self,
        batch_size: int | None = None,
        device: "torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
        **overwrites: "torch.Tensor | np.ndarray",
    ) -> "torch.Tensor":
        """Combine learnable parameters into a flat tensor, preserving differentiability.

        Uses ``torch.cat``, indexing, and reshaping — all differentiable — so gradients
        flow back to the original parameter tensors in the ``overwrites`` dict.

        Args:
            batch_size: Batch size. Required.
            device: Target device for the output tensor. Required.
            dtype: Target dtype for the output tensor. Required.
            **overwrites: Named parameter overrides as tensors or numpy arrays.

        Returns:
            Tensor of shape ``(batch_size, N_learnable)``.

        Raises:
            ImportError: If torch is not installed.
        """
        torch = require_torch()

        inferred_batch_size = next(iter(overwrites.values())).shape[0] if overwrites else None

        if batch_size is not None and inferred_batch_size is not None:
            if batch_size != inferred_batch_size:
                raise ValueError(
                    f"Provided batch_size={batch_size} does not match "
                    f"inferred batch_size={inferred_batch_size} from overwrites."
                )

        batch_size = inferred_batch_size if inferred_batch_size is not None else batch_size or 1

        batch_param = (
            torch.from_numpy(self.learnable_default_flat)
            .to(device, dtype)
            .expand(batch_size, -1)
            .clone()
        )

        if not overwrites:
            return batch_param

        for name, values in overwrites.items():
            if name not in self.parameters:
                raise ValueError(
                    f"Parameter '{name}' not found. "
                    f"Available parameters: {list(self.parameters.keys())}"
                )

            param = self.parameters[name]

            if param.interface != "learnable":
                raise ValueError(
                    f"Parameter '{name}' has interface '{param.interface}', "
                    "but only 'learnable' parameters can be used in this method."
                )

            if values.shape[0] != batch_size:
                raise ValueError(
                    f"Parameter '{name}' values have batch size {values.shape[0]}, "
                    f"but expected {batch_size}."
                )

            if param.is_stage_varying:
                expected_n_segments = param.overwrite_shape(self.N_horizon)[0]
                if values.shape[1] != expected_n_segments:
                    raise ValueError(
                        f"Parameter '{name}' is stage-varying and requires shape "
                        f"(batch_size, {expected_n_segments}, ...), but got shape {values.shape}."
                    )

        batch_param.requires_grad_()

        for name in self.learnable_parameter_names:
            param = self.parameters[name]
            if name in overwrites:
                val = overwrites[name]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).to(device=device, dtype=dtype)
                elif isinstance(val, torch.Tensor):
                    val = val.to(device=device, dtype=dtype)
            else:
                val = None

            if param.is_stage_varying:
                starts, ends = _define_starts_and_ends(
                    splits=param.splits, N_horizon=self.N_horizon
                )
                for seg_idx, (start, end) in enumerate(zip(starts, ends)):
                    key = f"{name}_{start}_{end}"
                    s, e = self._learnable_parameter_store.indices[key]
                    if val is not None:
                        batch_param = torch.cat(
                            [
                                batch_param[:, :s],
                                val[:, seg_idx].reshape(batch_size, -1),
                                batch_param[:, e:],
                            ],
                            dim=-1,
                        )
            else:
                s, e = self._learnable_parameter_store.indices[name]
                if val is not None:
                    batch_param = torch.cat([batch_param[:, :s], val, batch_param[:, e:]], dim=-1)

        return batch_param

    def combine_learnable_parameters_jax(
        self,
        batch_size: int | None = None,
        **overwrites: Any,
    ) -> Any:
        """Combine learnable parameters into a flat JAX array, preserving differentiability.

        .. note::
            This method is a placeholder. Contributions implementing JAX-native operations
            (e.g. using ``jax.numpy``) are welcome.

        Args:
            batch_size: Batch size.
            **overwrites: Named parameter overrides as JAX or numpy arrays.

        Returns:
            JAX array of shape ``(batch_size, N_learnable)``.

        Raises:
            ImportError: If jax is not installed.
            NotImplementedError: Until a JAX implementation is contributed.
        """
        require_jax()
        raise NotImplementedError(
            "combine_learnable_parameters_jax is not yet implemented. "
            "Contributions are welcome! See combine_learnable_parameters_torch for reference."
        )

    def combine_non_learnable_parameters(
        self, batch_size: int | None = None, **overwrite: np.ndarray
    ) -> np.ndarray:
        """Combine all non-learnable parameters into a single numpy array.

        Args:
            batch_size: The batch size for the parameters.
                Not needed if overwrite is provided.
            **overwrite: Overwrite values for specific parameters.
                The keys should correspond to the parameter names to overwrite.
                The values need to be np.ndarray with shape ``(batch_size, N_horizon, pdim)``,
                where ``pdim`` is the number of dimensions of the parameter to overwrite.

        Returns:
            np.ndarray: shape ``(batch_size, N_horizon, np)`` with ``np`` being the number of
            ``parameter_values``.
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

    def get(self, name: str) -> ca.SX | ca.MX | np.ndarray:
        """Get the symbolic variable for a parameter.

        For stage-varying learnable parameters (those with ``splits``), the returned
        expression is a weighted sum over all stage blocks, gated by the ``indicator`` vector
        in the non-learnable parameters.  The expression evaluates to the correct block value
        at each stage, but **only if the indicator is set correctly** via
        :meth:`combine_non_learnable_parameters`.  If the indicator is all-zero (e.g. the
        default), every stage silently evaluates to zero for these parameters.

        Args:
            name: The name of the parameter to retrieve.

        Returns:
            - CasADi ``SX``/``MX`` expression for ``"learnable"`` and ``"non-learnable"``
              parameters.

        Raises:
            ValueError: If ``name`` is not registered with this manager.
        """
        if name not in self.parameters:
            raise ValueError(f"Unknown name: {name}. Available names: {', '.join(self.parameters)}")

        if (
            self.parameters[name].interface == "learnable"
            and self.parameters[name].is_stage_varying
        ):
            starts, ends = _define_starts_and_ends(
                splits=self.parameters[name].splits, N_horizon=self.N_horizon
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

    # TODO (Mazen): I guess this needs to be deprecated. After separating, all the code should use
    # the dictionary API.
    def get_labeled_learnable_parameters(
        self,
        param_values: Any,
        label: str,
    ) -> Any:
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

    def assign_to_ocp(self, ocp: AcadosOcp) -> None:
        """Synchronize the parameter manager's symbols and defaults onto the OCP.

        Sets ``model.p``, ``model.p_global``, ``parameter_values``, and ``p_global_values``
        on the provided acados OCP instance. Should be called before solver code generation.

        Args:
            ocp: An acados ``AcadosOcp`` instance.
        """
        self._finalized = True
        ocp.model.p = self.non_learnable_symbols
        ocp.model.p_global = self.learnable_symbols
        ocp.parameter_values = self.non_learnable_default_flat
        ocp.p_global_values = self.learnable_default_flat


def _define_starts_and_ends(
    splits: list[int] | int | Literal["stagewise"], N_horizon: int
) -> tuple[list[int], list[int]]:
    """Define the start and end indices for stage-varying parameters."""
    # TODO (dirk): Document the usage of this function and the expected output format.
    if splits == "stagewise":
        ends = list(range(N_horizon + 1))
    elif isinstance(splits, int):
        split_size = (N_horizon + 1) // splits
        remainder = (N_horizon + 1) % splits
        sizes = [split_size] * splits
        for i in range(remainder):
            sizes[i] += 1
        ends = (np.cumsum(sizes) - 1).tolist()
    elif isinstance(splits, list):
        ends = splits
    else:
        raise ValueError(f"Invalid splits value: {splits}")
    starts = [0] + [v + 1 for v in ends if v + 1 <= N_horizon]
    return starts, ends
