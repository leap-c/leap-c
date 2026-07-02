from dataclasses import dataclass, field
from typing import Literal
from warnings import warn

import casadi as ca
import numpy as np

from leap_c.utils.parameters import ParamSplits, _define_starts_and_ends


@dataclass
class _AcadosParameter:
    """Parameter container for flexible parameter management with acados.

    It provides an interface for defining parameter sets without requiring knowledge of
    internal CasADi tools or acados interface details.

    Attributes:
        name: The name identifier for the parameter.
        default: The parameter's default numerical value(s).
        interface: Parameter interface type.
            Either `"non-differentiable"` (not exposed to the learning interface, but will be
            changeable parameters also after creation of the solver), or `"differentiable"`
            (parameters directly exposed to the learning interface, in particular supporting
            sensitivities). Defaults to `"differentiable"`.
        splits: Defines how the parameter varies across stages. Only used for the `"differentiable"`
            interface. Accepts:
            - `list[int]`: Sorted (ascending) stage boundaries. The parameter takes one value per
            resulting segment. Example: with `9` stages (`0` to `9`) and `splits = [4, 9]`, the
            parameter has one value for stages `0` to `4` and another for stages `5` to `9`.
            - `int`: Number of equal-sized splits.
            - `"stagewise"`: One value per stage. Equivalent to `list(range(N_horizon + 1))`.
            - `"global"`: A single value across all stages. Equivalent to `[N_horizon]`.
            Defaults to `"global"`.
    """

    name: str
    default: np.ndarray
    interface: Literal["differentiable", "non-differentiable"] = "differentiable"
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

        if self.interface != "differentiable" and self.splits != "global":
            warn(
                f"Parameter '{self.name}' with interface '{self.interface}' defines splits."
                " The splits will be ignored as only 'differentiable' parameters supports it.",
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
    _size: int = field(default=0, init=False)

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        name: str,
        symbol: ca.SX | ca.MX,
        default: np.ndarray,
    ) -> None:
        n = default.size
        self.symbols[name] = symbol
        self.indices[name] = (self._size, self._size + n)
        self.defaults[name] = default
        self._size += n

    def get_values(self) -> np.ndarray:
        return (
            np.concatenate([v.reshape(-1, order="F") for v in self.defaults.values()])
            if self.defaults
            else np.array([])
        )

    def get_symbols(self) -> ca.SX | ca.MX:
        return ca.vertcat(*list(self.symbols.values()))
