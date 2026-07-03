from typing import TYPE_CHECKING, Any, Literal

import casadi as ca
import numpy as np
from acados_template import AcadosOcp

from leap_c.parameters.data import _AcadosParameter, _ParameterStore
from leap_c.parameters.utils import ParamSplits, _define_starts_and_ends
from leap_c.utils.dependencies import require_jax, require_torch

if TYPE_CHECKING:
    import torch


class AcadosParameterManager:
    """acados parameter management.

    Handles parameter registration, validation, CasADi symbol creation, and provides default
    numpy implementations for combining parameters. Framework-specific methods
    (e.g. :meth:`combine_differentiable_parameters_torch`) preserve differentiability through the
    composition.

    **Stage-varying differentiable parameters** (``splits`` are not ``"global"``) are implemented
    via a one-hot *indicator* vector that is appended to the non-differentiable parameters.  At
    stage ``k`` only ``indicator[k]`` is 1; :meth:`get` returns a weighted sum over all stage
    blocks so the same symbolic expression evaluates to the correct block value at every stage.

    .. warning::
        If you forget to set the indicator correctly in
        :meth:`combine_non_differentiable_parameters`, every stage will silently evaluate to zero
        for all stage-varying differentiable parameters.

    Attributes:
        parameters: Dictionary of parameter names to _AcadosParameter instances.
        N_horizon: The horizon length for the ocp.
        casadi_type: The CasADi symbolic type used for the parameters, either "SX" or "MX".
        differentiable_default_flat:
            Differentiable parameters' default values as a flattened NDArray.
        non_differentiable_default_flat:
            Non-differentiable parameters' default values as a flattened NDArray.
        differentiable_symbols: CasADi SX/MX expression for the differentiable parameters.
        non_differentiable_symbols: CasADi SX/MX expression for the non-differentiable parameters.
    """

    parameters: dict[str, _AcadosParameter]
    N_horizon: int
    casadi_type: Literal["SX", "MX"]
    _differentiable_parameter_store: _ParameterStore
    _non_differentiable_parameter_store: _ParameterStore
    _need_indicator: bool
    _finalized: bool

    @property
    def differentiable_default_flat(self) -> np.ndarray:
        return self._differentiable_parameter_store.get_values()

    @property
    def non_differentiable_default_flat(self) -> np.ndarray:
        return self._non_differentiable_parameter_store.get_values()

    @property
    def differentiable_parameter_names(self) -> list[str]:
        return [
            name for name, param in self.parameters.items() if param.interface == "differentiable"
        ]

    @property
    def non_differentiable_parameter_names(self) -> list[str]:
        return [
            name
            for name, param in self.parameters.items()
            if param.interface == "non-differentiable"
        ]

    @property
    def differentiable_symbols(self) -> ca.SX | ca.MX:
        return self._differentiable_parameter_store.get_symbols()

    @property
    def non_differentiable_symbols(self) -> ca.SX | ca.MX:
        return self._non_differentiable_parameter_store.get_symbols()

    def __repr__(self) -> str:
        from leap_c.utils.repr import format_parameter_manager_repr

        return format_parameter_manager_repr(self)

    @staticmethod
    def _create_symbol(name: str, size: int, casadi_type: Literal["SX", "MX"]) -> ca.SX | ca.MX:
        if casadi_type == "SX":
            return ca.SX.sym(name, size, 1)
        elif casadi_type == "MX":
            return ca.MX.sym(name, size, 1)
        else:
            raise ValueError(f"Unsupported casadi_type: {casadi_type}")

    def _store_differentiable_parameter(self, parameter: _AcadosParameter) -> None:
        if parameter.is_stage_varying:
            self._need_indicator = True
            if "indicator" not in self._non_differentiable_parameter_store.symbols:
                indicator = _AcadosParameter(
                    name="indicator",
                    default=np.zeros(self.N_horizon + 1),
                    interface="non-differentiable",
                )
                self._store_non_differentiable_parameter(indicator)
            starts, ends = _define_starts_and_ends(
                splits=parameter.splits, N_horizon=self.N_horizon
            )
            for start, end in zip(starts, ends):
                p_name = f"{parameter.name}_{start}_{end}"
                symbol = self._create_symbol(p_name, parameter.default.size, self.casadi_type)
                self._differentiable_parameter_store.add(p_name, symbol, parameter.default)
        else:
            symbol = self._create_symbol(parameter.name, parameter.default.size, self.casadi_type)
            self._differentiable_parameter_store.add(parameter.name, symbol, parameter.default)

    def _store_non_differentiable_parameter(self, parameter: _AcadosParameter) -> None:
        symbol = self._create_symbol(parameter.name, parameter.default.size, self.casadi_type)
        self._non_differentiable_parameter_store.add(parameter.name, symbol, parameter.default)

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
        self._differentiable_parameter_store = _ParameterStore()
        self._non_differentiable_parameter_store = _ParameterStore()
        self._need_indicator = False
        self._finalized = False

    def register_parameter(
        self,
        name: str,
        default: np.ndarray,
        differentiable: bool = False,
        splits: ParamSplits = "global",
    ) -> ca.SX | ca.MX:
        """Register a parameter and return a CasADi symbolic for immediate use.

        The returned symbolic is a real CasADi SX (or MX) expression (not a placeholder).
        It can be used directly in cost, dynamics, and constraint expressions.

        Args:
            name: The name of the parameter.
            default: The default value(s) for the parameter.
            differentiable: If True, the parameter supports sensitivities (differentiable).
                If False, the parameter is changeable at runtime but not differentiable
                (non-differentiable). Defaults to ``False``.
            splits: Defines how the parameter varies across stages. See
                :class:`_AcadosParameter` for details. Defaults to ``"global"``.

        Returns:
            A CasADi symbolic expression for the parameter.
        """
        if self._finalized:
            raise ValueError("Cannot register parameters after assigning to OCP")

        parameter = _AcadosParameter(
            name=name,
            default=default,
            interface="differentiable" if differentiable else "non-differentiable",
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
        if parameter.interface == "differentiable":
            self._store_differentiable_parameter(parameter)
        if parameter.interface == "non-differentiable":
            self._store_non_differentiable_parameter(parameter)
        return self.get(parameter.name)

    def combine_differentiable_parameters_torch(
        self,
        batch_size: int | None = None,
        device: "torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
        **overwrites: "torch.Tensor | np.ndarray",
    ) -> "torch.Tensor":
        """Combine differentiable parameters into a flat tensor, preserving differentiability.

        Uses ``torch.cat``, indexing, and reshaping — all differentiable — so gradients
        flow back to the original parameter tensors in the ``overwrites`` dict.

        Args:
            batch_size: Batch size. Required.
            device: Target device for the output tensor. Required.
            dtype: Target dtype for the output tensor. Required.
            **overwrites: Named parameter overrides as tensors or numpy arrays.

        Returns:
            Tensor of shape ``(batch_size, N_differentiable)``.

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
            torch.from_numpy(self.differentiable_default_flat)
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

            if param.interface != "differentiable":
                raise ValueError(
                    f"Parameter '{name}' has interface '{param.interface}', "
                    "but only 'differentiable' parameters can be used in this method."
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

        for name in self.differentiable_parameter_names:
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
                    s, e = self._differentiable_parameter_store.indices[key]
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
                s, e = self._differentiable_parameter_store.indices[name]
                if val is not None:
                    batch_param = torch.cat([batch_param[:, :s], val, batch_param[:, e:]], dim=-1)

        return batch_param

    def combine_differentiable_parameters_jax(
        self,
        batch_size: int | None = None,
        **overwrites: Any,
    ) -> Any:
        """Combine differentiable parameters into a flat JAX array, preserving differentiability.

        .. note::
            This method is a placeholder. Contributions implementing JAX-native operations
            (e.g. using ``jax.numpy``) are welcome.

        Args:
            batch_size: Batch size.
            **overwrites: Named parameter overrides as JAX or numpy arrays.

        Returns:
            JAX array of shape ``(batch_size, N_differentiable)``.

        Raises:
            ImportError: If jax is not installed.
            NotImplementedError: Until a JAX implementation is contributed.
        """
        require_jax()
        raise NotImplementedError(
            "combine_differentiable_parameters_jax is not yet implemented. "
            "Contributions are welcome! See combine_differentiable_parameters_torch for reference."
        )

    def combine_non_differentiable_parameters(
        self, batch_size: int | None = None, **overwrite: np.ndarray
    ) -> np.ndarray:
        """Combine all non-differentiable parameters into a single numpy array.

        Args:
            batch_size: The batch size for the parameters.
                Not needed if overwrite is provided.
            **overwrite: Overwrite values for specific parameters.
                The keys should correspond to the parameter names to overwrite.
                The values need to be np.ndarray with shape ``(batch_size, N_horizon, pdim)``,
                where ``pdim`` is the number of dimensions of the parameter to overwrite.
                Expects the time dimension to be ``N_horizon + 1`` (``N`` stages 0..N).

        Returns:
            np.ndarray: shape ``(batch_size, N_horizon + 1, np)`` with ``np`` being the number of
            ``parameter_values``.
        """
        batch_size = next(iter(overwrite.values())).shape[0] if overwrite else batch_size or 1
        Np1 = self.N_horizon + 1
        expected_shape = (batch_size, Np1, self._non_differentiable_parameter_store.size)

        non_differentiable_default_flat = self._non_differentiable_parameter_store.get_values()
        if not (self._need_indicator or overwrite):
            return np.broadcast_to(non_differentiable_default_flat, expected_shape)
        batch_parameter_values = np.tile(non_differentiable_default_flat, (batch_size, Np1, 1))

        if self._need_indicator:
            s, e = self._non_differentiable_parameter_store.indices["indicator"]
            batch_parameter_values[:, :, s:e] = np.eye(Np1)

        for key, val in overwrite.items():
            s, e = self._non_differentiable_parameter_store.indices[key]
            batch_parameter_values[:, :, s:e] = val.reshape(batch_size, Np1, -1)

        assert batch_parameter_values.shape == expected_shape, (
            f"batch_parameter_values should have shape {expected_shape}, "
            f"got {batch_parameter_values.shape}."
        )
        return batch_parameter_values

    def get(self, name: str) -> ca.SX | ca.MX | np.ndarray:
        """Get the symbolic variable for a parameter.

        For stage-varying differentiable parameters (those with ``splits``), the returned
        expression is a weighted sum over all stage blocks, gated by the ``indicator`` vector
        in the non-differentiable parameters.  The expression evaluates to the correct block value
        at each stage, but **only if the indicator is set correctly** via
        :meth:`combine_non_differentiable_parameters`.  If the indicator is all-zero (e.g. the
        default), every stage silently evaluates to zero for these parameters.

        Args:
            name: The name of the parameter to retrieve.

        Returns:
            - CasADi ``SX``/``MX`` expression for ``"differentiable"`` and ``"non-differentiable"``
              parameters.

        Raises:
            ValueError: If ``name`` is not registered with this manager.
        """
        if name not in self.parameters:
            raise ValueError(f"Unknown name: {name}. Available names: {', '.join(self.parameters)}")

        if (
            self.parameters[name].interface == "differentiable"
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
                        self._non_differentiable_parameter_store.symbols["indicator"][
                            start : end + 1
                        ]
                    )
                )
                variables.append(
                    self._differentiable_parameter_store.symbols[f"{name}_{start}_{end}"]
                )

            terms = []
            for indicator, variable in zip(indicators, variables):
                terms.append(indicator * variable)
            return sum(terms)

        if self.parameters[name].interface == "differentiable":
            return self._differentiable_parameter_store.symbols[name]

        if self.parameters[name].interface == "non-differentiable":
            return self._non_differentiable_parameter_store.symbols[name]
        else:
            raise ValueError(
                f"Unknown interface type for field '{name}': {self.parameters[name].interface}"
            )

    def assign_to_ocp(self, ocp: AcadosOcp) -> None:
        """Synchronize the parameter manager's symbols and defaults onto the OCP.

        Sets ``model.p``, ``model.p_global``, ``parameter_values``, and ``p_global_values``
        on the provided acados OCP instance. Should be called before solver code generation.

        Args:
            ocp: An acados ``AcadosOcp`` instance.
        """
        self._finalized = True
        ocp.model.p = self.non_differentiable_symbols
        ocp.model.p_global = self.differentiable_symbols
        ocp.parameter_values = self.non_differentiable_default_flat
        ocp.p_global_values = self.differentiable_default_flat
