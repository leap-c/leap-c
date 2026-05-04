import weakref
from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosModel, AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager


class AcadosDiffModel(AcadosModel):
    """An AcadosModel with built-in support for parameters registered via AcadosDiffOcp."""

    def __init__(self, manager: AcadosParameterManager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manager = manager

    @property
    def p(self):
        return self.manager.p

    @p.setter
    def p(self, _):
        raise AttributeError(
            "Directly setting p is not allowed. "
            "Register parameters via AcadosDiffOcp.register_param() instead."
        )

    @property
    def p_global(self):
        return self.manager.p_global

    @p_global.setter
    def p_global(self, _):
        raise AttributeError(
            "Directly setting p_global is not allowed. "
            "Use AcadosOcpSolver.set_p_global_and_precompute_dependencies() instead."
        )


class AcadosDiffOcp(AcadosOcp):
    """An AcadosOcp with built-in parameter registration for differentiable MPC.

    Requires N_horizon and optionally casadi_type at construction time so that register_param can
    eagerly build indicator expressions for split parameters.
    """

    # NOTE: We use a class-level dict to store parameter managers because `AcadosOcp` gets
    # serialized into JSON, and `AcadosParameterManager` is not JSON-serializable. The dict
    # is keyed by a generated UUID to allow multiple OCPs in the same process.
    _managers: dict[str, AcadosParameterManager] = {}

    def __init__(self, N_horizon: int, casadi_type: Literal["SX", "MX"] = "SX", **kwargs):
        super().__init__(**kwargs)
        self.solver_options.N_horizon = N_horizon
        self.casadi_type = casadi_type
        self.manager_id = id(self)
        self._managers[self.manager_id] = AcadosParameterManager(
            parameters=[], N_horizon=self.solver_options.N_horizon, casadi_type=self.casadi_type
        )
        weakref.finalize(self, self._managers.pop, self.manager_id, None)
        self.model = AcadosDiffModel(manager=self.parameter_manager)

    def register_param(
        self,
        name: str,
        default: np.ndarray,
        space: gym.spaces.Space | None = None,
        differentiable: bool = False,
        end_stages: list[int] | None = None,
    ) -> ca.SX | ca.MX:
        """Register a parameter and return a CasADi symbolic for immediate use.

        The returned symbolic is a real CasADi SX (or MX) expression (not a placeholder).
        It can be used directly in cost, dynamics, and constraint expressions.

        Args:
            name: Parameter name.
            default: Default numerical value.
            space: Gymnasium space defining valid range (for learnable params).
            differentiable: If True, parameter supports sensitivities (learnable).
                If False, parameter is changeable at runtime but not differentiable
                (non-learnable).
            end_stages: Stage variation boundaries (see AcadosParameter.end_stages).

        Returns:
            A CasADi SX (or MX) symbolic expression for immediate use in OCP formulation.
        """
        interface = "learnable" if differentiable else "non-learnable"
        param = AcadosParameter(
            name=name,
            default=default,
            space=space,
            interface=interface,
            end_stages=end_stages or [],
        )
        self.parameter_manager.add_parameter(param)
        return self.parameter_manager.get(name)

    @property
    def parameter_manager(self) -> AcadosParameterManager:
        """Return the AcadosParameterManager for this OCP."""
        return self._managers[self.manager_id]

    @property
    def parameter_values(self):
        return self.parameter_manager.non_learnable_default_flat

    @parameter_values.setter
    def parameter_values(self, _):
        raise AttributeError(
            "Directly setting parameter values is not allowed. "
            "Register parameters via AcadosDiffOcp.register_param() instead."
        )

    @property
    def p_global_values(self):
        return self.parameter_manager.learnable_default_flat

    @p_global_values.setter
    def p_global_values(self, _):
        raise AttributeError(
            "Directly setting p_global_values is not allowed. "
            "Use AcadosOcpSolver.set_p_global_and_precompute_dependencies() instead."
        )
