from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager


class AcadosDiffOcp(AcadosOcp):
    """An AcadosOcp with built-in parameter registration for differentiable MPC.

    Requires N_horizon and optionally casadi_type at construction time so that register_param can
    eagerly build indicator expressions for split parameters.
    """

    _managers: dict[int, AcadosParameterManager] = {}

    def __init__(self, N_horizon: int, casadi_type: Literal["SX", "MX"] = "SX", **kwargs):
        super().__init__(**kwargs)
        self.solver_options.N_horizon = N_horizon
        self.casadi_type = casadi_type
        type(self)._managers[id(self)] = AcadosParameterManager(
            parameters=[], N_horizon=self.solver_options.N_horizon, casadi_type=self.casadi_type
        )
        self._finalized: bool = False

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
        if self._finalized:
            raise RuntimeError("Cannot register params after finalize().")

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

    def finalize(self) -> None:
        """Build the AcadosParameterManager and assign params to the OCP.

        Called internally by AcadosDiffMpc — not by the user.
        Idempotent — safe to call multiple times.
        """
        self.parameter_manager.assign_to_ocp(self)
        self._finalized = True

    @property
    def parameter_manager(self) -> AcadosParameterManager:
        """Return the AcadosParameterManager for this OCP."""
        return type(self)._managers[id(self)]

    def __del__(self):
        """Clean up the parameter manager for this OCP."""
        type(self)._managers.pop(id(self), None)
