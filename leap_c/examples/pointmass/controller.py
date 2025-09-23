from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from leap_c.examples.pointmass.acados_ocp import (
    PointMassAcadosParamInterface,
    create_pointmass_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.controller import AcadosController
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpc


@dataclass(kw_only=True)
class PointMassControllerConfig:
    """Configuration for the PointMass controller.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
        T_horizon: The duration of the MPC horizon.
        Fmax: The maximum force that can be applied to the point mass.
        param_interface: Determines the exposed parameter interface of the controller.
    """

    N_horizon: int = 20
    T_horizon: float = 2.0
    Fmax: float = 10.0
    param_interface: PointMassAcadosParamInterface = "global"


class PointMassController(AcadosController):
    """Acados-based controller for the PointMass system.
    The state corresponds to the observation of the PointMass environment, without the wind force.
    The cost function takes a weighted least-squares form,
    and the dynamics correspond to the ones in the environment, but without the wind force.
    The inequality constraints are box constraints on the action (hard)
    and on the position of the ball, the latter representing the bounds of the world (soft/slacked).

    Attributes:
        cfg: A configuration object containing high-level settings for the MPC problem,
            such as horizon length.
    """

    cfg: PointMassControllerConfig

    def __init__(
        self,
        cfg: PointMassControllerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        """Initializes the PointMassController.

        Args:
            cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length and maximum force.
                If not provided, a default config is used.
            params: An optional list of parameters to define the
                ocp object. If not provided, default parameters for the PointMass
                system will be created based on the cfg.
            export_directory: Optional directory for generated acados solver code.
        """
        super().__init__()
        self.cfg = PointMassControllerConfig() if cfg is None else cfg
        params = (
            create_pointmass_params(
                param_interface=self.cfg.param_interface,
                N_horizon=self.cfg.N_horizon,
            )
            if params is None
            else params
        )

        self.param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)

        self.ocp = export_parametric_ocp(
            param_manager=self.param_manager,
            name="pointmass",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
        )

        self.diff_mpc = AcadosDiffMpc(self.ocp, export_directory=export_directory)

    def forward(self, obs, param, ctx=None) -> tuple[Any, np.ndarray]:
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=obs.shape[0]
        )
        # remove wind field from observation, this is only observed by
        # the network, not used in the MPC
        x0 = obs[:, :4]
        ctx, u0, x, u, value = self.diff_mpc(x0, p_global=param, p_stagewise=p_stagewise, ctx=ctx)
        return ctx, u0
