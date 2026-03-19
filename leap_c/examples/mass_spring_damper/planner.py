from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from leap_c.examples.mass_spring_damper.acados_ocp import (
    export_parametric_ocp,
    make_default_msd_params,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch


@dataclass(kw_only=True)
class MassSpringDamperPlannerConfig:
    """Configuration for the MassSpringDamper planner.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
            The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal
            node N).
        discount_factor: discount factor along the MPC horizon.
            If `None`, it defaults to the behavior of `AcadosOcpOptions.cost_scaling`.
        n_batch_init: Initially supported batch size of the batch OCP solver.
            Using larger batches will trigger a delay for the creation of more solvers.
            If `None`, a default value is used.
        num_threads_batch_solver: Number of parallel threads to use for the batch OCP solver.
            If `None`, a default value is used.
        dtype: Type the planner output tensors will automatically be cast to. If `None`, PyTorch
            default dtype is used.
    """

    N_horizon: int = 20

    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None


class MassSpringDamperPlanner(AcadosPlanner[AcadosDiffMpcCtx]):
    """Acados-based planner for the mass-spring-damper system.

    The state corresponds to [position, velocity] and the action is [force].
    The cost function is a quadratic cost with Q and R matrices, plus a
    terminal cost matrix P. The dynamics are discrete-time double integrator.

    Attributes:
        cfg: A configuration object containing high-level settings for the MPC problem,
            such as horizon length.
    """

    cfg: MassSpringDamperPlannerConfig

    def __init__(
        self,
        cfg: MassSpringDamperPlannerConfig | None = None,
        params: tuple[AcadosParameter, ...] | None = None,
        export_directory: Path | None = None,
    ):
        """Initializes the MassSpringDamperPlanner.

        Args:
            cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length. If not provided,
                a default config is used.
            params: An optional tuple of parameters to define the
                ocp object. If not provided, default parameters for the mass-spring-damper
                system will be created based on the cfg.
            export_directory: An optional directory path where the generated
                `acados` solver code will be exported.
        """
        self.cfg = MassSpringDamperPlannerConfig() if cfg is None else cfg
        params = make_default_msd_params(N_horizon=self.cfg.N_horizon) if params is None else params

        param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)

        ocp = export_parametric_ocp(
            param_manager=param_manager,
            N_horizon=self.cfg.N_horizon,
            name="mass_spring_damper",
            x0=np.array([1.0, 0.0]),
        )

        diff_mpc = AcadosDiffMpcTorch(
            ocp,
            discount_factor=self.cfg.discount_factor,
            export_directory=export_directory,
            n_batch_init=self.cfg.n_batch_init,
            num_threads_batch_solver=self.cfg.num_threads_batch_solver,
            dtype=self.cfg.dtype,
        )
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)
