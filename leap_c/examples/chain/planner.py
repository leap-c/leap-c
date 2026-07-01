from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from leap_c.examples.chain.acados_ocp import (
    ChainInitializer,
    export_parametric_ocp,
)
from leap_c.examples.chain.dynamics import define_f_expl_expr
from leap_c.examples.chain.utils.resting_chain_solver import RestingChainSolver
from leap_c.ocp.acados.diff_mpc import collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch
from leap_c.planner import ParameterizedPlanner
from leap_c.utils.parameters import broadcast_default_param


@dataclass(kw_only=True)
class ChainControllerConfig:
    """Configuration for the Chain controller.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
            The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal
            node N).
        T_horizon: The duration of the MPC horizon. One step during planning
            will equal T_horizon/N_horizon simulation time.
        n_mass: The number of masses in the chain.
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
    T_horizon: float = 0.25
    n_mass: int = 5

    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None


class ChainPlanner(ParameterizedPlanner[AcadosDiffMpcCtx]):
    """Acados-based controller for the hanging `Chain` system.

    The state and action correspond to the observation and action of the `Chain` environment. The
    cost function takes the form of a weighted least-squares cost on the full state and action and
    the dynamics correspond to the simulated ODE also found in the `Chain` environment (using RK4).
    The inequality constraints are box constraints on the action.

    Attributes:
        cfg: A configuration object containing high-level settings for the MPC problem,
            such as horizon length.
    """

    cfg: ChainControllerConfig
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: ChainControllerConfig | None = None,
        export_directory: Path | None = None,
    ) -> None:
        """Initializes the ChainController.

        Args:
            cfg: cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length and maximum force. If not provided,
                a default config is used.
            export_directory: Directory to export the acados ocp files.
        """
        self.cfg = ChainControllerConfig() if cfg is None else cfg
        super().__init__()

        fix_point = np.zeros(3)

        # find resting reference position
        rest_length = 0.033
        pos_last_mass_ref = fix_point + np.array([rest_length * (self.cfg.n_mass - 1), 0, 0])

        dyn_param_dict = {
            "L": np.repeat([0.033, 0.033, 0.033], self.cfg.n_mass - 1),
            "D": np.repeat([1.0, 1.0, 1.0], self.cfg.n_mass - 1),
            "C": np.repeat([0.1, 0.1, 0.1], self.cfg.n_mass - 1),
            "m": np.repeat([0.033], self.cfg.n_mass - 1),
            "w": np.repeat([0.0, 0.0, 0.0], self.cfg.n_mass - 2),
        }

        resting_chain_solver = RestingChainSolver(
            n_mass=self.cfg.n_mass,
            f_expl=define_f_expl_expr,
            **dyn_param_dict,
            fix_point=fix_point,
        )

        x_ref, _ = resting_chain_solver(p_last=pos_last_mass_ref)

        ocp, param_manager, param_space, default_param = export_parametric_ocp(
            x_ref=x_ref,
            fix_point=fix_point,
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            n_mass=self.cfg.n_mass,
        )

        initializer = ChainInitializer(ocp, x_ref=x_ref)

        diff_mpc = AcadosDiffMpcTorch(
            ocp,
            param_manager,
            initializer=initializer,
            discount_factor=self.cfg.discount_factor,
            export_directory=export_directory,
            n_batch_init=self.cfg.n_batch_init,
            num_threads_batch_solver=self.cfg.num_threads_batch_solver,
            dtype=self.cfg.dtype,
        )
        self.param_manager = param_manager
        self.diff_mpc = diff_mpc
        self._param_space = param_space
        self._default_param = default_param

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        params: dict[str, torch.Tensor | np.ndarray] | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ):
        return self.diff_mpc(x0=obs, u0=action, params=params, ctx=ctx)

    def default_param(self, obs: np.ndarray | torch.Tensor | None = None) -> dict[str, np.ndarray]:
        return broadcast_default_param(self._default_param, obs)
