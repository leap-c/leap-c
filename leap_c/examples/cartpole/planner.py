from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from leap_c.examples.cartpole.acados_ocp import (
    CartPoleAcadosCostType,
    export_parametric_ocp,
)
from leap_c.ocp.acados.diff_mpc import collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch
from leap_c.planner import ParameterizedPlanner
from leap_c.utils.parameters import broadcast_default_param


@dataclass(kw_only=True)
class CartPolePlannerConfig:
    """Configuration for the CartPole planner.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
            The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal
            node N).
        T_horizon: The simulation time between two MPC nodes will equal
            T_horizon/N_horizon [s] simulation time.
        Fmax: Bounds of the box constraints on the maximum force that can be
            applied to the cart [N] (hard constraint)
        x_threshold: Bounds of the box constraints of the maximum absolute position
            of the cart [m] (soft/slacked constraint)
        cost_type: The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS".
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

    N_horizon: int = 5
    T_horizon: float = 0.25
    Fmax: float = 80.0
    x_threshold: float = 2.4

    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS"

    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None


class CartPolePlanner(ParameterizedPlanner[AcadosDiffMpcCtx]):
    """Acados-based planner for `CartPole`, aka inverted pendulum.

    The state and action correspond to the observation and action of the CartPole environment.
    The cost function takes the form of a weighted least-squares cost on the full state and action,
    and the dynamics correspond to the simulated ODE of the standard CartPole environment
    (using RK4). The inequality constraints are box constraints on the action and
    on the cart position.

    Attributes:
        cfg: A configuration object containing high-level settings for the MPC problem,
            such as horizon length.
    """

    cfg: CartPolePlannerConfig
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: CartPolePlannerConfig | None = None,
        export_directory: Path | None = None,
    ) -> None:
        """Initializes the CartPoleController.

        Args:
            cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length and maximum force. If not provided,
                a default config is used.
            export_directory: An optional directory path where the generated
                `acados` solver code will be exported.
        """
        self.cfg = CartPolePlannerConfig() if cfg is None else cfg

        super().__init__()
        ocp, param_manager, param_space, default_param = export_parametric_ocp(
            cost_type=self.cfg.cost_type,
            name="cartpole",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
            x_threshold=self.cfg.x_threshold,
        )

        diff_mpc = AcadosDiffMpcTorch(
            ocp=ocp,
            parameter_manager=param_manager,
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
        params: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ):
        if params is not None:
            params = {"xref1": params}
        return self.diff_mpc(x0=obs, u0=action, params=params, ctx=ctx)

    def default_param(self, obs: np.ndarray | torch.Tensor | None = None) -> np.ndarray:
        return broadcast_default_param(self._default_param, obs)
