from dataclasses import dataclass
from pathlib import Path

from leap_c.examples.cartpole.acados_ocp import (
    CartPoleAcadosCostType,
    CartPoleAcadosParamInterface,
    create_cartpole_params,
    export_parametric_ocp,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch


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
        param_interface: Determines the exposed parameter interface of the planner.
    """

    N_horizon: int = 5
    T_horizon: float = 0.25
    Fmax: float = 80.0
    x_threshold: float = 2.4

    cost_type: CartPoleAcadosCostType = "NONLINEAR_LS"
    param_interface: CartPoleAcadosParamInterface = "global"


class CartPolePlanner(AcadosPlanner[AcadosDiffMpcCtx]):
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

    def __init__(
        self,
        cfg: CartPolePlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        """Initializes the CartPoleController.

        Args:
            cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length and maximum force. If not provided,
                a default config is used.
            params: An optional list of parameters to define the
                ocp object. If not provided, default parameters for the CartPole
                system will be created based on the cfg.
            export_directory: An optional directory path where the generated
                `acados` solver code will be exported.
        """
        self.cfg = CartPolePlannerConfig() if cfg is None else cfg
        params = (
            create_cartpole_params(
                param_interface=self.cfg.param_interface,
                N_horizon=self.cfg.N_horizon,
            )
            if params is None
            else params
        )

        param_manager = AcadosParameterManager(parameters=params, N_horizon=self.cfg.N_horizon)

        ocp = export_parametric_ocp(
            param_manager=param_manager,
            cost_type=self.cfg.cost_type,
            name="cartpole",
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            Fmax=self.cfg.Fmax,
            x_threshold=self.cfg.x_threshold,
        )

        diff_mpc = AcadosDiffMpcTorch(ocp, export_directory=export_directory)
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)
