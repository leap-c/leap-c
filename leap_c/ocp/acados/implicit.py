"""Provides a differentiable implicit function based on Acados."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from typing import Literal
import numpy as np

from acados_template import AcadosOcp
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate
from leap_c.ocp.acados.utils.file_manager import AcadosFileManager
from leap_c.ocp.acados.initializer import AcadosInitStateFunction
from leap_c.ocp.acados.utils.utils import set_standard_sensitivity_options, SX_to_labels

from leap_c.autograd.function import DiffFunction


@dataclass
class AcadosCtx:
    iterate: AcadosOcpFlattenedBatchIterate
    log: dict[str, float]
    du0_dp_global: np.ndarray | None = None
    du0_dx0: np.ndarray | None = None
    dvalue_du: np.ndarray | None = None
    dvalue_dx0: np.ndarray | None = None
    dx_dp_global: np.ndarray | None = None
    du_dp_global: np.ndarray | None = None


SensitivityField = Literal[
    "du0_dp_global",
    "du0_dx0",
    "dvalue_du",
    "dvalue_dx0",
    "dx_dp_global",
    "du_dp_global",
]


class AcadosImplicitFunction(DiffFunction):
    """Function for differentiable implicit function based on Acados."""

    def __init__(
        self,
        ocp: AcadosOcp,
        ocp_sensitivity: AcadosOcp | None = None,
        discount_factor: float | None = None,
        init_state_fn: AcadosInitStateFunction | None = None,
        n_batch_max: int = 256,
        num_threads_in_batch_methods: int = 1,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
    ):
        """
        Initialize the MPC object.

        Args:
            ocp: Optimal control problem formulation used for solving the OCP.
            ocp_sensitivity: The optimal control problem formulation to use for sensitivities.
                If None, the sensitivity problem is derived from the ocp, however only the EXTERNAL cost type is allowed then.
                For an example of how to set up other cost types refer, e.g., to examples/pendulum_on_cart.py .
            discount_factor: Discount factor. If None, acados default cost scaling is used, i.e. dt for intermediate stages, 1 for terminal stage.
            init_state_fn: Function to use as default iterate initialization for the solver. If None, the solver iterate is initialized with zeros.
            n_batch_max: Maximum batch size.
            num_threads_in_batch_methods: Number of threads to use in the batch methods.
            export_directory: Directory to export the generated code.
            export_directory_sensitivity: Directory to export the generated
                code for the sensitivity problem.
        """
        self.ocp = ocp

        # TODO (Jasper): Move this into a AcadosSolverManager class.
        if ocp_sensitivity is None:
            # setup OCP for sensitivity solver
            if (
                ocp.cost.cost_type not in ["EXTERNAL", "NONLINEAR_LS"]
                or ocp.cost.cost_type_0 not in ["EXTERNAL", "NONLINEAR_LS", None]
                or ocp.cost.cost_type_e not in ["EXTERNAL", "NONLINEAR_LS"]
            ):
                raise ValueError(
                    "Automatic derivation of sensitivity problem is only supported for EXTERNAL or NONLINEAR_LS cost types."
                )
            self.ocp_sensitivity = deepcopy(ocp)
            # TODO: check using acados if sens solver is needed, see __uses_exact_hessian in acados. Then remove linear_mpc class.

            set_standard_sensitivity_options(self.ocp_sensitivity)
        else:
            self.ocp_sensitivity = ocp_sensitivity

        if self.ocp.cost.cost_type_0 not in ["EXTERNAL", None]:
            self.ocp.translate_initial_cost_term_to_external(
                cost_hessian=ocp.solver_options.hessian_approx
            )
            self.ocp_sensitivity.translate_initial_cost_term_to_external(
                cost_hessian="EXACT"
            )

        if self.ocp.cost.cost_type not in ["EXTERNAL"]:
            self.ocp.translate_intermediate_cost_term_to_external(
                cost_hessian=ocp.solver_options.hessian_approx
            )
            self.ocp_sensitivity.translate_intermediate_cost_term_to_external(
                cost_hessian="EXACT"
            )

        if self.ocp.cost.cost_type_e not in ["EXTERNAL"]:
            self.ocp.translate_terminal_cost_term_to_external(
                cost_hessian=ocp.solver_options.hessian_approx
            )
            self.ocp_sensitivity.translate_terminal_cost_term_to_external(
                cost_hessian="EXACT"
            )

        turn_on_warmstart(self.ocp)

        # path management
        # TODO (Jasper): Make it such that we directly save the solver here.
        #                We can make it such that the solver, is stored here without any properties.
        self.afm_batch = AcadosFileManager(export_directory)
        self.afm_sens_batch = AcadosFileManager(export_directory_sensitivity)

        self._num_threads_in_batch_methods: int = num_threads_in_batch_methods
        self.n_batch_max: int = n_batch_max

        self._discount_factor = discount_factor
        if init_state_fn is None:
            ocp_solver = self.ocp_batch_solver.ocp_solvers[0]
            self.init_state_fn = create_zero_init_state_fn(ocp_solver)
        else:
            self.init_state_fn = init_state_fn

        self.param_labels = SX_to_labels(self.ocp.model.p_global)

        self.throw_error_if_u0_is_outside_ocp_bounds = (
            throw_error_if_u0_is_outside_ocp_bounds
        )

    @property
    def N(self) -> int:
        return self.ocp.solver_options.N_horizon  # type: ignore

    def forward(  # type: ignore
        self,
        x0: np.ndarray,
        u0: np.ndarray | None = None,
        p_global: np.ndarray | None = None,
        p_stagewise: np.ndarray | list[np.ndarray] | None = None,
        p_stagewise_sparse_idx: np.ndarray | list[np.ndarray] | None = None,
        ctx: AcadosCtx | None = None,
    ):
        """Performs the forward pass of the implicit function.

        Args:
            x0: Initial states with shape `(B, x_dim)`.
            u0: Initial actions with shape `(B, u_dim)`. Defaults to `None`.
            p_global: Global parameters shared across all stages,
                shape `(B, p_global_dim)`. Defaults to `None`.
            p_stagewise: Stagewise parameters.
                If `p_stagewise_sparse_idx` is `None`, shape is
                `(B, N+1, p_stagewise_dim)`.
                If `p_stagewise_sparse_idx` is provided, shape is
                `(B, N+1, len(p_stagewise_sparse_idx))`.
                For multi-phase MPC with `p_stagewise_sparse_idx`, this is
                a list of arrays (one per phase). Defaults to `None`.
            p_stagewise_sparse_idx: Indices for sparsely setting stagewise
                parameters. Shape is `(B, N+1, n_p_stagewise_sparse_idx)`.
                For multi-phase MPC, this is a list of arrays (one per phase).
                Defaults to `None`.
            ctx: An `AcadosCtx` object for storing context. Defaults to `None`.
        """
        batch_size = x0.shape[0]

        # TODO (Jasper): Add checks here.

        self.last_call_stats = _solve_shared(
            solver=self.ocp_batch_solver,
            mpc_input=mpc_input,
            mpc_state=mpc_state,
            backup_fn=self.init_state_fn,
            throw_error_if_u0_is_outside_ocp_bounds=self.throw_error_if_u0_is_outside_ocp_bounds,
        )
        solvers = self.ocp_batch_solver.ocp_solvers[:batch_size]

        status = np.array([s.status for s in solvers])
        output_u0 = np.array([s.get(0, "u") for s in solvers])
        value = np.array([s.get_cost() for s in solvers])

        flat_iterate = self.ocp_batch_solver.store_iterate_to_flat_obj(
            n_batch=batch_size
        )

        x = flat_iterate.x

        return 

    def backward(  # type: ignore
        self,
        ctx: AcadosCtx,
        u0_grad: np.ndarray | None,
        x_grad: np.ndarray | None,
        u_grad: np.ndarray | None,
        value_grad: np.ndarray | None,
    ):
        """Computes gradients of inputs given gradients of outputs.

        Args:
            ctx: The `AcadosCtx` object from the forward pass.
            x0_grad: Gradient with respect to `x0`.
            u0_grad: Gradient with respect to `u0`.
            p_global_grad: Gradient with respect to `p_global`.
            p_stagewise_idx_grad: Gradient with respect to
                `p_stagewise_sparse_idx`.
            p_stagewise_grad: Gradient with respect to `p_stagewise`.
        """
        raise NotImplementedError

    def sensitvity(self, ctx, field_name: SensitivityField):
        """Calculates a specific sensitivity field for a context.

        The `sensitivity` method retrieves a specific sensitivity field from the
        context object, or recalculates it if not already present.

        Args:
            ctx: The `AcadosCtx` object containing sensitivity information.
            field_name: The name of the sensitivity field to retrieve.
        """
        raise NotImplementedError
