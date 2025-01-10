from abc import ABC
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, List, NamedTuple

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpBatchSolver, AcadosOcpSolver
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
    AcadosOcpIterate,
)
from leap_c.util import AcadosFileManager


class MPCParameter(NamedTuple):
    """
    A named tuple to store the parameters of the MPC planner.

    Attributes:
        p_global: The part of p_global that should be learned in shape (n_p_global_learnable, ) or (B, n_p_global_learnable).
        p_stagewise: The stagewise parameters in shape
            (N+1, p_stagewise_dim) or (N+1, len(p_stagewise_sparse_idx)) if the next field is set or
            (B, N+1, p_stagewise_dim) or (B, N+1, len(p_stagewise_sparse_idx)) if the next field is set.
            If a multi-phase MPC is used this is a list containing the above arrays for the respective phases.
        p_stagewise_sparse_idx: If not None, stagewise parameters are set in a sparse manner, using these indices.
            The indices are in shape (N+1, n_p_stagewise_sparse_idx) or (B, N+1, n_p_stagewise_sparse_idx).
            If a multi-phase MPC is used this is a list containing the above arrays for the respective phases.
    """

    p_global: np.ndarray | None = None
    p_stagewise: List[np.ndarray] | np.ndarray | None = None
    p_stagewise_sparse_idx: List[np.ndarray] | np.ndarray | None = None

    def is_batched(self) -> bool:
        """The empty MPCParameter counts as non-batched."""
        if self.p_global is not None:
            return self.p_global.ndim == 2
        elif self.p_stagewise is not None:
            return self.p_stagewise[0].ndim == 3
        else:
            return False

    def get_sample(self, i: int) -> "MPCParameter":
        """Get the sample at index i from the batch."""
        if not self.is_batched():
            raise ValueError("Cannot sample from non-batched MPCParameter.")
        p_global = self.p_global[i] if self.p_global is not None else None
        p_stagewise = self.p_stagewise[i] if self.p_stagewise is not None else None
        p_stagewise_sparse_idx = (
            self.p_stagewise_sparse_idx[i]
            if self.p_stagewise_sparse_idx is not None
            else None
        )
        return MPCParameter(
            p_global=p_global,
            p_stagewise=p_stagewise,
            p_stagewise_sparse_idx=p_stagewise_sparse_idx,
        )


MPCSingleState = AcadosOcpIterate | AcadosOcpFlattenedIterate
MPCBatchedState = list[AcadosOcpIterate] | AcadosOcpFlattenedBatchIterate


class MPCInput(NamedTuple):
    """
    A named tuple to store the input of the MPC planner.

    Attributes:
        x0: The initial states in shape (B, x_dim) or (x_dim, ).
        u0: The initial actions in shape (B, u_dim) or (u_dim, ).
        parameters: The parameters of the MPC planner.
    """

    x0: np.ndarray
    u0: np.ndarray | None = None
    parameters: MPCParameter | None = None

    def is_batched(self) -> bool:
        return self.x0.ndim == 2

    def get_sample(self, i: int) -> "MPCInput":
        """Get the sample at index i from the batch."""
        if not self.is_batched():
            raise ValueError("Cannot sample from non-batched MPCInput.")
        x0 = self.x0[i]
        u0 = self.u0[i] if self.u0 is not None else None
        parameters = (
            self.parameters.get_sample(i) if self.parameters is not None else None
        )
        return MPCInput(x0=x0, u0=u0, parameters=parameters)


class MPCOutput(NamedTuple):
    """
    A named tuple to store the solution of the MPC planner.

    Attributes:
        status: The status of the solver.
        u0: The first optimal action.
        Q: The state-action value function.
        V: The value function.
        dvalue_du0: The sensitivity of the value function with respect to the initial action.
        dvalue_dp_global: The sensitivity of the value function with respect to the global parameters.
        du0_dp_global: The sensitivity of the initial action with respect to the global parameters.
        du0_dx0: The sensitivity of the initial action with respect to the initial state.
    """

    status: np.ndarray | None = None  # (B, ) or (1, )
    u0: np.ndarray | None = None  # (B, u_dim) or (u_dim, )
    Q: np.ndarray | None = None  # (B, ) or (1, )
    V: np.ndarray | None = None  # (B, ) or (1, )
    dvalue_dx0: np.ndarray | None = None  # (B, x_dim) or (x_dim, )
    dvalue_du0: np.ndarray | None = None  # (B, u_dim) or (u_dim, )
    dvalue_dp_global: np.ndarray | None = None  # (B, p_dim) or (p_dim, )
    du0_dp_global: np.ndarray | None = None  # (B, udim, p_dim) or (udim, p_dim)
    du0_dx0: np.ndarray | None = None  # (B, u_dim, x_dim) or (u_dim, x_dim)


def set_ocp_solver_mpc_params(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    mpc_parameter: MPCParameter | None,
) -> None:
    if mpc_parameter is None:
        return
    if isinstance(ocp_solver, AcadosOcpSolver):
        if mpc_parameter is not None:
            if mpc_parameter.p_global is not None:
                ocp_solver.set_p_global_and_precompute_dependencies(
                    mpc_parameter.p_global
                )

            if mpc_parameter.p_stagewise is not None:
                if mpc_parameter.p_stagewise_sparse_idx is not None:
                    for stage, (p, idx) in enumerate(
                        zip(
                            mpc_parameter.p_stagewise,
                            mpc_parameter.p_stagewise_sparse_idx,
                        )
                    ):
                        ocp_solver.set_params_sparse(stage, p, idx)
                else:
                    for stage, p in enumerate(mpc_parameter.p_stagewise):
                        ocp_solver.set(stage, "p", p)
    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        for i, single_solver in enumerate(ocp_solver.ocp_solvers):
            set_ocp_solver_mpc_params(single_solver, mpc_parameter.get_sample(i))
    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, but got {type(ocp_solver)}."
        )


def set_ocp_solver_iterate(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    ocp_iterate: MPCSingleState | MPCBatchedState | None,
) -> None:
    if ocp_iterate is None:
        return
    if isinstance(ocp_solver, AcadosOcpSolver):
        if isinstance(ocp_iterate, AcadosOcpIterate):
            ocp_solver.load_iterate_from_obj(ocp_iterate)
        elif isinstance(ocp_iterate, AcadosOcpFlattenedIterate):
            ocp_solver.load_iterate_from_flat_obj(ocp_iterate)
        elif ocp_iterate is not None:
            raise ValueError(
                f"Expected AcadosOcpIterate or AcadosOcpFlattenedIterate for an AcadosOcpSolver, got {type(ocp_iterate)}."
            )

    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        if isinstance(ocp_iterate, AcadosOcpFlattenedBatchIterate):
            ocp_solver.load_iterate_from_flat_obj(ocp_iterate)
        elif isinstance(ocp_iterate, list):
            for i, ocp_iterate in enumerate(ocp_iterate):
                ocp_solver.ocp_solvers[i].load_iterate_from_obj(ocp_iterate)
        elif ocp_iterate is not None:
            raise ValueError(
                f"Expected AcadosOcpFlattenedBatchIterate or list of AcadosOcpIterates for an AcadosOcpBatchSolver, got {type(ocp_iterate)}."
            )
    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, got {type(ocp_solver)}."
        )


def set_ocp_solver_initial_condition(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver, mpc_input: MPCInput
) -> None:
    if isinstance(ocp_solver, AcadosOcpSolver):
        ocp_solver.set(0, "x", mpc_input.x0)
        ocp_solver.constraints_set(0, "lbx", mpc_input.x0)
        ocp_solver.constraints_set(0, "ubx", mpc_input.x0)
        if mpc_input.u0 is not None:
            ocp_solver.set(0, "u", mpc_input.u0)
            ocp_solver.constraints_set(0, "lbu", mpc_input.u0)
            ocp_solver.constraints_set(0, "ubu", mpc_input.u0)

    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        for i, ocp in enumerate(ocp_solver.ocp_solvers):
            set_ocp_solver_initial_condition(ocp, mpc_input.get_sample(i))

    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, got {type(ocp_solver)}."
        )


def initialize_ocp_solver(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    mpc_input: MPCInput,
    ocp_iterate: MPCSingleState | MPCBatchedState | None,
    set_params: bool = True,
) -> None:
    """Initializes the fields of the OCP (batch) solver with the given values.

    Args:
        ocp_solver: The OCP (batch) solver to initialize.
        mpc_input: The MPCInput containing the parameters to set in the OCP (batch) solver
            and the state and possibly control that will be used for the initial conditions.
        ocp_iterate: The iterate of the solver to use as initialization.
        set_params: Whether to set the MPC parameters of the OCP (batch) solver.
    """
    set_ocp_solver_iterate(ocp_solver, ocp_iterate)
    if set_params:
        set_ocp_solver_mpc_params(ocp_solver, mpc_input.parameters)
    # Set the initial conditions after setting the iterate, in case the given iterate contains a different value
    set_ocp_solver_initial_condition(ocp_solver, mpc_input)


def unset_ocp_solver_initial_control_constraints(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
) -> None:
    """Unset the initial control constraints of the OCP (batch) solver."""

    if isinstance(ocp_solver, AcadosOcpSolver):
        ocp_solver.constraints_set(0, "lbu", ocp_solver.acados_ocp.constraints.lbu)  # type: ignore
        ocp_solver.constraints_set(0, "ubu", ocp_solver.acados_ocp.constraints.ubu)  # type: ignore

    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        for ocp_solver in ocp_solver.ocp_solvers:
            unset_ocp_solver_initial_control_constraints(ocp_solver)

    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, got {type(ocp_solver)}."
        )


def set_ocp_solver_to_default(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    default_mpc_parameters: MPCParameter,
    unset_u0: bool,
) -> None:
    """Resets the OCP (batch) solver to remove any "state" to be carried over in the next call.
    This entails:
    - Setting all Iterate-Variables to 0.
    - Setting the parameters to the default values (since the default is consistent over the batch, the given MPCParameter must not be batched).
    - Unsetting the initial control constraints if they were set.
    """
    if isinstance(ocp_solver, AcadosOcpSolver):
        if unset_u0:
            unset_ocp_solver_initial_control_constraints(ocp_solver)
        set_ocp_solver_mpc_params(ocp_solver, default_mpc_parameters)
        ocp_solver.reset()

    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        for ocp_solver in ocp_solver.ocp_solvers:
            set_ocp_solver_to_default(ocp_solver, default_mpc_parameters, unset_u0)

    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, got {type(ocp_solver)}."
        )


def set_discount_factor(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver, discount_factor: float
) -> None:
    if isinstance(ocp_solver, AcadosOcpSolver):
        for stage in range(ocp_solver.acados_ocp.solver_options.N_horizon + 1):  # type: ignore
            ocp_solver.cost_set(stage, "scaling", discount_factor**stage)

    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        for ocp_solver in ocp_solver.ocp_solvers:
            set_discount_factor(ocp_solver, discount_factor)

    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, got {type(ocp_solver)}."
        )


class MPC(ABC):
    """MPC abstract base class."""

    def __init__(
        self,
        ocp: AcadosOcp,
        discount_factor: float | None = None,
        default_init_state_fn: Callable[[MPCInput], MPCSingleState | MPCBatchedState]
        | None = None,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        cleanup: bool = True,
        n_batch: int = 1,
    ):
        """
        Initialize the MPC object.

        Args:
            ocp: Optimal control problem.
            discount_factor: Discount factor. If None, acados default cost scaling is used, i.e. dt for intermediate stages, 1 for terminal stage.
            default_init_state_fn: Function to use as default iterate initialization for the solver. If None, the solver iterate is initialized with zeros.
            export_directory: Directory to export the generated code.
            export_directory_sensitivity: Directory to export the generated
                code for the sensitivity problem.
            cleanup: Whether to clean up the export directory on exit or
                when the object is deleted.
            n_batch: Number of batched solvers to use.
        """
        self.ocp = ocp

        # setup OCP for sensitivity solver
        self.ocp_sensitivity = deepcopy(ocp)
        self.ocp_sensitivity.translate_cost_to_external_cost()
        self.ocp_sensitivity.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp_sensitivity.solver_options.globalization_fixed_step_length = 0.0
        self.ocp_sensitivity.solver_options.nlp_solver_max_iter = 1
        self.ocp_sensitivity.solver_options.qp_solver_iter_max = 200
        self.ocp_sensitivity.solver_options.tol = self.ocp.solver_options.tol / 1e3
        self.ocp_sensitivity.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp_sensitivity.solver_options.qp_solver_ric_alg = 1
        self.ocp_sensitivity.solver_options.qp_solver_cond_N = (
            self.ocp.solver_options.N_horizon
        )
        self.ocp_sensitivity.solver_options.hessian_approx = "EXACT"
        self.ocp_sensitivity.solver_options.with_solution_sens_wrt_params = True
        self.ocp_sensitivity.solver_options.with_value_sens_wrt_params = True
        self.ocp_sensitivity.model.name += "_sensitivity"  # type:ignore

        # path management
        self.afm = AcadosFileManager(export_directory)
        self.afm_batch = AcadosFileManager(export_directory)
        self.afm_sens = AcadosFileManager(export_directory_sensitivity)
        self.afm_sens_batch = AcadosFileManager(export_directory_sensitivity)

        self._discount_factor = discount_factor
        self._default_init_state_fn = default_init_state_fn

        # size of solver batch
        self.n_batch = n_batch

        # constraints and cost functions
        self._h_fn = None
        self._cost_fn = None

    @cached_property
    def ocp_solver(self) -> AcadosOcpSolver:
        solver = self.afm.setup_acados_ocp_solver(self.ocp)

        if self._discount_factor is not None:
            set_discount_factor(solver, self._discount_factor)

        default_params = MPCParameter(self.default_p_global, self.default_p_stagewise)
        set_ocp_solver_to_default(solver, default_params, unset_u0=True)

        return solver

    @cached_property
    def ocp_sensitivity_solver(self) -> AcadosOcpSolver:
        solver = self.afm_sens.setup_acados_ocp_solver(self.ocp_sensitivity)

        if self._discount_factor is not None:
            set_discount_factor(solver, self._discount_factor)

        default_params = MPCParameter(self.default_p_global, self.default_p_stagewise)
        set_ocp_solver_to_default(solver, default_params, unset_u0=True)

        return solver

    @cached_property
    def ocp_batch_solver(self) -> AcadosOcpBatchSolver:
        ocp = deepcopy(self.ocp)
        ocp.model.name += "_batch"  # type:ignore

        batch_solver = self.afm_batch.setup_acados_ocp_batch_solver(ocp, self.n_batch)

        default_params = MPCParameter(self.default_p_global, self.default_p_stagewise)
        if self._discount_factor is not None:
            set_discount_factor(batch_solver, self._discount_factor)
        set_ocp_solver_to_default(batch_solver, default_params, unset_u0=True)

        return batch_solver

    @cached_property
    def ocp_batch_sensitivity_solver(self) -> AcadosOcpBatchSolver:
        ocp = deepcopy(self.ocp_sensitivity)
        ocp.model.name += "_batch"  # type:ignore

        batch_solver = self.afm_sens_batch.setup_acados_ocp_batch_solver(
            ocp, self.n_batch
        )

        default_params = MPCParameter(self.default_p_global, self.default_p_stagewise)
        if self._discount_factor is not None:
            set_discount_factor(batch_solver, self._discount_factor)
        set_ocp_solver_to_default(batch_solver, default_params, unset_u0=True)

        return batch_solver

    @property
    def p_global_dim(self) -> int:
        """Return the dimension of p_global."""
        return (
            self.default_p_global.shape[0] if self.default_p_global is not None else 0
        )

    @property
    def N(self) -> int:
        return self.ocp.solver_options.N_horizon  # type: ignore

    @cached_property
    def default_p_global(self) -> np.ndarray | None:
        """Return the default p_global."""
        return (
            self.ocp.p_global_values
            if self.is_model_p_legal(self.ocp.model.p_global)
            else None
        )

    @cached_property
    def default_p_stagewise(self) -> np.ndarray | None:
        """Return the default p_stagewise."""
        return (
            np.tile(self.ocp.parameter_values, (self.N + 1, 1))
            if self.is_model_p_legal(self.ocp.model.p)
            else None
        )

    @property
    def default_init_state_fn(
        self,
    ) -> Callable[[MPCInput], MPCSingleState | MPCBatchedState] | None:
        return self._default_init_state_fn

    @default_init_state_fn.setter
    def default_init_state_fn(
        self, value: Callable[[MPCInput], MPCSingleState | MPCBatchedState] | None
    ) -> None:
        self._default_init_state_fn = value

    def is_model_p_legal(self, model_p: Any) -> bool:
        if model_p is None:
            return False
        elif isinstance(model_p, ca.SX):
            return 0 not in model_p.shape
        elif isinstance(model_p, np.ndarray):
            return model_p.size != 0
        elif isinstance(model_p, list) or isinstance(model_p, tuple):
            return len(model_p) != 0
        else:
            raise ValueError(f"Unknown case for model_p, type is {type(model_p)}")

    def state_value(
        self, state: np.ndarray, p_global: np.ndarray | None, sens: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Compute the value function for the given state.

        Args:
            state: The state for which to compute the value function.

        Returns:
            The value function, dvalue_dp_global if requested, and the status of the computation (whether it succeded, etc.).
        """

        mpc_input = MPCInput(x0=state, parameters=MPCParameter(p_global=p_global))
        mpc_output, _ = self.__call__(mpc_input=mpc_input, dvdp=sens)

        return (
            mpc_output.V,
            mpc_output.dvalue_dp_global,
            mpc_output.status,
        )  # type:ignore

    def state_action_value(
        self,
        state: np.ndarray,
        action: np.ndarray,
        p_global: np.ndarray | None,
        sens: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Compute the state-action value function for the given state and action.

        Args:
            state: The state for which to compute the value function.
            action: The action for which to compute the value function.

        Returns:
            The state-action value function, dQ_dp_global if requested, and the status of the computation (whether it succeded, etc.).
        """

        mpc_input = MPCInput(
            x0=state, u0=action, parameters=MPCParameter(p_global=p_global)
        )
        mpc_output, _ = self.__call__(mpc_input=mpc_input, dvdp=sens)

        return (
            mpc_output.Q,
            mpc_output.dvalue_dp_global,
            mpc_output.status,
        )  # type:ignore

    def policy(
        self,
        state: np.ndarray,
        p_global: np.ndarray | None,
        sens: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Compute the policy for the given state.

        Args:
            state: The state for which to compute the policy.
            p_global: The global parameters.
            sens: Whether to compute the sensitivity of the policy with respect to the parameters.
            use_adj_sens: Whether to use adjoint sensitivity.
        Returns:
            The policy, du0_dp_global if requested, and the status of the computation (whether it succeded, etc.).
        """

        mpc_input = MPCInput(x0=state, parameters=MPCParameter(p_global=p_global))

        mpc_output, _ = self.__call__(
            mpc_input=mpc_input, dudp=sens, use_adj_sens=use_adj_sens
        )

        return mpc_output.u0, mpc_output.du0_dp_global, mpc_output.status  # type:ignore

    def __call__(
        self,
        mpc_input: MPCInput,
        mpc_state: MPCSingleState | MPCBatchedState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MPCOutput, MPCSingleState | MPCBatchedState]:
        """
        Solve the OCP for the given initial state and parameters. If an mpc_state is given and the solver does not converge,
        AND the default_init_state_fn is not None, the solver will attempt another solve reinitialized with the default_init_state_fn
        (in the batched solve, only the non-converged samples will be reattempted to solve).

        Args:
            mpc_input: The input of the MPC controller.
            mpc_state: The iterate of the solver to use as initialization. If None, the solver is initialized using its default_init_state_fn.
            dudx: Whether to compute the sensitivity of the action with respect to the state.
            dudp: Whether to compute the sensitivity of the action with respect to the parameters.
            dvdx: Whether to compute the sensitivity of the value function with respect to the state.
            dvdu: Whether to compute the sensitivity of the value function with respect to the action.
            dvdp: Whether to compute the sensitivity of the value function with respect to the parameters.
            use_adj_sens: Whether to use adjoint sensitivity.

        Returns:
            mpc_output: The output of the MPC controller.
            mpc_state: The iterate of the solver.
        """

        if not mpc_input.is_batched():
            return self._solve(
                mpc_input=mpc_input,
                mpc_state=mpc_state,  # type: ignore
                dudx=dudx,
                dudp=dudp,
                dvdx=dvdx,
                dvdu=dvdu,
                dvdp=dvdp,
                use_adj_sens=use_adj_sens,
            )

        return self._batch_solve(
            mpc_input=mpc_input,
            mpc_state=mpc_state,  # type: ignore
            dudx=dudx,
            dudp=dudp,
            dvdx=dvdx,
            dvdu=dvdu,
            dvdp=dvdp,
            use_adj_sens=use_adj_sens,
        )

    @staticmethod
    def __solve_shared(
        solver: AcadosOcpSolver | AcadosOcpBatchSolver,
        sensitivity_solver: AcadosOcpSolver | AcadosOcpBatchSolver | None,
        mpc_input: MPCInput,
        mpc_state: MPCSingleState | MPCBatchedState | None,
        backup_fn: Callable[[MPCInput], MPCSingleState | MPCBatchedState] | None,
    ):
        initialize_ocp_solver(
            ocp_solver=solver,
            mpc_input=mpc_input,
            ocp_iterate=mpc_state,
        )
        # TODO: Cover case where we do not want to do a forward evaluation
        solver.solve()

        if backup_fn is not None:
            if isinstance(solver, AcadosOcpSolver):
                if solver.status != 0:
                    initialize_ocp_solver(
                        ocp_solver=solver,
                        mpc_input=mpc_input,
                        ocp_iterate=backup_fn(mpc_input),
                        set_params=False,
                    )
                    solver.solve()
            elif isinstance(solver, AcadosOcpBatchSolver):
                any_failed = False
                for i, ocp_solver in enumerate(solver.ocp_solvers):
                    if ocp_solver.status != 0:
                        single_input = mpc_input.get_sample(i)
                        initialize_ocp_solver(
                            ocp_solver=ocp_solver,
                            mpc_input=single_input,
                            ocp_iterate=backup_fn(single_input),
                            set_params=False,
                        )
                        any_failed = True
                if any_failed:
                    solver.solve()
            else:
                raise ValueError(
                    f"expected AcadosOcpSolver or AcadosOcpBatchSolver, got {type(solver)}."
                )

        if sensitivity_solver is not None:
            initialize_ocp_solver(
                ocp_solver=sensitivity_solver,
                mpc_input=mpc_input,  # type: ignore
                ocp_iterate=solver.store_iterate_to_flat_obj(),
            )
            sensitivity_solver.solve()

    def _solve(
        self,
        mpc_input: MPCInput,
        mpc_state: MPCSingleState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MPCOutput, AcadosOcpFlattenedIterate]:
        if mpc_input.u0 is None and dvdu:
            raise ValueError("dvdu is only allowed if u0 is set in the input.")

        use_sensitivity_solver = dudx or dudp or dvdp

        MPC.__solve_shared(
            solver=self.ocp_solver,
            sensitivity_solver=self.ocp_sensitivity_solver
            if use_sensitivity_solver
            else None,
            mpc_input=mpc_input,
            mpc_state=mpc_state,
            backup_fn=self.default_init_state_fn,
        )

        kw = {}

        kw["status"] = np.array([self.ocp_solver.status])

        kw["u0"] = self.ocp_solver.get(0, "u")

        if mpc_input.u0 is not None:
            kw["Q"] = np.array([self.ocp_solver.get_cost()])
        else:
            kw["V"] = np.array([self.ocp_solver.get_cost()])

        if use_sensitivity_solver:
            if dudx:
                kw["du0_dx0"] = self.ocp_sensitivity_solver.eval_solution_sensitivity(
                    stages=0,
                    with_respect_to="initial_state",
                    return_sens_u=True,
                    return_sens_x=False,
                )["sens_u"]

            if dudp:
                if use_adj_sens:
                    kw["du0_dp_global"] = (
                        self.ocp_sensitivity_solver.eval_adjoint_solution_sensitivity(
                            seed_x=[],
                            seed_u=[
                                (
                                    0,
                                    np.eye(self.ocp.dims.nu),  # type:ignore
                                )
                            ],
                            with_respect_to="p_global",
                            sanity_checks=True,
                        )
                    )
                else:
                    kw["du0_dp_global"] = (
                        self.ocp_sensitivity_solver.eval_solution_sensitivity(
                            0,
                            "p_global",
                            return_sens_u=True,
                            return_sens_x=False,
                        )["sens_u"]
                    )

            if dvdp:
                kw["dvalue_dp_global"] = (
                    self.ocp_sensitivity_solver.eval_and_get_optimal_value_gradient(
                        "p_global"
                    )
                )

        if dvdx:
            kw["dvalue_dx0"] = self.ocp_solver.eval_and_get_optimal_value_gradient(
                with_respect_to="initial_state"
            )

        # NB: Assumes we are evaluating dQdu0 here
        if dvdu:
            kw["dvalue_du0"] = self.ocp_solver.eval_and_get_optimal_value_gradient(
                with_respect_to="initial_control"
            )

        # get mpc state
        flat_iterate = self.ocp_solver.store_iterate_to_flat_obj()

        # Set solvers to default
        default_params = MPCParameter(
            p_global=self.default_p_global,
            p_stagewise=self.default_p_stagewise,  # type:ignore
        )
        unset_u0 = True if mpc_input.u0 is not None else False
        set_ocp_solver_to_default(
            ocp_solver=self.ocp_solver,
            default_mpc_parameters=default_params,
            unset_u0=unset_u0,
        )
        if use_sensitivity_solver:
            set_ocp_solver_to_default(
                ocp_solver=self.ocp_sensitivity_solver,
                default_mpc_parameters=default_params,
                unset_u0=unset_u0,
            )

        return MPCOutput(**kw), flat_iterate

    def _batch_solve(
        self,
        mpc_input: MPCInput,
        mpc_state: MPCBatchedState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MPCOutput, AcadosOcpFlattenedBatchIterate]:
        if mpc_input.u0 is None and dvdu:
            raise ValueError("dvdu is only allowed if u0 is set in the input.")

        use_sensitivity_solver = dudx or dudp or dvdp

        MPC.__solve_shared(
            solver=self.ocp_batch_solver,
            sensitivity_solver=self.ocp_batch_sensitivity_solver
            if use_sensitivity_solver
            else None,
            mpc_input=mpc_input,
            mpc_state=mpc_state,
            backup_fn=self.default_init_state_fn,
        )

        kw = {}
        kw["status"] = np.array(
            [ocp_solver.status for ocp_solver in self.ocp_batch_solver.ocp_solvers]
        )

        kw["u0"] = np.array(
            [ocp_solver.get(0, "u") for ocp_solver in self.ocp_batch_solver.ocp_solvers]
        )

        if mpc_input.u0 is not None:
            kw["Q"] = np.array(
                [
                    ocp_solver.get_cost()
                    for ocp_solver in self.ocp_batch_solver.ocp_solvers
                ]
            )
        else:
            kw["V"] = np.array(
                [
                    ocp_solver.get_cost()
                    for ocp_solver in self.ocp_batch_solver.ocp_solvers
                ]
            )

        if use_sensitivity_solver:
            if dudx:
                kw["du0_dx0"] = np.array(
                    [
                        ocp_sensitivity_solver.eval_solution_sensitivity(
                            stages=0,
                            with_respect_to="initial_state",
                            return_sens_u=True,
                            return_sens_x=False,
                        )["sens_u"]
                        for ocp_sensitivity_solver in self.ocp_batch_sensitivity_solver.ocp_solvers
                    ]
                )

            if dudp:
                if use_adj_sens:
                    # TODO: Tested for scalar u only
                    seed_vec = np.ones((self.n_batch, self.ocp.dims.nu, 1))

                    # n_seed can change when only subset of u is updated
                    n_seed = self.ocp.dims.nu

                    assert seed_vec.shape == (
                        self.n_batch,
                        self.ocp.dims.nu,
                        n_seed,
                    )

                    kw["du0_dp_global"] = (
                        self.ocp_batch_sensitivity_solver.eval_adjoint_solution_sensitivity(
                            seed_x=[],
                            seed_u=[(0, seed_vec)],
                            with_respect_to="p_global",
                            sanity_checks=True,
                        )
                    )

                else:
                    kw["du0_dp_global"] = np.array(
                        [
                            ocp_sensitivity_solver.eval_solution_sensitivity(
                                0,
                                "p_global",
                                return_sens_u=True,
                                return_sens_x=False,
                            )["sens_u"]
                            for ocp_sensitivity_solver in self.ocp_batch_sensitivity_solver.ocp_solvers
                        ]
                    ).reshape(self.n_batch, self.ocp.dims.nu, self.p_global_dim)

                assert kw["du0_dp_global"].shape == (
                    self.n_batch,
                    self.ocp.dims.nu,
                    self.p_global_dim,
                )

            if dvdp:
                kw["dvalue_dp_global"] = np.array(
                    [
                        ocp_sensitivity_solver.eval_and_get_optimal_value_gradient(
                            "p_global"
                        )
                        for ocp_sensitivity_solver in self.ocp_batch_sensitivity_solver.ocp_solvers
                    ]
                )

        if dudx:
            kw["du0_dx0"] = np.array(
                [
                    ocp_solver.eval_solution_sensitivity(
                        0,
                        with_respect_to="initial_state",
                        return_sens_u=True,
                        return_sens_x=False,
                    )["sens_u"]
                    for ocp_solver in self.ocp_batch_solver.ocp_solvers
                ]
            )
        if dvdx:
            kw["dvalue_dx0"] = np.array(
                [
                    solver.eval_and_get_optimal_value_gradient(
                        with_respect_to="initial_state"
                    )
                    for solver in self.ocp_batch_solver.ocp_solvers
                ]
            )
        if dvdu:
            kw["dvalue_du0"] = np.array(
                [
                    solver.eval_and_get_optimal_value_gradient(
                        with_respect_to="initial_control"
                    )
                    for solver in self.ocp_batch_solver.ocp_solvers
                ]
            )

        # TODO here we return a batch iterate object
        flat_iterate = self.ocp_batch_solver.store_iterate_to_flat_obj()

        # Set solvers to default
        default_params = MPCParameter(
            p_global=self.default_p_global,
            p_stagewise=self.default_p_stagewise,  # type:ignore
        )
        unset_u0 = True if mpc_input.u0 is not None else False
        set_ocp_solver_to_default(
            ocp_solver=self.ocp_batch_solver,
            default_mpc_parameters=default_params,
            unset_u0=unset_u0,
        )
        if use_sensitivity_solver:
            set_ocp_solver_to_default(
                ocp_solver=self.ocp_batch_sensitivity_solver,
                default_mpc_parameters=default_params,
                unset_u0=unset_u0,
            )

        return MPCOutput(**kw), flat_iterate  # type: ignore

    def last_solve_diagnostics(
        self, ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver
    ) -> dict | list[dict]:
        """Print statistics for the last solve and collect QP-diagnostics for the solvers."""

        if isinstance(ocp_solver, AcadosOcpSolver):
            diagnostics = ocp_solver.qp_diagnostics()
            ocp_solver.print_statistics()
            return diagnostics
        elif isinstance(ocp_solver, AcadosOcpBatchSolver):
            diagnostics = []
            for i, single_solver in enumerate(ocp_solver.ocp_solvers):
                diagnostics.append(single_solver.qp_diagnostics())
                single_solver.print_statistics()
            return diagnostics
        else:
            raise ValueError(
                f"Unknown solver type, expected AcadosOcpSolver or AcadosOcpBatchSolver, but got {type(ocp_solver)}."
            )

    def fetch_param(
        self,
        mpc_param: MPCParameter | None = None,
        stage: int = 0,
    ) -> tuple[None | np.ndarray, None | np.ndarray]:
        """
        Fetch the parameters for the given stage.

        Args:
            mpc_param: The parameters.
            stage: The stage.

        Returns:
            The parameters for the given stage.
        """
        p_global = self.default_p_global
        p_stage = (
            self.default_p_stagewise[stage]
            if self.default_p_stagewise is not None
            else None
        )

        if mpc_param is not None:
            if mpc_param.p_global is not None:
                p_global = mpc_param.p_global

            if mpc_param.p_stagewise is not None:
                if mpc_param.p_stagewise_sparse_idx is None:
                    p_stage = mpc_param.p_stagewise[stage]
                else:
                    p_stage = p_stage.copy()
                    p_stage[mpc_param.p_stagewise_sparse_idx[stage]] = (
                        mpc_param.p_stagewise[stage]
                    )

        return p_global, p_stage

    def stage_cons(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: MPCParameter | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Get the value of the stage constraints.

        Args:
            x: State.
            u: Control.
            p: Parameters.

        Returns:
            stage_cons: Stage constraints.
        """
        assert x.ndim == 1 and u.ndim == 1

        def relu(value):
            return value * (value > 0)

        cons = {}

        # state constraints
        if self.ocp.constraints.lbx.size > 0:
            cons["lbx"] = relu(self.ocp.constraints.lbx - x[self.ocp.constraints.idxbx])
        if self.ocp.constraints.ubx.size > 0:
            cons["ubx"] = relu(x[self.ocp.constraints.idxbx] - self.ocp.constraints.ubx)
        # control constraints
        if self.ocp.constraints.lbu.size > 0:
            cons["lbu"] = relu(self.ocp.constraints.lbu - u[self.ocp.constraints.idxbu])
        if self.ocp.constraints.ubu.size > 0:
            cons["ubu"] = relu(u[self.ocp.constraints.idxbu] - self.ocp.constraints.ubu)

        # h constraints
        if self.ocp.model.con_h_expr != []:
            if self._h_fn is None:
                inputs = [self.ocp.model.x, self.ocp.model.u]

                if self.default_p_global is not None:
                    inputs.append(self.ocp.model.p_global)

                if self.default_p_stagewise is not None:
                    inputs.append(self.ocp.model.p)  # type: ignore

                self._h_fn = ca.Function("h", inputs, [self.ocp.model.con_h_expr])

            inputs = [x, u]

            p_global, p_stage = self.fetch_param(p)
            if p_global is not None:
                inputs.append(p_global)

            if p_stage is not None:
                inputs.append(p_stage)

            h = self._h_fn(*inputs)
            cons["lh"] = relu(self.ocp.constraints.lh - h)
            cons["uh"] = relu(h - self.ocp.constraints.uh)

        # Todo (Jasper): Add phi constraints.

        return cons

    def stage_cost(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: MPCParameter | None = None,
    ) -> float:
        """
        Get the value of the stage cost.

        Args:
            x: State.
            u: Control.
            p: Parameters.

        Returns:
            stage_cost: Stage cost.
        """
        assert self.ocp.cost.cost_type == "EXTERNAL"
        assert x.ndim == 1 and u.ndim == 1

        if self._cost_fn is None:
            inputs = [self.ocp.model.x, self.ocp.model.u]

            if self.default_p_global is not None:
                inputs.append(self.ocp.model.p_global)

            if self.default_p_stagewise is not None:
                inputs.append(self.ocp.model.p)

            self._cost_fn = ca.Function(
                "cost", inputs, [self.ocp.model.cost_expr_ext_cost]
            )

        inputs = [x, u]

        p_global, p_stage = self.fetch_param(p)

        if p_global is not None:
            inputs.append(p_global)

        if p_stage is not None:
            inputs.append(p_stage)

        return self._cost_fn(*inputs).full().item()  # type: ignore


# def sequential_batch_solve(
#     mpc: MPC,
#     mpc_input: MPCInput,
#     mpc_state_given: list[MPCState] | None = None,
#     dudx: bool = False,
#     dudp: bool = False,
#     dvdx: bool = False,
#     dvdu: bool = False,
#     dvdp: bool = False,
#     use_adj_sens: bool = True,
# ) -> tuple[MPCOutput, list[MPCState]]:
#     """Perform one solve after another for every sample of the batch (contrary to the parallelized batch_solve of the mpc class).
#     Useful for debugging and timing.
#     """
#
#     def get_idx(data, index):
#         if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
#             elem_type = type(data)
#             return elem_type(*(get_idx(elem, index) for elem in data))  # type: ignore
#
#         return None if data is None else data[index]
#
#     batch_size = mpc_input.x0.shape[0]
#     outputs = []
#     states = []
#
#     for idx in range(batch_size):
#         mpc_output, mpc_state = mpc._solve(
#             mpc_input=mpc_input.get_sample(idx),
#             mpc_state=mpc_state_given[idx] if mpc_state_given is not None else None,
#             dudx=dudx,
#             dudp=dudp,
#             dvdp=dvdp,
#             dvdx=dvdx,
#             dvdu=dvdu,
#             use_adj_sens=use_adj_sens,
#         )
#
#         outputs.append(mpc_output)
#         states.append(mpc_state)
#
#     def collate(key):
#         value = getattr(outputs[0], key)
#         if value is None:
#             return value
#         return np.stack([getattr(output, key) for output in outputs])
#
#     fields = {key: collate(key) for key in MPCOutput._fields}
#     fields["status"] = fields["status"].squeeze()
#     fields["V"] = fields["V"].squeeze() if fields["V"] is not None else None
#     fields["Q"] = fields["Q"].squeeze() if fields["Q"] is not None else None
#     mpc_output = MPCOutput(**fields)  # type: ignore
#
#     return mpc_output, states  # type: ignore
