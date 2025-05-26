import time
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from dataclasses import fields
from enum import IntEnum
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, List, NamedTuple

import casadi as ca
import numpy as np
import torch
from acados_template import AcadosOcp, AcadosOcpBatchSolver, AcadosOcpSolver
from acados_template.acados_multiphase_ocp import AcadosMultiphaseOcp
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
)

from leap_c.ocp.acados.file_manager import AcadosFileManager
from leap_c.ocp.acados.utils import set_standard_sensitivity_options, SX_to_labels


class AcadosStatus(IntEnum):
    ACADOS_SUCCESS = 0
    ACADOS_NAN_DETECTED = 1
    ACADOS_MAXITER = 2
    ACADOS_MINSTEP = 3
    ACADOS_QP_FAILURE = 4
    ACADOS_READY = 5
    ACADOS_UNBOUNDED = 6
    ACADOS_TIMEOUT = 7


class MpcParameter(NamedTuple):
    """
    A named tuple to store the parameters of the MPC planner.

    Attributes:
        p_global: The part of p_global that should be learned in shape (n_p_global_learnable, ) or (B, n_p_global_learnable).
        p_stagewise: The stagewise parameters in shape
            (N+1, p_stagewise_dim) or (N+1, len(p_stagewise_sparse_idx)) if the next field is set or
            (B, N+1, p_stagewise_dim) or (B, N+1, len(p_stagewise_sparse_idx)) if the next field is set.
            If a multi-phase MPC is used and p_stagewise_sparse_idx is given, this is a list containing the above arrays for the respective phases.
        p_stagewise_sparse_idx: If not None, stagewise parameters are set in a sparse manner, using these indices.
            The indices are in shape (N+1, n_p_stagewise_sparse_idx) or (B, N+1, n_p_stagewise_sparse_idx).
            If a multi-phase MPC is used this is a list containing the above arrays for the respective phases.
    """

    p_global: np.ndarray | None = None
    p_stagewise: List[np.ndarray] | np.ndarray | None = None
    p_stagewise_sparse_idx: List[np.ndarray] | np.ndarray | None = None

    def is_batched(self) -> bool:
        """The empty MpcParameter counts as non-batched."""
        if self.p_global is not None:
            return self.p_global.ndim == 2
        elif self.p_stagewise is not None:
            return self.p_stagewise[0].ndim == 3
        else:
            return False

    def get_sample(self, i: int) -> "MpcParameter":
        """Get the sample at index i from the batch."""
        if not self.is_batched():
            raise ValueError("Cannot sample from non-batched MpcParameter.")
        p_global = self.p_global[i] if self.p_global is not None else None
        p_stagewise = self.p_stagewise[i] if self.p_stagewise is not None else None
        p_stagewise_sparse_idx = (
            self.p_stagewise_sparse_idx[i]
            if self.p_stagewise_sparse_idx is not None
            else None
        )

        return MpcParameter(
            p_global=p_global,
            p_stagewise=p_stagewise,
            p_stagewise_sparse_idx=p_stagewise_sparse_idx,
        )

    def ensure_float64(self) -> "MpcParameter":
        def convert(k, v):
            if k not in ["p_stagewise_sparse_idx"]:
                return v.astype(np.float64) if v is not None else None
            return v

        kw = {k: convert(k, v) for k, v in self._asdict().items()}

        return MpcParameter(**kw)


MpcBatchedState = AcadosOcpFlattenedBatchIterate


class MpcInput(NamedTuple):
    """
    A named tuple to store the input of the MPC planner.

    Attributes:
        x0: The initial states in shape (B, x_dim) or (x_dim, ).
        u0: The initial actions in shape (B, u_dim) or (u_dim, ).
        parameters: The parameters of the MPC planner.
    """

    x0: np.ndarray
    u0: np.ndarray | None = None
    parameters: MpcParameter | None = None

    def is_batched(self) -> bool:
        return self.x0.ndim == 2

    def get_sample(self, i: int) -> "MpcInput":
        """Get the sample at index i from the batch."""
        if not self.is_batched():
            raise ValueError("Cannot sample from non-batched MPCInput.")
        x0 = self.x0[i]
        u0 = self.u0[i] if self.u0 is not None else None
        parameters = (
            self.parameters.get_sample(i) if self.parameters is not None else None
        )
        return MpcInput(x0=x0, u0=u0, parameters=parameters)


class MpcOutput(NamedTuple):
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

    status: np.ndarray | torch.Tensor | None = None  # (B, ) or (1, )
    u0: np.ndarray | torch.Tensor | None = None  # (B, u_dim) or (u_dim, )
    Q: np.ndarray | torch.Tensor | None = None  # (B, ) or (1, )
    V: np.ndarray | torch.Tensor | None = None  # (B, ) or (1, )
    dvalue_dx0: np.ndarray | None = None  # (B, x_dim) or (x_dim, )
    dvalue_du0: np.ndarray | None = None  # (B, u_dim) or (u_dim, )
    dvalue_dp_global: np.ndarray | None = None  # (B, p_dim) or (p_dim, )
    du0_dp_global: np.ndarray | None = None  # (B, udim, p_dim) or (udim, p_dim)
    du0_dx0: np.ndarray | None = None  # (B, u_dim, x_dim) or (u_dim, x_dim)


def set_ocp_solver_mpc_params_global(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    mpc_parameter: MpcParameter,
    batch_size: int | None = None,
) -> None:
    if mpc_parameter.p_global is None:
        return
    if isinstance(ocp_solver, AcadosOcpSolver):
        ocp_solver.set_p_global_and_precompute_dependencies(mpc_parameter.p_global)
    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        if batch_size is None:
            raise ValueError("batch_size must be set when using AcadosOcpBatchSolver.")
        for i, single_solver in enumerate(ocp_solver.ocp_solvers):
            if i >= batch_size:
                break
            single_solver.set_p_global_and_precompute_dependencies(
                mpc_parameter.p_global[i]
            )
    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, but got {type(ocp_solver)}."
        )


def set_ocp_solver_mpc_params_stagewise(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    mpc_parameter: MpcParameter,
    batch_size: int | None = None,
):
    if mpc_parameter.p_stagewise is None:
        return
    if isinstance(ocp_solver, AcadosOcpSolver):
        if mpc_parameter.p_stagewise_sparse_idx is not None:
            for stage, (p, idx) in enumerate(
                zip(
                    mpc_parameter.p_stagewise,
                    mpc_parameter.p_stagewise_sparse_idx,
                )
            ):
                ocp_solver.set_params_sparse(stage, p, idx)
        else:
            ocp_solver.set_flat("p", mpc_parameter.p_stagewise.reshape(-1))  # type:ignore
    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        if batch_size is None:
            raise ValueError("batch_size must be set when using AcadosOcpBatchSolver.")
        if mpc_parameter.p_stagewise_sparse_idx is None:  # not sparse
            p = mpc_parameter.p_stagewise.reshape(batch_size, -1)  # type:ignore
            ocp_solver.set_flat("p", p)
        else:
            for i, single_solver in enumerate(ocp_solver.ocp_solvers):
                if i >= batch_size:
                    break
                set_ocp_solver_mpc_params(single_solver, mpc_parameter.get_sample(i))
    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, but got {type(ocp_solver)}."
        )


def set_ocp_solver_mpc_params(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    mpc_parameter: MpcParameter | None,
    batch_size: int | None = None,
) -> None:
    if mpc_parameter is None:
        return
    set_ocp_solver_mpc_params_global(ocp_solver, mpc_parameter, batch_size)
    set_ocp_solver_mpc_params_stagewise(ocp_solver, mpc_parameter, batch_size)


def set_ocp_solver_iterate(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    ocp_iterate: MpcBatchedState | None,
) -> None:
    if ocp_iterate is None:
        return
    elif not isinstance(ocp_iterate, AcadosOcpFlattenedBatchIterate):
        raise ValueError(
            f"Expected AcadosOcpFlattenedBatchIterate, got {type(ocp_iterate)}."
        )

    if isinstance(ocp_solver, AcadosOcpSolver):
        if ocp_iterate.N_batch != 1:
            raise ValueError("Expected a batch size of 1 for a single AcadosOcpSolver.")
        single_iterate = AcadosOcpFlattenedIterate(
            x=ocp_iterate.x[0],
            u=ocp_iterate.u[0],
            z=ocp_iterate.z[0],
            pi=ocp_iterate.pi[0],
            lam=ocp_iterate.lam[0],
            sl=ocp_iterate.sl[0],
            su=ocp_iterate.su[0],
        )
        ocp_solver.load_iterate_from_flat_obj(single_iterate)

    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        ocp_solver.load_iterate_from_flat_obj(ocp_iterate)
    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, got {type(ocp_solver)}."
        )


def set_ocp_solver_initial_condition(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    mpc_input: MpcInput,
    throw_error_if_u0_is_outside_ocp_bounds: bool,
    batch_size: int | None = None,
) -> None:
    if isinstance(ocp_solver, AcadosOcpSolver):
        x0 = mpc_input.x0
        ocp_solver.set(0, "x", x0)
        ocp_solver.constraints_set(0, "lbx", x0)
        ocp_solver.constraints_set(0, "ubx", x0)
        if mpc_input.u0 is not None:
            u0 = mpc_input.u0
            ocp_solver.set(0, "u", u0)
            if throw_error_if_u0_is_outside_ocp_bounds:
                constr = (
                    ocp_solver.acados_ocp.constraints[0]
                    if isinstance(ocp_solver.acados_ocp, AcadosMultiphaseOcp)
                    else ocp_solver.acados_ocp.constraints
                )
                if constr.lbu.size != 0 and np.any(u0 < constr.lbu):
                    raise ValueError(
                        "You are about to set an initial control that is below the defined lower bound."
                    )
                elif constr.ubu.size != 0 and np.any(u0 > constr.ubu):
                    raise ValueError(
                        "You are about to set an initial control that is above the defined upper bound."
                    )
            ocp_solver.constraints_set(0, "lbu", u0)
            ocp_solver.constraints_set(0, "ubu", u0)

    elif isinstance(ocp_solver, AcadosOcpBatchSolver):
        if batch_size is None:
            raise ValueError("batch_size must be set when using AcadosOcpBatchSolver.")
        for i, ocp in enumerate(ocp_solver.ocp_solvers):
            if i >= batch_size:
                break

            set_ocp_solver_initial_condition(
                ocp,
                mpc_input.get_sample(i),
                throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
            )

    else:
        raise ValueError(
            f"expected AcadosOcpSolver or AcadosOcpBatchSolver, got {type(ocp_solver)}."
        )


def initialize_ocp_solver(
    ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver,
    mpc_input: MpcInput,
    ocp_iterate: MpcBatchedState | None,
    set_params: bool = True,
    throw_error_if_u0_is_outside_ocp_bounds: bool = True,
) -> None:
    """Initializes the fields of the OCP (batch) solver with the given values.

    Args:
        ocp_solver: The OCP (batch) solver to initialize.
        mpc_input: The MPCInput containing the parameters to set in the OCP (batch) solver
            and the state and possibly control that will be used for the initial conditions.
        ocp_iterate: The iterate of the solver to use as initialization.
        set_params: Whether to set the MPC parameters of the OCP (batch) solver.
        throw_error_if_u0_is_outside_ocp_bounds: If True, an error will be thrown when given an u0 in mpc_input that
            is outside the box constraints defined in the cop
    """
    if mpc_input.is_batched():
        batch_size = mpc_input.x0.shape[0]
    else:
        if isinstance(ocp_solver, AcadosOcpBatchSolver):
            raise ValueError("Input has to be batched when using a batch solver")
        batch_size = None

    set_ocp_solver_iterate(ocp_solver, ocp_iterate)
    if set_params:
        set_ocp_solver_mpc_params(
            ocp_solver, mpc_input.parameters, batch_size=batch_size
        )
    # Set the initial conditions after setting the iterate, in case the given iterate contains a different value
    set_ocp_solver_initial_condition(
        ocp_solver,
        mpc_input,
        throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        batch_size=batch_size,
    )


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
    default_mpc_parameters: MpcParameter,
    unset_u0: bool,
) -> None:
    """Resets the OCP (batch) solver to remove any "state" to be carried over in the next call.
    Since the init function or a given iterate is being used to override the state of the solver anyways,
    we don't need to call ocp_solver.reset().
    This entails:
    - Setting the parameters to the default values (since the default is consistent over the batch, the given MpcParameter must not be batched).
    - Unsetting the initial control constraints if they were set.
    """
    if isinstance(ocp_solver, AcadosOcpSolver):
        if unset_u0:
            unset_ocp_solver_initial_control_constraints(ocp_solver)
        set_ocp_solver_mpc_params(ocp_solver, default_mpc_parameters)

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


def _solve_shared(
    solver: AcadosOcpBatchSolver,
    sensitivity_solver: AcadosOcpBatchSolver | None,
    mpc_input: MpcInput,
    mpc_state: MpcBatchedState | None,
    backup_fn: Callable[[MpcInput], MpcBatchedState] | None,
    throw_error_if_u0_is_outside_ocp_bounds: bool = True,
) -> dict[str, Any]:
    batch_size = mpc_input.x0.shape[0]

    # Use the backup function to get an iterate in the first solve already, if no iterate is given.
    # Else no iterate is used, which means the iterate of the resetted solver is used (i.e. all iterates are set to 0).
    if mpc_state is None and backup_fn is not None:
        iterate = backup_fn(mpc_input)  # type:ignore
    else:
        iterate = mpc_state
    initialize_ocp_solver(
        ocp_solver=solver,
        mpc_input=mpc_input,
        ocp_iterate=iterate,
        throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
    )
    start = time.perf_counter()
    solver.solve(n_batch=batch_size)
    stop = time.perf_counter()

    solve_stats = dict()
    solve_stats["whole_solve"] = stop - start  # type:ignore

    stats_batch = defaultdict(list)
    status_batch = []
    backup_status_batch = []
    any_failed = False

    ocp_solvers = solver.ocp_solvers[:batch_size]

    for i, ocp_solver in enumerate(ocp_solvers):
        # TODO (Jasper): Providing batch statistics could be moved to acados
        status = ocp_solver.status
        status_batch.append(status)
        stats_batch["sqp_iter"].append(ocp_solver.get_stats("sqp_iter"))
        stats_batch["qp_iter"].append(ocp_solver.get_stats("qp_iter").sum())  # type:ignore
        stats_batch["time_tot"].append(ocp_solver.get_stats("time_tot"))
        if status != 0:
            any_failed = True

    if (
        any_failed and backup_fn is not None and iterate is not None
    ):  # Reattempt with backup
        # TODO (Jasper): Currently we do a global resolve!
        #   - It is not clear how problematic this is, as the individual solver directly converges.
        #   - Option A: We leave it like this.
        #   - Option B: We generate a new batch and load the iterate into the first n solvers.
        #   - Option C: Acados provides an idx functionality allowing to select which solvers to run.

        # build new batch
        for i, ocp_solver in enumerate(ocp_solvers):
            if status_batch[i] != 0:
                single_input = mpc_input.get_sample(i)
                initialize_ocp_solver(
                    ocp_solver=ocp_solver,
                    mpc_input=single_input,
                    ocp_iterate=backup_fn(single_input),  # type:ignore
                    set_params=False,
                    throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
                )
        start = time.perf_counter()
        solver.solve(n_batch=batch_size)
        stop = time.perf_counter()
        solve_stats["whole_solve"] += stop - start

        reattempts = 0

        for i, ocp_solver in enumerate(ocp_solvers):
            # Only update the stats if a resolve was attempted
            if status_batch[i] == 0:
                continue
            reattempts += 1
            stats_batch["sqp_iter"][i] += ocp_solver.get_stats("sqp_iter")
            stats_batch["qp_iter"][i] += ocp_solver.get_stats("qp_iter").sum()  # type:ignore
            stats_batch["time_tot"][i] += ocp_solver.get_stats("time_tot")  # type:ignore

            backup_status_batch.append(ocp_solver.status)

        solve_stats["backup_rate"] = reattempts / batch_size

    # report each status individually
    status_batch = np.array(status_batch)
    backup_status_batch = np.array(backup_status_batch)

    for i, status_enum in enumerate(AcadosStatus):
        name = status_enum.name.lower()

        n_equal = (status_batch == status_enum.value).sum()
        solve_stats[f"status_{name}"] = float(n_equal / batch_size)

        if len(backup_status_batch) > 0:
            backup_n_equal = (backup_status_batch == status_enum.value).sum()
            solve_stats[f"backup_status_{name}"] = float(
                backup_n_equal / len(backup_status_batch)
            )

    # report batch statistics with avg, min, max
    for key, values in stats_batch.items():
        values = np.array(values)
        solve_stats[f"{key}_avg"] = values.mean()
        solve_stats[f"{key}_min"] = values.min()
        solve_stats[f"{key}_max"] = values.max()

    if sensitivity_solver is not None:
        # Mask LS-parameters
        if mpc_input.parameters is not None:
            sens_input = MpcInput(
                x0=mpc_input.x0,
                u0=mpc_input.u0,
                parameters=MpcParameter(
                    p_global=mpc_input.parameters.p_global,
                    p_stagewise=mpc_input.parameters.p_stagewise,
                    p_stagewise_sparse_idx=mpc_input.parameters.p_stagewise_sparse_idx,
                ),
            )
        else:
            sens_input = mpc_input
        initialize_ocp_solver(
            ocp_solver=sensitivity_solver,
            mpc_input=sens_input,
            ocp_iterate=solver.store_iterate_to_flat_obj(),
            throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        )
        sensitivity_solver.setup_qp_matrices_and_factorize(n_batch=batch_size)

    return solve_stats


def turn_on_warmstart(acados_ocp: AcadosOcp):
    if not (
        acados_ocp.solver_options.qp_solver_warm_start
        and acados_ocp.solver_options.nlp_solver_warm_start_first_qp
        and acados_ocp.solver_options.nlp_solver_warm_start_first_qp_from_nlp
    ):
        print(
            "WARNING: Warmstart is not enabled. We will enable it for our initialization strategies to work properly."
        )
    acados_ocp.solver_options.qp_solver_warm_start = 0
    acados_ocp.solver_options.nlp_solver_warm_start_first_qp = True
    acados_ocp.solver_options.nlp_solver_warm_start_first_qp_from_nlp = True


def create_zero_init_state_fn(
    solver: AcadosOcpSolver,
) -> Callable[[MpcInput], MpcBatchedState]:
    """Create a function that initializes the solver iterate with zeros.

    Args:
        solver: The solver to initialize.

    Returns:
        The function that initializes the solver iterate with zeros.
    """
    iterate = solver.store_iterate_to_flat_obj()

    # overwrite the iterate with zeros
    for f in fields(iterate):
        n = f.name
        setattr(iterate, n, np.zeros_like(getattr(iterate, n)))  # type: ignore

    def init_state_fn(mpc_input: MpcInput) -> MpcBatchedState:
        if not mpc_input.is_batched():
            batch_size = 1
        else:
            batch_size = len(mpc_input.x0)
        kw = {}

        for f in fields(iterate):
            n = f.name
            kw[n] = np.tile(getattr(iterate, n), (batch_size, 1))

        return AcadosOcpFlattenedBatchIterate(**kw, N_batch=batch_size)

    return init_state_fn


class Mpc(ABC):
    """Mpc abstract base class."""

    def __init__(
        self,
        ocp: AcadosOcp,
        ocp_sensitivity: AcadosOcp | None = None,
        discount_factor: float | None = None,
        init_state_fn: Callable[[MpcInput], MpcBatchedState] | None = None,
        n_batch_max: int = 256,
        num_threads_in_batch_methods: int = 1,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
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
            throw_error_if_u0_is_outside_ocp_bounds: If True, an error will be thrown when given an u0 in mpc_input that
                is outside the box constraints defined in the ocp.
        """
        self.ocp = ocp

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

        # turn_on_warmstart(self.ocp_sensitivity)

        # path management
        self.afm = AcadosFileManager(export_directory)
        self.afm_batch = AcadosFileManager(export_directory)
        self.afm_sens = AcadosFileManager(export_directory_sensitivity)
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

        self.last_call_stats: dict = dict()

    @cached_property
    def ocp_solver(self) -> AcadosOcpSolver:
        return self.ocp_batch_solver.ocp_solvers[0]

    @cached_property
    def ocp_sensitivity_solver(self) -> AcadosOcpSolver:
        return self.ocp_batch_sensitivity_solver.ocp_solvers[0]

    @cached_property
    def ocp_batch_solver(self) -> AcadosOcpBatchSolver:
        ocp = self.ocp
        ocp.model.name += "_batch"  # type:ignore

        batch_solver = self.afm_batch.setup_acados_ocp_batch_solver(
            ocp, self.n_batch_max, self._num_threads_in_batch_methods
        )

        if self._discount_factor is not None:
            set_discount_factor(batch_solver, self._discount_factor)
        set_ocp_solver_to_default(
            batch_solver, self.default_full_mpcparameter, unset_u0=True
        )

        return batch_solver

    @cached_property
    def ocp_batch_sensitivity_solver(self) -> AcadosOcpBatchSolver:
        ocp = self.ocp_sensitivity
        ocp.model.name += "_batch"  # type:ignore

        batch_solver = self.afm_sens_batch.setup_acados_ocp_batch_solver(
            ocp, self.n_batch_max, self._num_threads_in_batch_methods
        )

        if self._discount_factor is not None:
            set_discount_factor(batch_solver, self._discount_factor)
        set_ocp_solver_to_default(
            batch_solver, self.default_sens_mpcparameter, unset_u0=True
        )

        return batch_solver

    @property
    def p_global_dim(self) -> int:
        """Return the dimension of p_global."""
        return (
            self.default_p_global.shape[0] if self.default_p_global is not None else 0
        )

    @property
    def num_threads_batch_methods(self) -> int:
        """The number of threads to use in the batch methods."""
        return self._num_threads_in_batch_methods

    @num_threads_batch_methods.setter
    def num_threads_batch_methods(self, n_threads):
        self._num_threads_in_batch_methods = n_threads
        self.ocp_batch_solver.num_threads_in_batch_solve = n_threads
        self.ocp_batch_sensitivity_solver.num_threads_in_batch_solve = n_threads

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
            if self.is_model_p_legal(self.ocp_sensitivity.model.p)
            else None
        )

    @cached_property
    def default_full_mpcparameter(self) -> MpcParameter:
        """Return the full default MpcParameter."""
        return MpcParameter(
            p_global=self.default_p_global,
            p_stagewise=self.default_p_stagewise,
        )

    @cached_property
    def default_sens_mpcparameter(self) -> MpcParameter:
        """Return the default MpcParameter for sensitivity solver.
        It does not contain the LS-parameters"""
        return MpcParameter(
            p_global=self.default_p_global,
            p_stagewise=self.default_p_stagewise,
        )

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

    def policy(
        self,
        state: np.ndarray,
        p_global: np.ndarray | None,
        p_stagewise: np.ndarray | None = None,
        p_stagewise_sparse_idx: np.ndarray | None = None,
        solver_state: MpcBatchedState | None = None,
    ) -> tuple[np.ndarray, MpcBatchedState, np.ndarray]:
        """Compute the policy for a given state.

        Args:
            state: The state for which to compute the policy.
            p_global: The global parameters.
            p_stagewise: The stagewise parameters.
            p_stagewise_sparse_idx: The sparse indices of the stagewise parameters.
            solver_state: The iterate of the solver to use as initialization. If None, the
                solver is initialized using its init_state_fn.

        Returns:
            The action (or control), the state and status of the solver.
        """

        def batchify(x: np.ndarray | None) -> np.ndarray:
            if x is None:
                return x
            if x.ndim != 1:
                raise ValueError("Policy input should not be batched!")

            return x.reshape(1, -1)

        state = batchify(state)

        if all(p is None for p in [p_global, p_stagewise, p_stagewise_sparse_idx]):
            param = None
        else:
            p_global = batchify(p_global)
            p_stagewise = batchify(p_stagewise)
            p_stagewise_sparse_idx = batchify(p_stagewise_sparse_idx)

            param = MpcParameter(
                p_global=p_global,
                p_stagewise=p_stagewise,
                p_stagewise_sparse_idx=p_stagewise_sparse_idx,
            )

        mpc_input = MpcInput(x0=state, parameters=param)
        mpc_output, mpc_state = self(mpc_input=mpc_input, mpc_state=solver_state)

        return mpc_output.u0[0], mpc_state, mpc_output.status[0]  # type:ignore

    def __call__(
        self,
        mpc_input: MpcInput,
        mpc_state: MpcBatchedState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MpcOutput, MpcBatchedState]:
        """Solve the OCP for the given initial state and parameters.

        If an mpc_state is given and the solver does not converge, the solver does a
        reattempt using the init_state_fn. If the mpc_state is None, the init_state_fn is
        used to initialize the solver.

        Note:
            Information of this call is stored in the public member self.last_call_stats.
            The solution state of this call is stored in the public member
                self.last_call_state.

        Args:
            mpc_input: The input of the MPC controller.
            mpc_state: The iterate of the solver to use as initialization. If None, the
                solver is initialized using its init_state_fn.
            dudx: Whether to compute the sensitivity of the action with respect to the
                state.
            dudp: Whether to compute the sensitivity of the action with respect to the
                parameters.
            dvdx: Whether to compute the sensitivity of the value function with respect
                to the state.
            dvdu: Whether to compute the sensitivity of the value function with respect
                to the action.
            dvdp: Whether to compute the sensitivity of the value function with respect
                to the parameters.
            use_adj_sens: Whether to use adjoint sensitivity.

        Returns:
            A collection of outputs from the MPC controller and the state of the solver.
        """
        batch_size = mpc_input.x0.shape[0]

        if not mpc_input.is_batched():
            raise ValueError("The input to `__call__` of MPC must be batched.")

        if mpc_input.u0 is None and dvdu:
            raise ValueError("dvdu is only allowed if u0 is set in the input.")

        use_sensitivity_solver = dudx or dudp or dvdp

        self.last_call_stats = _solve_shared(
            solver=self.ocp_batch_solver,
            sensitivity_solver=(
                self.ocp_batch_sensitivity_solver if use_sensitivity_solver else None
            ),
            mpc_input=mpc_input,
            mpc_state=mpc_state,
            backup_fn=self.init_state_fn,
            throw_error_if_u0_is_outside_ocp_bounds=self.throw_error_if_u0_is_outside_ocp_bounds,
        )

        solvers = self.ocp_batch_solver.ocp_solvers[:batch_size]
        if use_sensitivity_solver:
            sens_solvers = self.ocp_batch_sensitivity_solver.ocp_solvers[:batch_size]

        kw = {}
        kw["status"] = np.array([s.status for s in solvers])
        kw["u0"] = np.array([s.get(0, "u") for s in solvers])

        if mpc_input.u0 is not None:
            kw["Q"] = np.array([s.get_cost() for s in solvers])
        else:
            kw["V"] = np.array([s.get_cost() for s in solvers])

        if dudx:
            kw["du0_dx0"] = np.array(
                [
                    s.eval_solution_sensitivity(
                        stages=0,
                        with_respect_to="initial_state",
                        return_sens_u=True,
                        return_sens_x=False,
                    )["sens_u"]
                    for s in sens_solvers
                ]
            )

        if dudp:
            if use_adj_sens:
                single_seed = np.eye(self.ocp.dims.nu)
                seed_vec = np.repeat(single_seed[np.newaxis, :, :], batch_size, axis=0)

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
                        s.eval_solution_sensitivity(
                            stages=0,
                            with_respect_to="p_global",
                            return_sens_u=True,
                            return_sens_x=False,
                        )["sens_u"]
                        for s in sens_solvers
                    ]
                ).reshape(mpc_input.x0.shape[0], self.ocp.dims.nu, self.p_global_dim)  # type:ignore

            assert kw["du0_dp_global"].shape == (
                mpc_input.x0.shape[0],
                self.ocp.dims.nu,
                self.p_global_dim,
            )

        if dvdp:
            kw["dvalue_dp_global"] = np.array(
                [
                    s.eval_and_get_optimal_value_gradient("p_global")
                    for s in sens_solvers
                ]
            )

        if dudx:
            kw["du0_dx0"] = np.array(
                [
                    s.eval_solution_sensitivity(
                        stages=0,
                        with_respect_to="initial_state",
                        return_sens_u=True,
                        return_sens_x=False,
                    )["sens_u"]
                    for s in sens_solvers
                ]
            )
        if dvdx:
            kw["dvalue_dx0"] = np.array(
                [
                    s.eval_and_get_optimal_value_gradient(
                        with_respect_to="initial_state"
                    )
                    for s in sens_solvers
                ]
            )
        if dvdu:
            kw["dvalue_du0"] = np.array(
                [
                    s.eval_and_get_optimal_value_gradient(
                        with_respect_to="initial_control"
                    )
                    for s in sens_solvers
                ]
            )

        flat_iterate = self.ocp_batch_solver.store_iterate_to_flat_obj(
            n_batch=batch_size
        )

        # Set solvers to default
        unset_u0 = True if mpc_input.u0 is not None else False
        set_ocp_solver_to_default(
            ocp_solver=self.ocp_batch_solver,
            default_mpc_parameters=self.default_full_mpcparameter,
            unset_u0=unset_u0,
        )
        if use_sensitivity_solver:
            set_ocp_solver_to_default(
                ocp_solver=self.ocp_batch_sensitivity_solver,
                default_mpc_parameters=self.default_sens_mpcparameter,
                unset_u0=unset_u0,
            )

        return MpcOutput(**kw), flat_iterate

    def last_solve_diagnostics(
        self, ocp_solver: AcadosOcpSolver | AcadosOcpBatchSolver
    ) -> dict | list[dict]:
        """Print statistics for the last solve and collect QP-diagnostics for the solvers.

        Simpler information about the last call is stored in self.last_call_stats.
        """

        if isinstance(ocp_solver, AcadosOcpSolver):
            diagnostics = ocp_solver.qp_diagnostics()
            ocp_solver.print_statistics()
            return diagnostics
        elif isinstance(ocp_solver, AcadosOcpBatchSolver):
            diagnostics = []
            for single_solver in ocp_solver.ocp_solvers:
                diagnostics.append(single_solver.qp_diagnostics())
                single_solver.print_statistics()
            return diagnostics
        else:
            raise ValueError(
                f"Unknown solver type, expected AcadosOcpSolver or AcadosOcpBatchSolver, but got {type(ocp_solver)}."
            )
