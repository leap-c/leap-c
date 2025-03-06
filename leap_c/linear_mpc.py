from pathlib import Path
from typing import Callable

import numpy as np
from acados_template import AcadosOcp
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
)

from leap_c.mpc import (
    Mpc,
    MpcBatchedState,
    MpcInput,
    MpcOutput,
    MpcParameter,
    MpcSingleState,
    _solve_shared,
    set_ocp_solver_to_default,
)


class LinearMPC(Mpc):
    """docstring for LinearMPC."""

    def __init__(
        self,
        ocp: AcadosOcp,
        discount_factor: float | None = None,
        default_init_state_fn: Callable[[MpcInput], MpcSingleState | MpcBatchedState]
        | None = None,
        n_batch: int = 1,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        cleanup: bool = True,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
    ):
        self.ocp = ocp

        super().__init__(ocp=self.ocp, n_batch=n_batch)

    def __call__(
        self,
        mpc_input: MpcInput,
        mpc_state: MpcSingleState | MpcBatchedState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MpcOutput, MpcSingleState | MpcBatchedState]:
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
        else:
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

    def _solve(
        self,
        mpc_input: MpcInput,
        mpc_state: MpcSingleState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MpcOutput, AcadosOcpFlattenedIterate]:
        if mpc_input.u0 is None and dvdu:
            raise ValueError("dvdu is only allowed if u0 is set in the input.")

        _solve_shared(
            solver=self.ocp_solver,
            sensitivity_solver=None,
            mpc_input=mpc_input,
            mpc_state=mpc_state,
            backup_fn=self.init_state_fn,
            throw_error_if_u0_is_outside_ocp_bounds=self.throw_error_if_u0_is_outside_ocp_bounds,
        )

        kw = {}

        kw["status"] = np.array([self.ocp_solver.status])

        kw["u0"] = self.ocp_solver.get(0, "u")

        if mpc_input.u0 is not None:
            kw["Q"] = np.array([self.ocp_solver.get_cost()])
        else:
            kw["V"] = np.array([self.ocp_solver.get_cost()])

        if dudx:
            kw["du0_dx0"] = self.ocp_solver.eval_solution_sensitivity(
                stages=0,
                with_respect_to="initial_state",
                return_sens_u=True,
                return_sens_x=False,
            )["sens_u"]

        if dudp:
            if use_adj_sens:
                kw["du0_dp_global"] = self.ocp_solver.eval_adjoint_solution_sensitivity(
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
            else:
                kw["du0_dp_global"] = self.ocp_solver.eval_solution_sensitivity(
                    0,
                    "p_global",
                    return_sens_u=True,
                    return_sens_x=False,
                )["sens_u"]

        if dvdp:
            kw["dvalue_dp_global"] = (
                self.ocp_solver.eval_and_get_optimal_value_gradient("p_global")
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
        default_params = MpcParameter(
            p_global=self.default_p_global,
            p_stagewise=self.default_p_stagewise,  # type:ignore
        )
        unset_u0 = True if mpc_input.u0 is not None else False
        set_ocp_solver_to_default(
            ocp_solver=self.ocp_solver,
            default_mpc_parameters=default_params,
            unset_u0=unset_u0,
        )

        return MpcOutput(**kw), flat_iterate

    def _batch_solve(
        self,
        mpc_input: MpcInput,
        mpc_state: MpcBatchedState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MpcOutput, AcadosOcpFlattenedBatchIterate]:
        if mpc_input.u0 is None and dvdu:
            raise ValueError("dvdu is only allowed if u0 is set in the input.")

        _solve_shared(
            solver=self.ocp_batch_solver,
            sensitivity_solver=None,
            mpc_input=mpc_input,
            mpc_state=mpc_state,
            backup_fn=self.init_state_fn,
            throw_error_if_u0_is_outside_ocp_bounds=self.throw_error_if_u0_is_outside_ocp_bounds,
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

        if dudx:
            kw["du0_dx0"] = np.array(
                [
                    ocp_solver.eval_solution_sensitivity(
                        stages=0,
                        with_respect_to="initial_state",
                        return_sens_u=True,
                        return_sens_x=False,
                    )["sens_u"]
                    for ocp_solver in self.ocp_batch_solver.ocp_solvers
                ]
            )

        if dudp:
            if use_adj_sens:
                # TODO: Tested for scalar u only
                seed_vec = np.ones((self.n_batch, self.ocp.dims.nu, self.ocp.dims.nu))

                # n_seed can change when only subset of u is updated
                n_seed = self.ocp.dims.nu

                assert seed_vec.shape == (
                    self.n_batch,
                    self.ocp.dims.nu,
                    n_seed,
                )

                kw["du0_dp_global"] = (
                    self.ocp_batch_solver.eval_adjoint_solution_sensitivity(
                        seed_x=[],
                        seed_u=[(0, seed_vec)],
                        with_respect_to="p_global",
                        sanity_checks=True,
                    )
                )

            else:
                kw["du0_dp_global"] = np.array(
                    [
                        ocp_solver.eval_solution_sensitivity(
                            0,
                            "p_global",
                            return_sens_u=True,
                            return_sens_x=False,
                        )["sens_u"]
                        for ocp_solver in self.ocp_batch_solver.ocp_solvers
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
                    ocp_solver.eval_and_get_optimal_value_gradient("p_global")
                    for ocp_solver in self.ocp_batch_solver.ocp_solvers
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
        default_params = MpcParameter(
            p_global=self.default_p_global,
            p_stagewise=self.default_p_stagewise,  # type:ignore
        )
        unset_u0 = True if mpc_input.u0 is not None else False
        set_ocp_solver_to_default(
            ocp_solver=self.ocp_batch_solver,
            default_mpc_parameters=default_params,
            unset_u0=unset_u0,
        )

        return MpcOutput(**kw), flat_iterate  # type: ignore
