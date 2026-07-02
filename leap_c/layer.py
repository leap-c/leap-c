"""Module defining the abstract interface for differentiable, parameterized MPC layers."""

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from acados_template import AcadosOcp

from leap_c.diff_mpc.function import AcadosDiffMpcCtx, AcadosDiffMpcFunction
from leap_c.diff_mpc.initializer import AcadosDiffMpcInitializer
from leap_c.parameters.base import AcadosParameterManager

TensorType = TypeVar("TensorType")


class AcadosDiffMpcLayer(Generic[TensorType], metaclass=ABCMeta):
    """Abstract base class for differentiable MPC layers.

    This module wraps acados solvers to enable their use in differentiable machine learning
    pipelines.

    Accepts a plain ``AcadosOcp`` together with a parameter manager. The parameter manager's
    :meth:`~AcadosParameterManager.combine_differentiable_parameters_*` is called in the forward
    pass to build a flat differentiable tensor from the ``params`` dict.

    Attributes:
        diff_mpc_fun: The differentiable MPC function wrapper for acados.
        parameter_manager: An instance of the parameter manager.
    """

    diff_mpc_fun: AcadosDiffMpcFunction
    parameter_manager: AcadosParameterManager

    def __init__(
        self,
        ocp: AcadosOcp,
        parameter_manager: AcadosParameterManager,
        initializer: AcadosDiffMpcInitializer | None = None,
        discount_factor: float | None = None,
        export_directory: Path | None = None,
        n_batch_init: int | None = None,
        num_threads_batch_solver: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Initializes the AcadosDiffMpcLayer module.

        Calls ``parameter_manager.assign_to_ocp(ocp)`` to synchronise CasADi symbols and default
        values onto the OCP, then creates the solvers.

        Args:
            ocp: The acados OCP object.  Must not yet have ``model.p`` / ``model.p_global`` set
                (they will be set by ``assign_to_ocp``).
            parameter_manager: A parameter manager with registered parameters.
            initializer: The initializer used to provide initial guesses for the solver.
                Uses a zero iterate by default.
            discount_factor: An optional discount factor for the sensitivity problem.
            export_directory: An optional directory for generated C code.
            n_batch_init: Initially supported batch size.  If ``None``, a default is used.
            num_threads_batch_solver: Number of parallel threads for the batch solver.
            verbose: Whether to print solver generation output.

        """
        super().__init__()
        parameter_manager.assign_to_ocp(ocp)
        self.diff_mpc_fun = AcadosDiffMpcFunction(
            ocp=ocp,
            initializer=initializer,
            discount_factor=discount_factor,
            export_directory=export_directory,
            n_batch_init=n_batch_init,
            num_threads_batch_solver=num_threads_batch_solver,
            verbose=verbose,
        )
        self.parameter_manager = parameter_manager

    @abstractmethod
    def forward(
        self,
        x0: TensorType,
        u0: TensorType | None = None,
        params: dict[str, TensorType | np.typing.NDArray] | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, TensorType | None, TensorType, TensorType, TensorType]:
        """Performs the forward pass by solving the provided problem instances.

        Builds flat ``p_global`` and ``p_stagewise`` arrays from the ``params`` dict using the
        parameter manager, then calls the underlying differentiable MPC function. Gradients flow
        back through ``p_global`` to the individual parameters provided in ``params``.

        Args:
            x0: Initial observation input (e.g., state vector).
            u0: Optional initial action.
            params: Parameters that define the behavior of the MPC, as a dictionary keyed by the
                parameter names. Default values are used for any parameter missing from the
                dictionary. If the whole dictionary is not provided, all parameters are set to
                their default values.
            ctx: Optional internal context passed between invocations.

        Returns:
            ctx: A context object containing any intermediate values needed for backward
                computation and further invocations.
            u0: The computed initial control action. Is ``None`` if the action is already provided.
            x: The computed sequence of states. Expected shape ``(N+1, *state_dims)``.
            u: The computed sequence of controls. Expected shape ``(N, *control_dims)``.
            value: The cost value of the computed trajectory.
        """
        ...
