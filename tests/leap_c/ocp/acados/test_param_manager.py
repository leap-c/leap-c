import numpy as np

from acados_template import AcadosOcp
from leap_c.ocp.acados.parameters import AcadosParamManager, Parameter


def test_param_manger_intializes(
    acados_param_manager: AcadosParamManager,
) -> None:
    """
    Test the initialization of the AcadosParamManger.

    Args:
        acados_param_manager (AcadosParamManger): The AcadosParamManger instance.

    """
    assert isinstance(acados_param_manager, AcadosParamManager), (
        "AcadosParamManger should be an instance of AcadosParamManager."
    )


def test_param_manager_combine_parameter_values(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
    nominal_varying_params_for_param_manager_tests: tuple[Parameter, ...],
    rng: np.random.Generator,
) -> None:
    """
    Test the addition of parameters to the AcadosParamManager and verify correct
    retrieval and mapping of dense parameter values.

    Args:
        acados_test_ocp_with_stagewise_varying_params (AcadosOcp): AcadosOcp instance with
         stagewise varying parameters.
        nominal_varying_params_for_param_manager_tests (tuple[Parameter, ...]): Tuple of
         test parameters to overwrite.
        rng (np.random.Generator): Random number generator for reproducible noise.

    Raises:
        AssertionError: If the mapped and retrieved dense values do not match within
        the specified tolerance.
    """
    acados_param_manager = AcadosParamManager(
        params=nominal_varying_params_for_param_manager_tests,
        ocp=acados_test_ocp_with_stagewise_varying_params,
    )

    keys = [
        key
        for key in list(acados_param_manager.parameter_values.keys())
        if not key.startswith("indicator")
    ]

    # Get a random batch_size
    batch_size = rng.integers(low=5, high=10)

    N_horizon = acados_test_ocp_with_stagewise_varying_params.solver_options.N_horizon
    # Pick random keys

    # Build random overwrites
    overwrite = {}
    for key in keys:
        overwrite[key] = rng.random(
            size=(
                batch_size,
                N_horizon,
                acados_param_manager.parameter_values[key].shape[0],
            )
        )

    acados_param_manager.combine_parameter_values(**overwrite)
