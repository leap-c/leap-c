import numpy as np

from leap_c.ocp.acados.parameters import AcadosParamManager, Parameter


def test_param_manger_intializes(
    acados_param_manager: AcadosParamManager,
) -> None:
    """
    Test the initialization of the AcadosParamManger.

    Args:
        acados_param_manager (AcadosParamManger): The AcadosParamManger instance to test.

    Raises:
        AssertionError: If the AcadosParamManger is None
    """
    assert acados_param_manager is not None, (
        "AcadosParamManger should not be None after initialization."
    )

    batch_size = 3
    N_horizon = 5
    m = 8
    n = 4
    overwrite = {}

    # Scalar parameter should have shape (batch_size, N_horizon, 1)
    overwrite["p"] = np.random.rand(batch_size, N_horizon, 1)

    # Vector parameter should have shape (batch_size, N_horizon, m)

    # Matrix parameter should have shape (batch_size, N_horizon, m, n)

    overwrite["cx"] = (np.linspace(0, 1, batch_size),)
    acados_param_manager.combine_parameter_values(batch_size=batch_size, **overwrite)
    # acados_param_manager.combine_parameters()


def test_param_manager_add_and_get(
    acados_param_manager: AcadosParamManager,
    nominal_varying_params: tuple[Parameter, ...],
    rng: np.random.Generator,
) -> None:
    """
    Test the addition of parameters to the AcadosParamManager and verify correct
    retrieval and mapping of dense parameter values.

    Args:
        acados_param_manager (AcadosParamManager): The parameter manager instance
         to test.
        nominal_varying_params (tuple[Parameter, ...]): Tuple of parameters to add and
         test.
        rng (np.random.Generator): Random number generator for reproducible noise.

    Raises:
        AssertionError: If the mapped and retrieved dense values do not match within
        the specified tolerance.
    """
    [acados_param_manager.add(param) for param in nominal_varying_params]
    acados_param_manager.initialize_p_global_values()
    acados_param_manager.get_default_parameter_values()

    for field in ["p", "p_global"]:
        values = acados_param_manager.get_dense(field_=field)
        values += rng.normal(loc=values, scale=0.1, size=values.shape)
        acados_param_manager.map_dense_to_structured(field_=field, values_=values)
        assert np.allclose(
            acados_param_manager.get_dense(field_=field), values, atol=1e-12
        ), f"{field} values do not match expected values."
