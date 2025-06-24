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

    for field in ["p_global", "p"]:
        values = acados_param_manager.get_dense(field_=field)
        values += rng.normal(loc=values, scale=0.1, size=values.shape)
        acados_param_manager.map_dense_to_structured(field_=field, values_=values)
        assert np.allclose(
            acados_param_manager.get_dense(field_=field), values, atol=1e-12
        ), f"{field} values do not match expected values."
