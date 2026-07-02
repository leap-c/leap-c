import re

import casadi as ca
import numpy as np
import pytest
import torch
from acados_template import AcadosOcp, AcadosOcpSolver

from leap_c.parameters.base import AcadosParameterManager
from leap_c.parameters.data import _AcadosParameter
from leap_c.torch import AcadosDiffMpcLayerTorch
from leap_c.utils.parameters import _define_starts_and_ends, n_segments


def test_acados_param_manager_basic_initialization():
    """Test basic initialization of AcadosParamManager."""
    manager = AcadosParameterManager(N_horizon=10)
    manager.register_parameter(name="scalar", default=np.array([1.0]), differentiable=False)
    manager.register_parameter(name="vector", default=np.array([2.0, 3.0]), differentiable=True)

    assert len(manager.parameters) == 2
    assert "scalar" in manager.parameters
    assert "vector" in manager.parameters
    assert manager.N_horizon == 10


def test_parameter_interface_differentiable_no_vary_stages():
    """Test differentiable parameters without vary_stages."""
    manager = AcadosParameterManager(N_horizon=5)
    manager.register_parameter(
        name="scalar_differentiable", default=np.array([1.0]), differentiable=True
    )
    manager.register_parameter(
        name="vector_differentiable", default=np.array([2.0, 3.0]), differentiable=True
    )
    manager.register_parameter(
        name="matrix_differentiable",
        default=np.array([[4.0, 5.0], [6.0, 7.0]]),
        differentiable=True,
    )

    # All should appear in differentiable_parameters with original names
    assert len(manager._differentiable_parameter_store.symbols) == 3
    assert "scalar_differentiable" in manager._differentiable_parameter_store.symbols
    assert "vector_differentiable" in manager._differentiable_parameter_store.symbols
    assert "matrix_differentiable" in manager._differentiable_parameter_store.symbols

    # Check default values are set correctly (CasADi returns column vectors)
    np.testing.assert_array_equal(
        manager._differentiable_parameter_store.defaults["scalar_differentiable"], np.array([1.0])
    )
    np.testing.assert_array_equal(
        manager._differentiable_parameter_store.defaults["vector_differentiable"],
        np.array([2.0, 3.0]),
    )
    np.testing.assert_array_equal(
        manager._differentiable_parameter_store.defaults["matrix_differentiable"],
        np.array([[4.0, 5.0], [6.0, 7.0]]),  # Preserves matrix shape
    )


def test_parameter_interface_differentiable_with_vary_stages():
    """Test differentiable parameters with vary_stages."""
    N_horizon = 10
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="price",
        default=np.array([10.0]),
        differentiable=True,
        splits=[3, 7, N_horizon],
    )
    manager.register_parameter(
        name="demand",
        default=np.array([5.0, 6.0]),
        differentiable=True,
        splits=[2, 5, 8, N_horizon],
    )

    # Should create staged parameters with {name}_{start}_{end} template
    differentiable_keys = list(manager._differentiable_parameter_store.symbols.keys())

    # price changes at [3, 7], so we expect: price_0_2, price_3_6, price_7_10
    price_keys = [k for k in differentiable_keys if k.startswith("price_")]
    assert len(price_keys) == 3
    assert "price_0_3" in price_keys
    assert "price_4_7" in price_keys
    assert "price_8_10" in price_keys

    # demand changes at [2, 5, 8], so we expect: demand_0_1, demand_2_4, demand_5_7, demand_8_10
    demand_keys = [k for k in differentiable_keys if k.startswith("demand_")]
    assert len(demand_keys) == 4
    assert "demand_0_2" in demand_keys
    assert "demand_3_5" in demand_keys
    assert "demand_6_8" in demand_keys
    assert "demand_9_10" in demand_keys

    # Check that values are set correctly for each stage (CasADi format)
    for key in price_keys:
        np.testing.assert_array_equal(
            manager._differentiable_parameter_store.defaults[key], np.array([10.0])
        )

    for key in demand_keys:
        np.testing.assert_array_equal(
            manager._differentiable_parameter_store.defaults[key], np.array([5.0, 6.0])
        )


def test_parameter_interface_non_differentiable_no_vary_stages():
    """Test non-differentiable parameters without vary_stages."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="scalar_non_differentiable", default=np.array([1.0]), differentiable=False
    )
    manager.register_parameter(
        name="vector_non_differentiable", default=np.array([2.0, 3.0]), differentiable=False
    )
    manager.register_parameter(
        name="matrix_non_differentiable",
        default=np.array([[4.0, 5.0], [6.0, 7.0]]),
        differentiable=False,
    )

    # All should appear in non_differentiable_parameters with original names
    assert len(manager._non_differentiable_parameter_store.symbols) == 3
    assert "scalar_non_differentiable" in manager._non_differentiable_parameter_store.symbols
    assert "vector_non_differentiable" in manager._non_differentiable_parameter_store.symbols
    assert "matrix_non_differentiable" in manager._non_differentiable_parameter_store.symbols

    assert "scalar_non_differentiable" in manager._non_differentiable_parameter_store.defaults
    assert "vector_non_differentiable" in manager._non_differentiable_parameter_store.defaults
    assert "matrix_non_differentiable" in manager._non_differentiable_parameter_store.defaults

    for stage in range(N_horizon + 1):
        np.testing.assert_array_equal(
            manager._non_differentiable_parameter_store.defaults["scalar_non_differentiable"],
            np.array([1.0]),
        )
        np.testing.assert_array_equal(
            manager._non_differentiable_parameter_store.defaults["vector_non_differentiable"],
            np.array([2.0, 3.0]),
        )
        np.testing.assert_array_equal(
            manager._non_differentiable_parameter_store.defaults["matrix_non_differentiable"],
            np.array([[4.0, 5.0], [6.0, 7.0]]),
        )


def test_vary_stages_last_element_not_valid():
    """Test that ValueError is raised when vary_stages last element is invalid."""
    N_horizon = 10
    manager = AcadosParameterManager(N_horizon=N_horizon)

    with pytest.raises(
        ValueError,
        match=r"Parameter 'exceed_horizon' has splits \[5\] "
        r"but the last element must be either 9 or 10.",
    ):
        manager.register_parameter(
            name="exceed_horizon",
            default=np.array([1.0]),
            differentiable=True,
            splits=[5],
        )


def test_integer_splits_exceeds_horizon_on_init():
    """Test that integer splits cannot exceed N_horizon + 1 on init."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)

    with pytest.raises(
        ValueError,
        match=rf"Parameter 'too_many_splits' has {N_horizon + 2} splits, which exceeds the "
        rf"number of stages {N_horizon + 1}\.",
    ):
        manager.register_parameter(
            name="too_many_splits",
            default=np.array([1.0]),
            differentiable=True,
            splits=N_horizon + 2,
        )


def test_integer_splits_exceeds_horizon_on_add():
    """Test that integer splits cannot exceed N_horizon + 1 on add."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)

    with pytest.raises(
        ValueError,
        match=rf"Parameter 'too_many_splits' has {N_horizon + 2} splits, which exceeds the "
        rf"number of stages {N_horizon + 1}\.",
    ):
        manager.register_parameter(
            name="too_many_splits",
            default=np.array([1.0]),
            differentiable=True,
            splits=N_horizon + 2,
        )


def test_indicator_creation():
    """Test that indicator is created when vary_stages are used."""
    N_horizon = 5
    manager_no_vary = AcadosParameterManager(N_horizon=N_horizon)
    manager_no_vary.register_parameter(name="no_vary", default=np.array([1.0]), differentiable=True)

    manager_with_vary = AcadosParameterManager(N_horizon=N_horizon)
    manager_with_vary.register_parameter(
        name="with_vary",
        default=np.array([1.0]),
        differentiable=True,
        splits=[3, N_horizon],
    )

    # No vary_stages should not have indicator
    assert "indicator" not in manager_no_vary._non_differentiable_parameter_store.symbols

    # With vary_stages should have indicator
    assert "indicator" in manager_with_vary._non_differentiable_parameter_store.symbols


def test_mixed_parameter_types_and_interfaces():
    """Test complex scenario with mixed parameter types and interfaces."""
    N_horizon = 8
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(name="learn_scalar", default=np.array([6.0]), differentiable=True)
    manager.register_parameter(
        name="learn_vector", default=np.array([7.0, 8.0]), differentiable=True
    )
    manager.register_parameter(
        name="learn_staged",
        default=np.array([9.0]),
        differentiable=True,
        splits=[2, 6, N_horizon],
    )
    manager.register_parameter(
        name="non_learn_scalar", default=np.array([10.0]), differentiable=False
    )
    manager.register_parameter(
        name="non_learn_vector", default=np.array([11.0, 12.0]), differentiable=False
    )

    # Check differentiable parameters
    differentiable_keys = list(manager._differentiable_parameter_store.symbols.keys())
    expected_differentiable = [
        "learn_scalar",
        "learn_vector",
        "learn_staged_0_2",
        "learn_staged_3_6",
        "learn_staged_7_8",
    ]
    assert len(differentiable_keys) == len(expected_differentiable)
    for key in expected_differentiable:
        assert key in differentiable_keys

    # Check non-differentiable parameters (includes indicator)
    non_differentiable_keys = list(manager._non_differentiable_parameter_store.symbols.keys())
    expected_non_differentiable = [
        "non_learn_scalar",
        "non_learn_vector",
        "indicator",
    ]
    assert len(non_differentiable_keys) == len(expected_non_differentiable)
    for key in expected_non_differentiable:
        assert key in non_differentiable_keys


def test_n_segments_returns_expected_counts():
    assert n_segments("global", 5) == 1
    assert n_segments("stagewise", 5) == 6
    assert n_segments(3, 5) == 3
    assert n_segments([2, 5], 5) == 2
    assert n_segments([0, 4, 9], 9) == 3


def test_variable_splits_parameter_layout():
    """Registered stage-varying parameters create one differentiable block per stage segment.

    The flat differentiable vector scales with the number of stage variations and the
    dimension of each parameter.
    """
    N_horizon = 10
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="scalar",
        default=np.array([5.0]),
        differentiable=True,
        splits=[3, 7, N_horizon],
    )
    manager.register_parameter(
        name="vector",
        default=np.array([10.0, 15.0]),
        differentiable=True,
        splits=[2, 5, 8, N_horizon],
    )
    manager.register_parameter(
        name="scalar_unbounded",
        default=np.array([2.5]),
        differentiable=True,
        splits=[1, 3, 6, 8, N_horizon],
    )
    manager.register_parameter(
        name="matrix",
        default=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        differentiable=True,
        splits=[4, N_horizon],
    )
    manager.register_parameter(
        name="regular_param",
        default=np.array([1.0]),
        differentiable=True,
    )
    # The flat differentiable vector scales with stage variations and dimensions:
    #   scalar 3 + vector 8 + scalar_unbounded 5 + matrix 18 + regular_param 1 = 35
    assert manager.differentiable_default_flat.size == 35

    # Verify differentiable parameter keys match expected staged parameter names
    differentiable_keys = list(manager._differentiable_parameter_store.symbols.keys())

    # Check scalar variations (but not scalar_unbounded)
    scalar_keys = [
        k
        for k in differentiable_keys
        if k.startswith("scalar_") and not k.startswith("scalar_unbounded_")
    ]
    assert len(scalar_keys) == 3
    assert "scalar_0_3" in scalar_keys
    assert "scalar_4_7" in scalar_keys
    assert "scalar_8_10" in scalar_keys

    # Check vector variations
    vector_keys = [k for k in differentiable_keys if k.startswith("vector_")]
    assert len(vector_keys) == 4
    assert "vector_0_2" in vector_keys
    assert "vector_3_5" in vector_keys
    assert "vector_6_8" in vector_keys
    assert "vector_9_10" in vector_keys

    # Check scalar_unbounded variations
    scalar_unbounded_keys = [k for k in differentiable_keys if k.startswith("scalar_unbounded_")]
    assert len(scalar_unbounded_keys) == 5
    assert "scalar_unbounded_0_1" in scalar_unbounded_keys
    assert "scalar_unbounded_2_3" in scalar_unbounded_keys
    assert "scalar_unbounded_4_6" in scalar_unbounded_keys
    assert "scalar_unbounded_7_8" in scalar_unbounded_keys
    assert "scalar_unbounded_9_10" in scalar_unbounded_keys

    # Check matrix variations
    matrix_keys = [k for k in differentiable_keys if k.startswith("matrix_")]
    assert len(matrix_keys) == 2
    assert "matrix_0_4" in matrix_keys
    assert "matrix_5_10" in matrix_keys

    # Check regular parameter
    assert "regular_param" in differentiable_keys

    # Total differentiable parameters: 3 + 4 + 5 + 2 + 1 = 15 distinct parameter names
    assert len(differentiable_keys) == 15


def test_get_method_differentiable_parameters():
    """Test get method for differentiable parameters."""
    manager = AcadosParameterManager(N_horizon=5)
    manager.register_parameter(
        name="differentiable_param", default=np.array([1.0]), differentiable=True
    )

    # Should return the symbolic variable
    result = manager.get("differentiable_param")

    # Check that result has type ca.SX and shape (1,1) and that its name is "differentiable_param"
    assert isinstance(result, ca.SX)
    assert result.shape == (1, 1)
    assert result.str() == "differentiable_param"


def test_get_method_non_differentiable_parameters():
    """Test get method for non-differentiable parameters."""
    manager = AcadosParameterManager(N_horizon=5)
    manager.register_parameter(
        name="non_differentiable_param", default=np.array([1.0]), differentiable=False
    )

    # Should return the symbolic variable
    result = manager.get("non_differentiable_param")

    # Check that result has type ca.SX and shape (1,1) and that its name is "differentiable_param"
    assert isinstance(result, ca.SX)
    assert result.shape == (1, 1)
    assert result.str() == "non_differentiable_param"


def test_get_method_vary_stages():
    """Test get method for parameters with vary_stages."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="staged_param",
        default=np.array([1.0]),
        differentiable=True,
        splits=[3, N_horizon],
    )

    # Should return a combination of staged parameters
    result = manager.get("staged_param")

    # Check that result has type ca.SX and shape (1,1)
    assert isinstance(result, ca.SX)
    assert result.shape == (1, 1)

    assert result.str() == (
        "(((((indicator_0+indicator_1)+indicator_2)+indicator_3)*staged_param_0_3)"
        "+((indicator_4+indicator_5)*staged_param_4_5))"
    )


def test_get_method_unknown_field():
    """Test get method with unknown field name."""
    manager = AcadosParameterManager(N_horizon=5)
    manager.register_parameter(name="existing_param", default=np.array([1.0]), differentiable=False)

    with pytest.raises(ValueError, match="Unknown name: nonexistent"):
        manager.get("nonexistent")


def test_empty_parameter_list():
    """Test AcadosParamManager with empty parameter list."""
    manager = AcadosParameterManager(N_horizon=5)

    assert len(manager.parameters) == 0
    assert len(manager._differentiable_parameter_store.symbols) == 0
    assert len(manager._non_differentiable_parameter_store.symbols) == 0


def test_parameter_name_with_underscores():
    """Test parameters with underscores in their names.

    Test is due to a potential conflict with template for stages: {name}_{start}_{end}).
    """
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="param_with_underscores",
        default=np.array([1.0]),
        differentiable=True,
        splits=[3, N_horizon],
    )

    # Should properly handle names with underscores
    differentiable_keys = list(manager._differentiable_parameter_store.symbols.keys())
    staged_keys = [k for k in differentiable_keys if k.startswith("param_with_underscores_")]

    assert len(staged_keys) == 2
    assert "param_with_underscores_0_3" in staged_keys
    assert "param_with_underscores_4_5" in staged_keys

    # Values should be set correctly (CasADi format)
    for key in staged_keys:
        np.testing.assert_array_equal(
            manager._differentiable_parameter_store.defaults[key], np.array([1.0])
        )


def test_large_dimension_parameters():
    """Test that CasADi limitation with >2D arrays is handled gracefully."""
    # CasADi only supports up to 2D arrays, test that 2D arrays are accepted and work as expected.
    manager = AcadosParameterManager(N_horizon=5)
    manager.register_parameter(
        name="matrix_param",
        default=np.array([[1.0, 2.0], [3.0, 4.0]]),
        differentiable=True,
    )

    # Should handle 2D arrays correctly (flattened in CasADi)
    assert "matrix_param" in manager._differentiable_parameter_store.symbols

    # CasADi preserves matrix shapes
    expected_value = np.array([[1.0, 2.0], [3.0, 4.0]])

    np.testing.assert_array_equal(
        manager._differentiable_parameter_store.defaults["matrix_param"], expected_value
    )

    # Test that 3D arrays raise an error
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Parameter 'tensor_param' has 3 dimensions, "
            "but CasADi only supports arrays up to 2 dimensions. "
            "Parameter shape: (2, 2, 2)"
        ),
    ):
        _AcadosParameter(
            name="tensor_param",
            default=np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            interface="differentiable",
        )


def test_combine_parameter_values():
    """Test combining non-differentiable parameter values across multiple batches and time stages.

    Verifies that AcadosParameterManager.combine_non_differentiable_parameters()
    correctly combines parameter values into a (batch_size, N_horizon+1, param_dim) array.
    """
    manager = AcadosParameterManager(N_horizon=5)
    manager.register_parameter(name="test_param", default=np.array([1.0]), differentiable=False)

    expected = np.ones((2, 6, 1))
    result = manager.combine_non_differentiable_parameters(batch_size=2)
    np.testing.assert_array_equal(result, expected)


def test_combine_parameter_values_complex():
    """Test combine_parameter_values with mixed parameter types, interfaces, and vary_stages."""
    N_horizon = 8
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="scalar_differentiable", default=np.array([3.0]), differentiable=True
    )
    manager.register_parameter(
        name="scalar_non_differentiable", default=np.array([4.0]), differentiable=False
    )
    manager.register_parameter(
        name="scalar_staged",
        default=np.array([5.0]),
        differentiable=True,
        splits=[2, 6, N_horizon],
    )
    manager.register_parameter(
        name="vector_differentiable", default=np.array([6.0, 7.0]), differentiable=True
    )
    manager.register_parameter(
        name="vector_non_differentiable", default=np.array([8.0, 9.0]), differentiable=False
    )
    manager.register_parameter(
        name="vector_staged",
        default=np.array([10.0, 11.0]),
        differentiable=True,
        splits=[3, N_horizon],
    )
    manager.register_parameter(
        name="matrix_differentiable",
        default=np.array([[12.0, 13.0], [14.0, 15.0]]),
        differentiable=True,
    )
    manager.register_parameter(
        name="matrix_non_differentiable",
        default=np.array([[16.0, 17.0], [18.0, 19.0]]),
        differentiable=False,
    )
    manager.register_parameter(
        name="matrix_staged",
        default=np.array([[20.0, 21.0], [22.0, 23.0]]),
        differentiable=True,
        splits=[1, 4, 7],
    )

    # Test with batch_size=3
    batch_size = 3
    result = manager.combine_non_differentiable_parameters(batch_size=batch_size)

    # Verify result shape: (batch_size, N_horizon + 1, total_non_differentiable_params)
    # Non-differentiable params: scalar_non_differentiable(1) + vector_non_differentiable(2) +
    # matrix_non_differentiable(4) + indicator(9) = 16
    expected_shape = (batch_size, manager.N_horizon + 1, 16)
    assert result.shape == expected_shape

    # Verify that the values are correctly replicated across batches and stages
    # All non-differentiable parameters should have the same values across all batches
    for batch_idx in range(batch_size):
        for stage_idx in range(manager.N_horizon + 1):
            # Check scalar_non_differentiable (first element)
            s, _ = manager._non_differentiable_parameter_store.indices["scalar_non_differentiable"]
            assert result[batch_idx, stage_idx, s] == 4.0

            # Check vector_non_differentiable (next 2 elements)
            s, e = manager._non_differentiable_parameter_store.indices["vector_non_differentiable"]
            expected_vector_flat = np.array([8.0, 9.0])
            np.testing.assert_array_equal(result[batch_idx, stage_idx, s:e], expected_vector_flat)

            # Check matrix_non_differentiable (next 4 elements)
            # Matrix is flattened in column-major (Fortran) order by CasADi
            s, e = manager._non_differentiable_parameter_store.indices["matrix_non_differentiable"]
            expected_matrix_flat = np.array(
                [16.0, 18.0, 17.0, 19.0]
            )  # [[16,17],[18,19]] -> [16,18,17,19]
            np.testing.assert_array_equal(result[batch_idx, stage_idx, s:e], expected_matrix_flat)

            # Check indicator values (last 9 elements for N_horizon=8)
            # indicator[stage_idx] should be 1.0, others should be 0.0
            s, e = manager._non_differentiable_parameter_store.indices["indicator"]
            expected_indicator = np.zeros(9)
            expected_indicator[stage_idx] = 1.0
            np.testing.assert_array_equal(result[batch_idx, stage_idx, s:e], expected_indicator)

    rng = np.random.default_rng(42)

    # Build random overwrites
    vector_non_differentiable = rng.random(
        size=(
            batch_size,
            manager.N_horizon + 1,
            manager._non_differentiable_parameter_store.defaults["vector_non_differentiable"].shape[
                0
            ],
        )
    )

    matrix_non_differentiable = rng.random(
        size=(
            batch_size,
            manager.N_horizon + 1,
            manager._non_differentiable_parameter_store.defaults["matrix_non_differentiable"].shape[
                0
            ],
            manager._non_differentiable_parameter_store.defaults["matrix_non_differentiable"].shape[
                1
            ],
        )
    )

    result = manager.combine_non_differentiable_parameters(
        matrix_non_differentiable=matrix_non_differentiable,
        vector_non_differentiable=vector_non_differentiable,
    )

    # Verify the result shape remains the same
    assert result.shape == expected_shape

    # Verify that the overwritten parameters are correctly incorporated
    for batch_idx in range(batch_size):
        for stage_idx in range(manager.N_horizon + 1):
            # scalar_non_differentiable should still be the default value (not overwritten)
            s, _ = manager._non_differentiable_parameter_store.indices["scalar_non_differentiable"]
            assert result[batch_idx, stage_idx, s] == 4.0

            # vector_non_differentiable should use the random overwrite values
            s, e = manager._non_differentiable_parameter_store.indices["vector_non_differentiable"]
            np.testing.assert_array_equal(
                result[batch_idx, stage_idx, s:e],
                vector_non_differentiable[batch_idx, stage_idx, :],
            )

            # matrix_non_differentiable should use the random overwrite values (flattened)
            # Note: overwrite values use C-order flattening, unlike default values which use F-order
            s, e = manager._non_differentiable_parameter_store.indices["matrix_non_differentiable"]
            expected_matrix_flat = matrix_non_differentiable[batch_idx, stage_idx, :, :].flatten(
                order="C"
            )
            np.testing.assert_array_equal(result[batch_idx, stage_idx, s:e], expected_matrix_flat)

            # indicator values should remain unchanged
            s, e = manager._non_differentiable_parameter_store.indices["indicator"]
            expected_indicator = np.zeros(9)
            expected_indicator[stage_idx] = 1.0
            np.testing.assert_array_equal(result[batch_idx, stage_idx, s:e], expected_indicator)


def test_param_manager_combine_parameter_values(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
    nominal_stagewise_params: tuple[_AcadosParameter, ...],
    rng: np.random.Generator,
) -> None:
    """Test the addition of parameters to the AcadosParamManager.

    Also test and verify correct retrieval and mapping of dense parameter values.

    Args:
        acados_test_ocp_with_stagewise_varying_params: AcadosOcp instance
        with stagewise varying parameters.
        nominal_stagewise_params: Tuple of
         test parameters to overwrite.
        rng: Random number generator for reproducible noise.

    Raises:
        AssertionError: If the mapped and retrieved dense values do not match within
        the specified tolerance.
    """
    N_horizon = acados_test_ocp_with_stagewise_varying_params.solver_options.N_horizon

    acados_param_manager = AcadosParameterManager(N_horizon=N_horizon)
    for param in nominal_stagewise_params:
        acados_param_manager.register_parameter(
            name=param.name,
            default=param.default,
            differentiable=(param.interface == "differentiable"),
            splits=param.splits,
        )

    keys = [
        key
        for key in acados_param_manager._non_differentiable_parameter_store.defaults
        if not key.startswith("indicator")
    ]

    # Get a random batch_size
    batch_size = rng.integers(low=5, high=10)

    # Build random overwrites
    overwrite = {}
    for key in keys:
        overwrite[key] = rng.random(
            size=(
                batch_size,
                N_horizon + 1,
                acados_param_manager._non_differentiable_parameter_store.defaults[key].shape[0],
            )
        )

    res = acados_param_manager.combine_non_differentiable_parameters(**overwrite)

    assert res.shape == (
        batch_size,
        N_horizon + 1,
        acados_param_manager._non_differentiable_parameter_store.size,
    ), "The shape of the combined parameter values does not match the expected shape."

    # Verify that the overwritten parameter values are correctly incorporated
    param_start_idx = 0
    for key in keys:
        param_dim = acados_param_manager._non_differentiable_parameter_store.defaults[key].shape[0]
        param_end_idx = param_start_idx + param_dim

        # Check that the overwritten values match exactly
        for batch_idx in range(batch_size):
            for stage_idx in range(N_horizon + 1):
                np.testing.assert_array_equal(
                    res[batch_idx, stage_idx, param_start_idx:param_end_idx],
                    overwrite[key][batch_idx, stage_idx, :],
                    err_msg=f"Mismatch in parameter '{key}' "
                    "at batch {batch_idx}, stage {stage_idx}",
                )

        param_start_idx = param_end_idx

    # Verify that indicator values are correctly set (they should be at the end)
    indicator_start_idx = param_start_idx
    for batch_idx in range(batch_size):
        for stage_idx in range(N_horizon + 1):
            expected_indicator = np.zeros(N_horizon + 1)
            expected_indicator[stage_idx] = 1.0
            np.testing.assert_array_equal(
                res[batch_idx, stage_idx, indicator_start_idx:],
                expected_indicator,
                err_msg=f"Mismatch in indicator values at batch {batch_idx}, stage {stage_idx}",
            )


def test_diff_mpc_with_stagewise_params_equivalent_to_diff_mpc(
    diff_mpc: AcadosDiffMpcLayerTorch,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpcLayerTorch,
) -> None:
    """Test diff_mpc with stagewise varying parameters is equivalent diff_mpc.

    This test verifies that statewise varying diff_mpc is equivalent to the
    diff_mpc with global parameters by comparing the forward pass results under
    the condition that the stagewise varying parameters are set to the nominal values.
    """
    mpc = {
        "stagewise": diff_mpc_with_stagewise_varying_params,
        "global": diff_mpc,
    }

    x0 = np.array([1.0, 1.0, 0.0, 0.0])

    sol_forward = {}
    sol_forward["global"] = mpc["global"].forward(
        x0=torch.tensor(x0, dtype=torch.float32).reshape(1, -1)
    )
    sol_forward["stagewise"] = mpc["stagewise"].forward(
        x0=torch.tensor(x0, dtype=torch.float32).reshape(1, -1),
    )

    out = ["ctx", "u0", "x", "u", "value"]
    for idx, label in enumerate(out[1:]):
        assert np.allclose(
            sol_forward["global"][idx + 1].detach().numpy(),
            sol_forward["stagewise"][idx + 1].detach().numpy(),
            atol=1e-3,
            rtol=1e-3,
        ), f"The {label} does not match between global and stagewise varying diff MPC."


def test_casadi_function_with_parameter_manager():
    """Test creating a CasADi function using symbolic variables from AcadosParameterManager."""
    for interface in ["differentiable", "non-differentiable"]:
        default_a = np.array([2.0])
        default_b = np.array([3.0, 4.0])
        differentiable = interface == "differentiable"
        manager = AcadosParameterManager(N_horizon=5)
        manager.register_parameter(name="param_a", default=default_a, differentiable=differentiable)
        manager.register_parameter(name="param_b", default=default_b, differentiable=differentiable)

        # Get symbolic expressions
        param_a_sym = manager.get("param_a")
        param_b_sym = manager.get("param_b")

        # Create a simple expression using the symbolic parameters
        expr = param_a_sym * ca.sum1(param_b_sym)

        # Define a CasADi function
        casadi_func = ca.Function(
            "test_func", [param_a_sym, param_b_sym], [expr], ["param_a", "param_b"], ["result"]
        )

        # Test the function with default values
        result = casadi_func(default_a, default_b)

        # Expected: 2.0 * (3.0 + 4.0) = 14.0
        expected_result = 2.0 * (3.0 + 4.0)
        np.testing.assert_allclose(float(result), expected_result, rtol=1e-6)


def test_combine_differentiable_parameters_torch_basic():
    """Test combine_differentiable_parameters_torch with basic parameters."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(name="scalar", default=np.array([1.0]), differentiable=True)
    manager.register_parameter(name="vector", default=np.array([2.0, 3.0]), differentiable=True)

    # Test default values without overwrites
    batch_size = 3
    result = manager.combine_differentiable_parameters_torch(batch_size=batch_size).detach().numpy()

    # Expected: tiled default values
    default_flat = np.concatenate(
        list(manager._differentiable_parameter_store.defaults.values())
    ).reshape(-1)
    expected = np.tile(default_flat, (batch_size, 1))

    np.testing.assert_array_equal(result, expected)
    assert result.shape == (batch_size, len(default_flat))


def test_combine_differentiable_parameters_torch_with_overwrites():
    """Test combine_differentiable_parameters_torch with overwrites for non-stagewise params."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(name="scalar", default=np.array([1.0]), differentiable=True)
    manager.register_parameter(name="vector", default=np.array([2.0, 3.0]), differentiable=True)

    batch_size = 3
    # Overwrite scalar with custom values
    scalar_values = np.array([[10.0], [20.0], [30.0]])

    result = (
        manager.combine_differentiable_parameters_torch(batch_size=batch_size, scalar=scalar_values)
        .detach()
        .numpy()
    )

    # Check that scalar was overwritten
    scalar_idx_start, scalar_idx_end = manager._differentiable_parameter_store.indices["scalar"]
    np.testing.assert_array_equal(result[:, scalar_idx_start:scalar_idx_end], scalar_values)

    # Check that vector kept default values
    vector_idx_start, vector_idx_end = manager._differentiable_parameter_store.indices["vector"]
    expected_vector = np.tile([[2.0], [3.0]], (1, batch_size)).T
    np.testing.assert_array_equal(result[:, vector_idx_start:vector_idx_end], expected_vector)


def test_combine_differentiable_parameters_torch_stagewise():
    """Test combine_differentiable_parameters_torch with stagewise parameters."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="temperature",
        default=np.array([20.0]),
        differentiable=True,
        splits=[2, N_horizon],
    )
    manager.register_parameter(
        name="price",
        default=np.array([10.0]),
        differentiable=True,
        splits=[N_horizon],
    )

    batch_size = 2
    # Provide per-segment forecasts: shape (batch_size, n_segments)
    temperature_forecast = np.array([[15.0, 16.0], [25.0, 26.0]])

    price_forecast = np.array([[5.0], [15.0]])

    result = (
        manager.combine_differentiable_parameters_torch(
            batch_size=batch_size, temperature=temperature_forecast, price=price_forecast
        )
        .detach()
        .numpy()
    )

    # Verify temperature stages
    temp_0_2_idx_start, temp_0_2_idx_end = manager._differentiable_parameter_store.indices[
        "temperature_0_2"
    ]
    temp_3_5_idx_start, temp_3_5_idx_end = manager._differentiable_parameter_store.indices[
        "temperature_3_5"
    ]

    # For batch 0: segment 0 -> temperature_0_2, segment 1 -> temperature_3_5
    np.testing.assert_array_equal(
        result[0, temp_0_2_idx_start:temp_0_2_idx_end], temperature_forecast[0, 0]
    )
    np.testing.assert_array_equal(
        result[0, temp_3_5_idx_start:temp_3_5_idx_end], temperature_forecast[0, 1]
    )

    # For batch 1
    np.testing.assert_array_equal(
        result[1, temp_0_2_idx_start:temp_0_2_idx_end], temperature_forecast[1, 0]
    )
    np.testing.assert_array_equal(
        result[1, temp_3_5_idx_start:temp_3_5_idx_end], temperature_forecast[1, 1]
    )

    # Verify price (single stage block 0-5)
    price_0_5_idx_start, price_0_5_idx_end = manager._differentiable_parameter_store.indices[
        "price_0_5"
    ]
    np.testing.assert_array_equal(
        result[0, price_0_5_idx_start:price_0_5_idx_end], price_forecast[0, 0]
    )
    np.testing.assert_array_equal(
        result[1, price_0_5_idx_start:price_0_5_idx_end], price_forecast[1, 0]
    )


def test_combine_differentiable_parameters_torch_errors():
    """Test error handling in combine_differentiable_parameters_torch."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(name="scalar", default=np.array([1.0]), differentiable=True)
    manager.register_parameter(
        name="temperature",
        default=np.array([20.0]),
        differentiable=True,
        splits=[2, N_horizon],
    )

    # Test error for unknown parameter
    with pytest.raises(ValueError, match="Parameter 'unknown' not found"):
        manager.combine_differentiable_parameters_torch(
            batch_size=2, unknown=np.array([[1.0], [2.0]])
        ).detach().numpy()

    # Test error for non-differentiable parameter
    manager2 = AcadosParameterManager(N_horizon=N_horizon)
    manager2.register_parameter(name="non_learn", default=np.array([1.0]), differentiable=False)

    with pytest.raises(ValueError, match="has interface 'non-differentiable'"):
        manager2.combine_differentiable_parameters_torch(
            batch_size=2, non_learn=np.array([[1.0], [2.0]])
        ).detach().numpy()

    # Test error for wrong batch size
    with pytest.raises(ValueError, match="batch_size=2 does not match.*batch_size=3"):
        manager.combine_differentiable_parameters_torch(
            batch_size=2, scalar=np.array([[1.0], [2.0], [3.0]])
        ).detach().numpy()

    # Test error for wrong shape in stagewise parameter
    with pytest.raises(ValueError, match="requires shape \\(batch_size, 2"):
        manager.combine_differentiable_parameters_torch(
            batch_size=2,
            temperature=np.array([[1.0], [2.0]]),  # Should be (2, 2)
        ).detach().numpy()


@pytest.mark.parametrize(
    "splits,expected_n_segments",
    [
        ("global", 1),
        ("stagewise", 6),
        ([2, 5], 2),
        (3, 3),
    ],
)
def test_acados_parameter_overwrite_shape_and_broadcasted_default(splits, expected_n_segments):
    """overwrite_shape and broadcasted_default agree across split types."""
    N_horizon = 5
    default = np.array([1.0])
    param = _AcadosParameter(name="p", default=default, interface="differentiable", splits=splits)

    shape = param.overwrite_shape(N_horizon)
    bd = param.broadcasted_default(N_horizon)

    if splits == "global":
        assert shape == default.shape
        assert bd.shape == default.shape
        np.testing.assert_array_equal(bd, default)
    else:
        assert shape == (expected_n_segments, *default.shape)
        assert bd.shape == shape
        np.testing.assert_array_equal(bd, np.tile(default, (expected_n_segments, 1)))


def test_acados_parameter_overwrite_shape_and_broadcasted_default_matrix():
    """overwrite_shape/broadcasted_default preserve trailing dims for 2-d defaults."""
    N_horizon = 5
    default = np.array([[1.0, 2.0], [3.0, 4.0]])
    for splits, expected_n_segments in [("stagewise", 6), ([2, 5], 2), (3, 3)]:
        param = _AcadosParameter(
            name="p", default=default, interface="differentiable", splits=splits
        )
        shape = param.overwrite_shape(N_horizon)
        assert shape == (expected_n_segments, 2, 2)
        bd = param.broadcasted_default(N_horizon)
        assert bd.shape == (expected_n_segments, 2, 2)
        np.testing.assert_array_equal(bd, np.tile(default, (expected_n_segments, 1, 1)))


def test_combine_differentiable_parameters_torch_list_splits():
    """Combine with list splits [2, 5] indexes by segment, not by stage start."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="p", default=np.array([1.0]), differentiable=True, splits=[2, N_horizon]
    )

    batch_size = 2
    values = np.array([[15.0, 25.0], [16.0, 26.0]])
    result = (
        manager.combine_differentiable_parameters_torch(batch_size=batch_size, p=values)
        .detach()
        .numpy()
    )

    s0, e0 = manager._differentiable_parameter_store.indices["p_0_2"]
    s1, e1 = manager._differentiable_parameter_store.indices["p_3_5"]
    np.testing.assert_array_equal(result[0, s0:e0], [15.0])
    np.testing.assert_array_equal(result[0, s1:e1], [25.0])
    np.testing.assert_array_equal(result[1, s0:e0], [16.0])
    np.testing.assert_array_equal(result[1, s1:e1], [26.0])


def test_combine_differentiable_parameters_torch_int_splits():
    """Combine with int splits indexes each of the n_segments segments."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(name="p", default=np.array([1.0]), differentiable=True, splits=3)

    starts, ends = _define_starts_and_ends(3, N_horizon)
    batch_size = 2
    values = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    result = (
        manager.combine_differentiable_parameters_torch(batch_size=batch_size, p=values)
        .detach()
        .numpy()
    )

    for seg_idx, (start, end) in enumerate(zip(starts, ends)):
        s, e = manager._differentiable_parameter_store.indices[f"p_{start}_{end}"]
        np.testing.assert_array_equal(result[0, s:e], values[0, seg_idx])
        np.testing.assert_array_equal(result[1, s:e], values[1, seg_idx])


def test_combine_differentiable_parameters_torch_literal_stagewise():
    """Combine with literal stagewise splits accepts (batch, N+1, ...)."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="p", default=np.array([1.0]), differentiable=True, splits="stagewise"
    )

    starts, ends = _define_starts_and_ends("stagewise", N_horizon)
    batch_size = 2
    values = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]])
    result = (
        manager.combine_differentiable_parameters_torch(batch_size=batch_size, p=values)
        .detach()
        .numpy()
    )

    for seg_idx, (start, end) in enumerate(zip(starts, ends)):
        s, e = manager._differentiable_parameter_store.indices[f"p_{start}_{end}"]
        np.testing.assert_array_equal(result[0, s:e], values[0, seg_idx])
        np.testing.assert_array_equal(result[1, s:e], values[1, seg_idx])


@pytest.mark.parametrize(
    "splits,expected_n_segments",
    [
        ("global", 1),
        ("stagewise", 6),
        ([2, 5], 2),
        (3, 3),
    ],
)
def test_combine_differentiable_parameters_torch_defaults(splits, expected_n_segments):
    """Combine with no overwrites returns broadcasted_default tiled across batch."""
    N_horizon = 5
    default = np.array([7.0])
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(name="p", default=default, differentiable=True, splits=splits)

    batch_size = 3
    result = manager.combine_differentiable_parameters_torch(batch_size=batch_size).numpy()
    bd = manager.parameters["p"].broadcasted_default(N_horizon)
    expected = np.tile(bd.reshape(-1), (batch_size, 1))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "splits,expected_n_segments",
    [
        ("global", 1),
        ("stagewise", 6),
        ([2, 5], 2),
        (3, 3),
    ],
)
def test_combine_non_differentiable_parameters_defaults(splits, expected_n_segments):
    """combine_non_differentiable_parameters with no overwrites tiles defaults across batch/stages.

    A differentiable parameter with the given ``splits`` is registered (which injects an
    indicator into the non-differentiable store when stage-varying) alongside a plain
    non-differentiable parameter. The non-differentiable defaults must be tiled across the
    batch and stage dimensions, and the indicator (if present) must be set to the identity.
    """
    N_horizon = 5
    nd_default = np.array([42.0])
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(name="nd", default=nd_default, differentiable=False)
    manager.register_parameter(
        name="d", default=np.array([1.0]), differentiable=True, splits=splits
    )

    batch_size = 3
    result = manager.combine_non_differentiable_parameters(batch_size=batch_size)

    # The non-differentiable param is tiled across batch and stages.
    s_nd, e_nd = manager._non_differentiable_parameter_store.indices["nd"]
    expected_nd = np.tile(nd_default, (batch_size, N_horizon + 1, 1))
    np.testing.assert_array_equal(result[:, :, s_nd:e_nd], expected_nd)

    if splits == "global":
        # No indicator injected for a global differentiable param.
        assert "indicator" not in manager._non_differentiable_parameter_store.symbols
    else:
        # Indicator is present and set to the identity matrix (one-hot per stage).
        s_ind, e_ind = manager._non_differentiable_parameter_store.indices["indicator"]
        for stage in range(N_horizon + 1):
            expected_indicator = np.zeros(N_horizon + 1)
            expected_indicator[stage] = 1.0
            np.testing.assert_array_equal(
                result[:, stage, s_ind:e_ind],
                np.tile(expected_indicator, (batch_size, 1)),
            )


def test_combine_differentiable_parameters_torch_uncovered_terminal_stage():
    """splits=[2, 4] with N_horizon=5 -> 2 segments (0:2), (3:4); stage 5 uncovered."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="p", default=np.array([1.0]), differentiable=True, splits=[2, 4]
    )

    assert manager.parameters["p"].overwrite_shape(N_horizon) == (2, 1)
    starts, ends = _define_starts_and_ends([2, 4], N_horizon)
    assert list(zip(starts, ends)) == [(0, 2), (3, 4)]

    batch_size = 2
    values = np.array([[15.0, 25.0], [16.0, 26.0]])
    result = (
        manager.combine_differentiable_parameters_torch(batch_size=batch_size, p=values)
        .detach()
        .numpy()
    )

    s0, e0 = manager._differentiable_parameter_store.indices["p_0_2"]
    s1, e1 = manager._differentiable_parameter_store.indices["p_3_4"]
    np.testing.assert_array_equal(result[0, s0:e0], [15.0])
    np.testing.assert_array_equal(result[0, s1:e1], [25.0])
    np.testing.assert_array_equal(result[1, s0:e0], [16.0])
    np.testing.assert_array_equal(result[1, s1:e1], [26.0])


def test_combine_differentiable_parameters_torch_gradient_flow():
    """Gradients flow from combine output back to per-segment overwrite tensors."""
    N_horizon = 5
    manager = AcadosParameterManager(N_horizon=N_horizon)
    manager.register_parameter(
        name="p", default=np.array([1.0]), differentiable=True, splits=[2, N_horizon]
    )

    values = torch.tensor([[15.0, 25.0], [16.0, 26.0]], dtype=torch.float64, requires_grad=True)
    result = manager.combine_differentiable_parameters_torch(
        batch_size=2, device=torch.device("cpu"), dtype=torch.float64, p=values
    )
    result.sum().backward()

    assert values.grad is not None
    assert values.grad.shape == values.shape
    np.testing.assert_allclose(values.grad.numpy(), np.ones((2, 2)))


def test_stagewise_solution_matches_global_solver_for_initial_reference_change(
    acados_test_ocp_no_p_global: AcadosOcp,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpcLayerTorch,
    diff_mpc: AcadosDiffMpcLayerTorch,
    rng: np.random.Generator,
) -> None:
    """Test stagewise solution matches global solver for initial reference change.

    Test that setting parameters stagewise has the expected effect by comparing it to
    an ocp_solver with global parameters and nonlinear_ls cost.
    """
    global_solver = AcadosOcpSolver(acados_test_ocp_no_p_global)

    ocp = diff_mpc_with_stagewise_varying_params.diff_mpc_fun.ocp
    pm = diff_mpc_with_stagewise_varying_params.parameter_manager

    xref_0 = rng.random(size=4)
    uref_0 = rng.random(size=2)
    yref_0 = np.concatenate((xref_0, uref_0))

    global_solver.cost_set(stage_=0, field_="yref", value_=yref_0)

    x0 = ocp.constraints.x0

    _ = global_solver.solve_for_x0(x0_bar=x0)

    u_global = np.vstack(
        [
            global_solver.get(stage_=stage, field_="u")
            for stage in range(ocp.solver_options.N_horizon)
        ]
    )

    x_global = np.vstack(
        [
            global_solver.get(stage_=stage, field_="x")
            for stage in range(ocp.solver_options.N_horizon + 1)
        ]
    )

    x0 = torch.tensor(x0, dtype=torch.float32).reshape(1, -1)

    # Build xref: tile default over all stages, override stage 0
    xref_values = np.tile(pm.parameters["xref"].default, (ocp.solver_options.N_horizon, 1))
    xref_values[0] = xref_0

    # Build uref: same approach
    uref_values = np.tile(pm.parameters["uref"].default, (ocp.solver_options.N_horizon, 1))
    uref_values[0] = uref_0

    params = {
        "xref": torch.tensor(xref_values, dtype=torch.float32).unsqueeze(0),
        "uref": torch.tensor(uref_values, dtype=torch.float32).unsqueeze(0),
    }
    sol_pert = diff_mpc_with_stagewise_varying_params.forward(x0=x0, params=params)

    u_stagewise = sol_pert[3].detach().numpy().reshape(-1, ocp.dims.nu)
    x_stagewise = sol_pert[2].detach().numpy().reshape(-1, ocp.dims.nx)

    # TODO: Use flattened_iterate.allclose() when available for batch iterates.
    # NOTE: Use flattened_iterate = global_solver.store_iterate_to_flat_obj()
    # NOTE: and sol_flattened_batch_iterate = sol_pert[0].iterate

    assert np.allclose(
        u_global,
        u_stagewise,
        atol=1e-3,
        rtol=1e-3,
    ), "The control trajectory does not match between global and stagewise diff MPC."

    assert np.allclose(
        x_global,
        x_stagewise,
        atol=1e-3,
        rtol=1e-3,
    ), "The state trajectory does not match between global and stagewise diff MPC."

    sol_nom = diff_mpc.forward(x0=x0)

    u_stagewise_nom = sol_nom[3].detach().numpy().reshape(-1, ocp.dims.nu)
    x_stagewise_nom = sol_nom[2].detach().numpy().reshape(-1, ocp.dims.nx)

    assert not np.allclose(
        u_stagewise_nom,
        u_stagewise,
        atol=1e-3,
        rtol=1e-3,
    ), (
        "The control trajectory matches between nominal and stagewise diff MPC despite different"
        " initial reference."
    )

    assert not np.allclose(
        x_stagewise_nom,
        x_stagewise,
        atol=1e-3,
        rtol=1e-3,
    ), (
        "The state trajectory matches between nominal and stagewise diff MPC despite different"
        " initial reference."
    )


def test_add_parameter_interface_differentiable_no_vary_stages():
    """Test adding differentiable parameters without vary_stages."""
    manager = AcadosParameterManager(N_horizon=5)
    manager.register_parameter(name="differentiable", default=np.array([1.0]), differentiable=True)

    # Differentiable should appear in differentiable_parameter_store with original name
    assert len(manager._differentiable_parameter_store.symbols) == 1
    assert "differentiable" in manager._differentiable_parameter_store.symbols

    # No indicator should be created
    assert len(manager._non_differentiable_parameter_store.symbols) == 0


def test_add_parameter_interface_differentiable_with_vary_stages():
    """Test adding differentiable parameters with vary_stages."""
    N_horizon = 10
    manager = AcadosParameterManager(N_horizon=N_horizon)

    # No indicator should be created before adding the parameter
    assert len(manager._non_differentiable_parameter_store.symbols) == 0

    manager.register_parameter(
        name="price",
        default=np.array([10.0]),
        differentiable=True,
        splits=[3, 7, N_horizon],
    )

    # Should create staged parameters with {name}_{start}_{end} template
    differentiable_keys = list(manager._differentiable_parameter_store.symbols.keys())

    # price changes at [3, 7], so we expect: price_0_2, price_3_6, price_7_10
    price_keys = [k for k in differentiable_keys if k.startswith("price_")]
    assert len(price_keys) == 3
    assert "price_0_3" in price_keys
    assert "price_4_7" in price_keys
    assert "price_8_10" in price_keys

    # Indicator should be created in non-differentiable parameters
    assert "indicator" in manager._non_differentiable_parameter_store.symbols


def test_add_parameter_interface_non_differentiable():
    """Test adding non-differentiable parameters."""
    manager = AcadosParameterManager(N_horizon=5)
    manager.register_parameter(
        name="non_differentiable", default=np.array([1.0]), differentiable=False
    )

    # Non-differentiable should appear in non_differentiable_parameter_store with original name
    assert len(manager._non_differentiable_parameter_store.symbols) == 1
    assert "non_differentiable" in manager._non_differentiable_parameter_store.symbols


def test_define_starts_and_ends_stagewise():
    """Test starts/ends for stagewise splits."""
    starts, ends = _define_starts_and_ends(splits="stagewise", N_horizon=3)

    assert starts == [0, 1, 2, 3]
    assert ends == [0, 1, 2, 3]


def test_define_starts_and_ends_global():
    """Test starts/ends for global splits: one segment covering all stages."""
    starts, ends = _define_starts_and_ends(splits="global", N_horizon=5)

    assert starts == [0]
    assert ends == [5]


def test_define_starts_and_ends_list_splits():
    """Test starts/ends for explicit list splits."""
    starts, ends = _define_starts_and_ends(splits=[2, 5], N_horizon=5)

    assert starts == [0, 3]
    assert ends == [2, 5]


def test_define_starts_and_ends_int_splits_balanced():
    """Test starts/ends for integer splits with remainder."""
    starts, ends = _define_starts_and_ends(splits=3, N_horizon=4)

    assert starts == [0, 2, 4]
    assert ends == [1, 3, 4]


@pytest.mark.parametrize("invalid_splits", ["foo", 0, -1, 3.5, [1, "a"], None])
def test_define_starts_and_ends_invalid_raises(invalid_splits):
    """Invalid splits values raise ValueError immediately."""
    with pytest.raises(ValueError, match="Invalid splits value"):
        _define_starts_and_ends(splits=invalid_splits, N_horizon=5)


def test_define_starts_and_ends_docstring_example():
    """The docstring example: _define_starts_and_ends([2, 5], 5) -> ([0, 3], [2, 5])."""
    starts, ends = _define_starts_and_ends(splits=[2, 5], N_horizon=5)
    assert starts == [0, 3]
    assert ends == [2, 5]


def test_acados_parameter_splits_empty_list_raises():
    """Test empty splits list raises a ValueError."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Parameter 'empty_splits' has empty splits list. Hint: if you meant to define a"
            " global parameter, please set splits='global' instead."
        ),
    ):
        _AcadosParameter(
            name="empty_splits",
            default=np.array([1.0]),
            interface="differentiable",
            splits=[],
        )


def test_acados_parameter_splits_unsorted_list_raises():
    """Test unsorted splits list raises a ValueError."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Parameter 'unsorted_splits' splits [3, 1] are not sorted in ascending order."
        ),
    ):
        _AcadosParameter(
            name="unsorted_splits",
            default=np.array([1.0]),
            interface="differentiable",
            splits=[3, 1],
        )


def test_acados_parameter_splits_invalid_int_raises():
    """Test invalid integer splits raises a ValueError."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Parameter 'invalid_int_splits' has invalid splits value 1, number of splits must be "
            "`>1`. If you meant to define a global parameter, please set splits='global'"
        ),
    ):
        _AcadosParameter(
            name="invalid_int_splits",
            default=np.array([1.0]),
            interface="differentiable",
            splits=1,
        )


def test_repr_empty_manager():
    """Repr of an empty manager shows zero flat sizes and both section headers, no rows."""
    m = AcadosParameterManager(N_horizon=5)
    r = repr(m)
    assert "differentiable_flat=0" in r
    assert "non_differentiable_flat=0" in r
    assert "  differentiable:" in r
    assert "  non-differentiable:" in r
    # No data rows: only the header line plus the two section header lines.
    assert len(r.splitlines()) == 3
    assert "indicator" not in r


def test_repr_global_only():
    """Repr of a global differentiable + non-differentiable param shows expected shapes."""
    m = AcadosParameterManager(N_horizon=5)
    m.register_parameter("Q", default=np.array([1.0, 2.0]), differentiable=True)
    m.register_parameter("mass", default=np.array([3.0]), differentiable=False)
    r = repr(m)
    lines = r.splitlines()
    # Header + section + col-header + row for diff; section + col-header + row for non-diff.
    assert lines[0].startswith("AcadosParameterManager(N_horizon=5, casadi_type='SX',")
    assert "  differentiable:" in r
    assert "  non-differentiable:" in r
    # Global diff row: splits=global, shape is default.shape (2,)
    assert "Q" in r and "global" in r and "(2,)" in r
    # Non-diff row: shape is (N+1, *default.shape) = (6, 1)
    assert "mass" in r and "(6, 1)" in r


def test_repr_stagewise():
    """Repr of a stagewise differentiable param shows broadcasted default and (N+1, ...) shape."""
    N = 5
    m = AcadosParameterManager(N_horizon=N)
    m.register_parameter("price", default=np.array([1.0]), differentiable=True, splits="stagewise")
    r = repr(m)
    # Stage-varying shape is (N+1, *default.shape) = (6, 1)
    assert "stagewise" in r
    assert "(6, 1)" in r
    # Broadcasted default has N+1 tiled entries on a single line.
    assert "[[1.], [1.], [1.], [1.], [1.], [1.]]" in r


def test_repr_list_splits():
    """Repr of a [2, 5] splits param shows 2 segments and 2 tiled defaults."""
    N = 5
    m = AcadosParameterManager(N_horizon=N)
    m.register_parameter("price", default=np.array([1.0]), differentiable=True, splits=[2, 5])
    r = repr(m)
    assert "[2, 5]" in r
    # 2 segments -> shape (2, 1), default tiled to 2 entries.
    assert "(2, 1)" in r
    assert "[[1.], [1.]]" in r


def test_repr_matrix_default_single_line():
    """Repr of a 2-d matrix default renders single-line (no embedded newlines)."""
    m = AcadosParameterManager(N_horizon=5)
    m.register_parameter(
        "mat",
        default=np.array([[1.0, 2.0], [3.0, 4.0]]),
        differentiable=True,
    )
    r = repr(m)
    # The default appears on a single line within its row (no line breaks inside).
    assert "[[1., 2.], [3., 4.]]" in r


def test_repr_large_default_truncated():
    """Repr of a large default (20 elements) is truncated with ellipsis on a single line."""
    m = AcadosParameterManager(N_horizon=5)
    m.register_parameter("big", default=np.zeros(20), differentiable=True)
    r = repr(m)
    assert "..." in r
    # Single-line rendering: the truncated array does not introduce extra newlines.
    for line in r.splitlines():
        if "big" in line:
            assert "..." in line


def test_repr_indicator_hidden_but_counted():
    """The auto-injected indicator is hidden from rows but counted in non_differentiable_flat."""
    N = 5
    m = AcadosParameterManager(N_horizon=N)
    # A stage-varying differentiable param triggers indicator injection (size N+1=6).
    m.register_parameter("p", default=np.array([1.0]), differentiable=True, splits="stagewise")
    m.register_parameter("mass", default=np.array([2.0]), differentiable=False)
    r = repr(m)
    # Indicator must not appear as a row.
    assert "indicator" not in r
    # mass default contributes 1; indicator contributes N+1 = 6; total 7.
    assert "non_differentiable_flat=7" in r
    # Only 'mass' shows up as a non-differentiable data row.
    assert "  non-differentiable:" in r
