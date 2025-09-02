import numpy as np
import pytest
from leap_c.ocp.acados.parameters import Parameter, AcadosParamManager
import casadi as ca


def test_acados_param_manager_basic_initialization():
    """Test basic initialization of AcadosParamManager."""
    params = [
        Parameter(name="scalar", value=np.array([1.0]), interface="fix"),
        Parameter(name="vector", value=np.array([2.0, 3.0]), interface="learnable"),
    ]

    manager = AcadosParamManager(params, N_horizon=10)

    assert len(manager.parameters) == 2
    assert "scalar" in manager.parameters
    assert "vector" in manager.parameters
    assert manager.N_horizon == 10


def test_parameter_interface_fix():
    """Test fixed parameters (interface='fix')."""
    params = [
        Parameter(name="scalar_fix", value=np.array([1.0]), interface="fix"),
        Parameter(name="vector_fix", value=np.array([2.0, 3.0]), interface="fix"),
        Parameter(
            name="matrix_fix", value=np.array([[4.0, 5.0], [6.0, 7.0]]), interface="fix"
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    # Fixed parameters should not appear in learnable or non-learnable structures
    assert len(manager.learnable_parameters.keys()) == 0
    assert len(manager.non_learnable_parameters.keys()) == 0


def test_parameter_interface_learnable_no_vary_stages():
    """Test learnable parameters without vary_stages."""
    params = [
        Parameter(
            name="scalar_learnable", value=np.array([1.0]), interface="learnable"
        ),
        Parameter(
            name="vector_learnable", value=np.array([2.0, 3.0]), interface="learnable"
        ),
        Parameter(
            name="matrix_learnable",
            value=np.array([[4.0, 5.0], [6.0, 7.0]]),
            interface="learnable",
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    # All should appear in learnable_parameters with original names
    assert len(manager.learnable_parameters.keys()) == 3
    assert "scalar_learnable" in manager.learnable_parameters.keys()
    assert "vector_learnable" in manager.learnable_parameters.keys()
    assert "matrix_learnable" in manager.learnable_parameters.keys()

    # Check default values are set correctly (CasADi returns column vectors)
    np.testing.assert_array_equal(
        manager.learnable_parameters_default["scalar_learnable"], np.array([[1.0]])
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_default["vector_learnable"],
        np.array([[2.0], [3.0]]),
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_default["matrix_learnable"],
        np.array([[4.0, 5.0], [6.0, 7.0]]),  # Preserves matrix shape
    )


def test_parameter_interface_learnable_with_vary_stages():
    """Test learnable parameters with vary_stages."""
    params = [
        Parameter(
            name="price",
            value=np.array([10.0]),
            interface="learnable",
            vary_stages=[3, 7],  # Changes at stages 3 and 7
        ),
        Parameter(
            name="demand",
            value=np.array([5.0, 6.0]),
            interface="learnable",
            vary_stages=[2, 5, 8],  # Changes at stages 2, 5, and 8
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=10)

    # Should create staged parameters with {name}_{start}_{end} template
    learnable_keys = list(manager.learnable_parameters.keys())

    # price changes at [3, 7], so we expect: price_0_2, price_3_6, price_7_10
    price_keys = [k for k in learnable_keys if k.startswith("price_")]
    assert len(price_keys) == 3
    assert "price_0_2" in price_keys
    assert "price_3_6" in price_keys
    assert "price_7_10" in price_keys

    # demand changes at [2, 5, 8], so we expect: demand_0_1, demand_2_4, demand_5_7, demand_8_10
    demand_keys = [k for k in learnable_keys if k.startswith("demand_")]
    assert len(demand_keys) == 4
    assert "demand_0_1" in demand_keys
    assert "demand_2_4" in demand_keys
    assert "demand_5_7" in demand_keys
    assert "demand_8_10" in demand_keys

    # Check that values are set correctly for each stage (CasADi format)
    for key in price_keys:
        np.testing.assert_array_equal(
            manager.learnable_parameters_default[key], np.array([[10.0]])
        )

    # TODO: The given value has shape (2,), but casadi.DM is (2,1). Fix this? Also in following tests.
    for key in demand_keys:
        np.testing.assert_array_equal(
            manager.learnable_parameters_default[key], np.array([[5.0], [6.0]])
        )


def test_parameter_interface_non_learnable_no_vary_stages():
    """Test non-learnable parameters without vary_stages."""
    params = [
        Parameter(
            name="scalar_non_learnable",
            value=np.array([1.0]),
            interface="non-learnable",
        ),
        Parameter(
            name="vector_non_learnable",
            value=np.array([2.0, 3.0]),
            interface="non-learnable",
        ),
        Parameter(
            name="matrix_non_learnable",
            value=np.array([[4.0, 5.0], [6.0, 7.0]]),
            interface="non-learnable",
        ),
    ]

    N_horizon = 5

    manager = AcadosParamManager(params, N_horizon=N_horizon)

    # All should appear in non_learnable_parameters with original names
    assert len(manager.non_learnable_parameters.keys()) == 3
    assert "scalar_non_learnable" in manager.non_learnable_parameters.keys()
    assert "vector_non_learnable" in manager.non_learnable_parameters.keys()
    assert "matrix_non_learnable" in manager.non_learnable_parameters.keys()

    assert "scalar_non_learnable" in manager.non_learnable_parameters_default.keys()
    assert "vector_non_learnable" in manager.non_learnable_parameters_default.keys()
    assert "matrix_non_learnable" in manager.non_learnable_parameters_default.keys()

    for stage in range(N_horizon + 1):
        np.testing.assert_array_equal(
            manager.non_learnable_parameters_default["scalar_non_learnable"],
            np.array([[1.0]]),
        )
        np.testing.assert_array_equal(
            manager.non_learnable_parameters_default["vector_non_learnable"],
            np.array([[2.0], [3.0]]),
        )
        np.testing.assert_array_equal(
            manager.non_learnable_parameters_default["matrix_non_learnable"],
            np.array([[4.0, 5.0], [6.0, 7.0]]),
        )


def test_parameter_bounds_learnable():
    """Test parameter bounds for learnable parameters."""
    params = [
        # Unbounded parameter
        Parameter(
            name="unbounded",
            value=np.array([1.0]),
            interface="learnable",
        ),
        # Parameter with only lower bound
        Parameter(
            name="lower_bounded",
            value=np.array([2.0]),
            lower_bound=np.array([0.0]),
            interface="learnable",
        ),
        # Parameter with only upper bound
        Parameter(
            name="upper_bounded",
            value=np.array([3.0]),
            upper_bound=np.array([10.0]),
            interface="learnable",
        ),
        # Fully bounded parameter
        Parameter(
            name="fully_bounded",
            value=np.array([4.0, 5.0]),
            lower_bound=np.array([-1.0, -2.0]),
            upper_bound=np.array([10.0, 20.0]),
            interface="learnable",
        ),
        # Matrix with only lower bounds
        Parameter(
            name="matrix_lower_bounded",
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            lower_bound=np.array([[0.0, 0.0], [0.0, 0.0]]),
            interface="learnable",
        ),
        # Matrix with only upper bounds
        Parameter(
            name="matrix_upper_bounded",
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            upper_bound=np.array([[10.0, 20.0], [30.0, 40.0]]),
            interface="learnable",
        ),
        # Matrix with bounds
        Parameter(
            name="matrix_bounded",
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            lower_bound=np.array([[0.0, 0.0], [0.0, 0.0]]),
            upper_bound=np.array([[10.0, 20.0], [30.0, 40.0]]),
            interface="learnable",
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    # Test that bounds are set correctly for each parameter (CasADi format)
    # lower_bounded should have lower bound set
    np.testing.assert_array_equal(
        manager.learnable_parameters_lb["lower_bounded"], np.array([[0.0]])
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_ub["lower_bounded"], np.array([[+np.inf]])
    )

    # upper_bounded should have upper bound set
    np.testing.assert_array_equal(
        manager.learnable_parameters_lb["upper_bounded"], np.array([[-np.inf]])
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_ub["upper_bounded"], np.array([[10.0]])
    )

    # fully_bounded should have both bounds set
    np.testing.assert_array_equal(
        manager.learnable_parameters_lb["fully_bounded"], np.array([[-1.0], [-2.0]])
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_ub["fully_bounded"], np.array([[10.0], [20.0]])
    )

    # matrix_lower_bounded should have matrix bounds set (preserves matrix shape)
    np.testing.assert_array_equal(
        manager.learnable_parameters_ub["matrix_lower_bounded"],
        np.array([[np.inf, np.inf], [np.inf, np.inf]]),
    )

    # matrix_upper_bounded should have matrix bounds set (preserves matrix shape)
    np.testing.assert_array_equal(
        manager.learnable_parameters_lb["matrix_upper_bounded"],
        np.array([[-np.inf, -np.inf], [-np.inf, -np.inf]]),
    )

    # matrix_bounded should have matrix bounds set (preserves matrix shape)
    np.testing.assert_array_equal(
        manager.learnable_parameters_lb["matrix_bounded"],
        np.array([[0.0, 0.0], [0.0, 0.0]]),
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_ub["matrix_bounded"],
        np.array([[10.0, 20.0], [30.0, 40.0]]),
    )


def test_parameter_bounds_learnable_with_vary_stages():
    """Test parameter bounds for learnable parameters with vary_stages."""
    params = [
        Parameter(
            name="bounded_staged",
            value=np.array([5.0]),
            lower_bound=np.array([0.0]),
            upper_bound=np.array([10.0]),
            interface="learnable",
            vary_stages=[3],  # Changes at stage 3
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    # Should have bounds set for both staged parameters
    assert "bounded_staged_0_2" in manager.learnable_parameters_lb.keys()
    assert "bounded_staged_3_5" in manager.learnable_parameters_lb.keys()

    np.testing.assert_array_equal(
        manager.learnable_parameters_lb["bounded_staged_0_2"], np.array([[0.0]])
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_ub["bounded_staged_0_2"], np.array([[10.0]])
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_lb["bounded_staged_3_5"], np.array([[0.0]])
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_ub["bounded_staged_3_5"], np.array([[10.0]])
    )


def test_vary_stages_clipping_to_horizon():
    """Test that vary_stages are properly clipped to N_horizon."""
    params = [
        Parameter(
            name="clipped",
            value=np.array([1.0]),
            interface="learnable",
            vary_stages=[3, 8, 15, 20],  # Only 3 and 8 should be used (N_horizon=10)
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=10)

    # Should only create parameters up to N_horizon
    learnable_keys = [
        k for k in manager.learnable_parameters.keys() if k.startswith("clipped_")
    ]
    assert len(learnable_keys) == 3  # clipped_0_2, clipped_3_7, clipped_8_10
    assert "clipped_0_2" in learnable_keys
    assert "clipped_3_7" in learnable_keys
    assert "clipped_8_10" in learnable_keys


def test_indicator_creation():
    """Test that indicator is created when vary_stages are used."""
    params_no_vary = [
        Parameter(name="no_vary", value=np.array([1.0]), interface="learnable"),
    ]

    params_with_vary = [
        Parameter(
            name="with_vary",
            value=np.array([1.0]),
            interface="learnable",
            vary_stages=[3],
        ),
    ]

    manager_no_vary = AcadosParamManager(params_no_vary, N_horizon=5)
    manager_with_vary = AcadosParamManager(params_with_vary, N_horizon=5)

    # No vary_stages should not have indicator
    assert "indicator" not in manager_no_vary.non_learnable_parameters.keys()

    # With vary_stages should have indicator
    assert "indicator" in manager_with_vary.non_learnable_parameters.keys()


def test_mixed_parameter_types_and_interfaces():
    """Test complex scenario with mixed parameter types and interfaces."""
    params = [
        # Fixed parameters
        Parameter(name="fix_scalar", value=np.array([1.0]), interface="fix"),
        Parameter(
            name="fix_matrix", value=np.array([[2.0, 3.0], [4.0, 5.0]]), interface="fix"
        ),
        # Learnable parameters without vary_stages
        Parameter(name="learn_scalar", value=np.array([6.0]), interface="learnable"),
        Parameter(
            name="learn_vector", value=np.array([7.0, 8.0]), interface="learnable"
        ),
        # Learnable parameters with vary_stages
        Parameter(
            name="learn_staged",
            value=np.array([9.0]),
            interface="learnable",
            vary_stages=[2, 6],
        ),
        # Non-learnable parameters without vary_stages
        Parameter(
            name="non_learn_scalar", value=np.array([10.0]), interface="non-learnable"
        ),
        Parameter(
            name="non_learn_vector",
            value=np.array([11.0, 12.0]),
            interface="non-learnable",
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=8)

    # Check learnable parameters
    learnable_keys = list(manager.learnable_parameters.keys())
    expected_learnable = [
        "learn_scalar",
        "learn_vector",
        "learn_staged_0_1",
        "learn_staged_2_5",
        "learn_staged_6_8",
    ]
    assert len(learnable_keys) == len(expected_learnable)
    for key in expected_learnable:
        assert key in learnable_keys

    # Check non-learnable parameters (includes indicator)
    non_learnable_keys = list(manager.non_learnable_parameters.keys())
    expected_non_learnable = [
        "non_learn_scalar",
        "non_learn_vector",
        "indicator",
    ]
    assert len(non_learnable_keys) == len(expected_non_learnable)
    for key in expected_non_learnable:
        assert key in non_learnable_keys


# TODO: Rename this test after we rename to def get_learnable_parameter_bounds
def test_get_p_global_bounds():
    """Test get_p_global_bounds method."""
    params = [
        Parameter(
            name="bounded",
            value=np.array([1.0, 2.0]),
            lower_bound=np.array([0.0, -1.0]),
            upper_bound=np.array([10.0, 20.0]),
            interface="learnable",
        ),
        Parameter(name="unbounded", value=np.array([3.0]), interface="learnable"),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    lb, ub = manager.get_learnable_parameters_bounds()

    # Should return flattened arrays with shape (n_params, 1) from CasADi
    expected_lb = np.array([[0.0], [-1.0], [-np.inf]])  # unbounded gets default -inf
    expected_ub = np.array([[10.0], [20.0], [+np.inf]])  # unbounded gets default +inf

    np.testing.assert_array_equal(lb, expected_lb)
    np.testing.assert_array_equal(ub, expected_ub)


def test_get_method_fix_parameters():
    """Test get method for fixed parameters."""
    params = [
        Parameter(name="fixed_param", value=np.array([42.0]), interface="fix"),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    # Should return the parameter value directly
    result = manager.get("fixed_param")
    np.testing.assert_array_equal(result, np.array([42.0]))


def test_get_method_learnable_parameters():
    """Test get method for learnable parameters."""
    params = [
        Parameter(name="learnable_param", value=np.array([1.0]), interface="learnable"),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    # Should return the symbolic variable
    result = manager.get("learnable_param")

    # Check that result has type ca.SX and shape (1,1) and that its name is "learnable_param"
    assert isinstance(result, ca.SX)
    assert result.shape == (1, 1)
    assert result.str() == "learnable_param"


def test_get_method_non_learnable_parameters():
    """Test get method for non-learnable parameters."""
    params = [
        Parameter(
            name="non_learnable_param", value=np.array([1.0]), interface="non-learnable"
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    # Should return the symbolic variable
    result = manager.get("non_learnable_param")

    # Check that result has type ca.SX and shape (1,1) and that its name is "learnable_param"
    assert isinstance(result, ca.SX)
    assert result.shape == (1, 1)
    assert result.str() == "non_learnable_param"


def test_get_method_vary_stages():
    """Test get method for parameters with vary_stages."""
    params = [
        Parameter(
            name="staged_param",
            value=np.array([1.0]),
            interface="learnable",
            vary_stages=[3],
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    # Should return a combination of staged parameters
    result = manager.get("staged_param")

    # Check that result has type ca.SX and shape (1,1)
    assert isinstance(result, ca.SX)
    assert result.shape == (1, 1)

    # TODO: This test might be fragile. Use a casadi.Function to validate the expression.
    assert (
        result.str()
        == "((((indicator_0+indicator_1)+indicator_2)*staged_param_0_2)+(((indicator_3+indicator_4)+indicator_5)*staged_param_3_5))"
    )


def test_get_method_unknown_field():
    """Test get method with unknown field name."""
    params = [
        Parameter(name="existing_param", value=np.array([1.0]), interface="fix"),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    with pytest.raises(ValueError, match="Unknown field: nonexistent"):
        manager.get("nonexistent")


def test_empty_parameter_list():
    """Test AcadosParamManager with empty parameter list."""
    params = []

    with pytest.warns(
        UserWarning, match="Empty parameter list provided to AcadosParamManager"
    ):
        manager = AcadosParamManager(params, N_horizon=5)

    assert len(manager.parameters) == 0
    assert len(manager.learnable_parameters.keys()) == 0
    assert len(manager.non_learnable_parameters.keys()) == 0


def test_parameter_name_with_underscores():
    """Test parameters with underscores in their names (potential conflict with template for stages: {name}_{start}_{end})."""
    params = [
        Parameter(
            name="param_with_underscores",
            value=np.array([1.0]),
            interface="learnable",
            vary_stages=[3],
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    # Should properly handle names with underscores
    learnable_keys = list(manager.learnable_parameters.keys())
    staged_keys = [k for k in learnable_keys if k.startswith("param_with_underscores_")]

    assert len(staged_keys) == 2
    assert "param_with_underscores_0_2" in staged_keys
    assert "param_with_underscores_3_5" in staged_keys

    # Values should be set correctly (CasADi format)
    for key in staged_keys:
        np.testing.assert_array_equal(
            manager.learnable_parameters_default[key], np.array([[1.0]])
        )


def test_large_dimension_parameters():
    """Test that CasADi limitation with >2D arrays is handled gracefully."""
    # CasADi only supports up to 2D arrays, test that 2D arrays are accepted and work as expected.
    params_2d = [
        Parameter(
            name="matrix_param",
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            interface="learnable",
            lower_bound=np.array([[0.0, 0.0], [0.0, 0.0]]),
            upper_bound=np.array([[10.0, 10.0], [10.0, 10.0]]),
        ),
    ]

    manager = AcadosParamManager(params_2d, N_horizon=5)

    # Should handle 2D arrays correctly (flattened in CasADi)
    assert "matrix_param" in manager.learnable_parameters.keys()

    # CasADi preserves matrix shapes
    expected_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_lb = np.array([[0.0, 0.0], [0.0, 0.0]])
    expected_ub = np.array([[10.0, 10.0], [10.0, 10.0]])

    np.testing.assert_array_equal(
        manager.learnable_parameters_default["matrix_param"], expected_value
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_lb["matrix_param"], expected_lb
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_ub["matrix_param"], expected_ub
    )

    # Test that 3D arrays raise an error
    params_3d = [
        Parameter(
            name="tensor_param",
            value=np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            interface="learnable",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Parameter 'tensor_param' has 3 dimensions.*CasADi only supports arrays up to 2 dimensions",
    ):
        AcadosParamManager(params_3d, N_horizon=5)

    # Test that 3D lower bounds raise an error
    params_3d_bounds = [
        Parameter(
            name="tensor_bounds_param",
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            lower_bound=np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
            interface="learnable",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Parameter 'tensor_bounds_param' lower_bound has 3 dimensions.*CasADi only supports arrays up to 2 dimensions",
    ):
        AcadosParamManager(params_3d_bounds, N_horizon=5)

    # Test that 3D upper bounds raise an error
    params_3d_upper_bounds = [
        Parameter(
            name="tensor_upper_bounds_param",
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            upper_bound=np.array(
                [[[10.0, 10.0], [10.0, 10.0]], [[10.0, 10.0], [10.0, 10.0]]]
            ),
            interface="learnable",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Parameter 'tensor_upper_bounds_param' upper_bound has 3 dimensions.*CasADi only supports arrays up to 2 dimensions",
    ):
        AcadosParamManager(params_3d_upper_bounds, N_horizon=5)


# TODO: Do we need this test?
def test_zero_horizon():
    """Test behavior with N_horizon=0."""
    params = [
        Parameter(
            name="zero_horizon",
            value=np.array([1.0]),
            interface="learnable",
            vary_stages=[1, 2],  # Should be filtered out
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=0)

    # vary_stages beyond horizon should be filtered out
    # With N_horizon=0, only stage 0 exists, so no vary_stages should apply
    learnable_keys = list(manager.learnable_parameters.keys())
    assert len(learnable_keys) == 1
    assert "zero_horizon_0_0" in learnable_keys  # Single stage from 0 to 0


def test_combine_parameter_values():
    """Test that combine_parameter_values method exists but may not be fully implemented."""
    params = [
        Parameter(name="test_param", value=np.array([1.0]), interface="non-learnable"),
    ]

    manager = AcadosParamManager(params, N_horizon=5)

    expected = np.ones((2, 6, 1))
    result = manager.combine_non_learnable_parameter_values(batch_size=2)
    np.testing.assert_array_equal(result, expected)


def test_combine_parameter_values_complex():
    """Test combine_parameter_values with mixed parameter types, interfaces, and vary_stages."""
    params = [
        # Scalar parameters
        Parameter(name="scalar_fix", value=np.array([2.0]), interface="fix"),
        Parameter(
            name="scalar_learnable", value=np.array([3.0]), interface="learnable"
        ),
        Parameter(
            name="scalar_non_learnable",
            value=np.array([4.0]),
            interface="non-learnable",
        ),
        Parameter(
            name="scalar_staged",
            value=np.array([5.0]),
            interface="learnable",
            vary_stages=[2, 6],
        ),
        # Vector parameters
        Parameter(name="vector_fix", value=np.array([1.0, 2.0]), interface="fix"),
        Parameter(
            name="vector_learnable",
            value=np.array([6.0, 7.0]),
            interface="learnable",
        ),
        Parameter(
            name="vector_non_learnable",
            value=np.array([8.0, 9.0]),
            interface="non-learnable",
        ),
        Parameter(
            name="vector_staged",
            value=np.array([10.0, 11.0]),
            interface="learnable",
            vary_stages=[3],
        ),
        # Matrix parameters
        Parameter(
            name="matrix_fix",
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            interface="fix",
        ),
        Parameter(
            name="matrix_learnable",
            value=np.array([[12.0, 13.0], [14.0, 15.0]]),
            interface="learnable",
        ),
        Parameter(
            name="matrix_non_learnable",
            value=np.array([[16.0, 17.0], [18.0, 19.0]]),
            interface="non-learnable",
        ),
        Parameter(
            name="matrix_staged",
            value=np.array([[20.0, 21.0], [22.0, 23.0]]),
            interface="learnable",
            vary_stages=[1, 4, 7],
        ),
    ]

    manager = AcadosParamManager(params, N_horizon=8)

    # Test with batch_size=3
    batch_size = 3
    result = manager.combine_non_learnable_parameter_values(batch_size=batch_size)

    # Verify result shape: (batch_size, N_horizon + 1, total_non_learnable_params)
    # Non-learnable params: scalar_non_learnable(1) + vector_non_learnable(2) + matrix_non_learnable(4) + indicator(9) = 16
    expected_shape = (batch_size, manager.N_horizon + 1, 16)
    assert result.shape == expected_shape

    # Verify that the values are correctly replicated across batches and stages
    # All non-learnable parameters should have the same values across all batches
    for batch_idx in range(batch_size):
        for stage_idx in range(manager.N_horizon + 1):
            # Check scalar_non_learnable (first element)
            assert result[batch_idx, stage_idx, 0] == 4.0

            # Check vector_non_learnable (next 2 elements)
            assert result[batch_idx, stage_idx, 1] == 8.0
            assert result[batch_idx, stage_idx, 2] == 9.0

            # Check matrix_non_learnable (next 4 elements)
            # Matrix is flattened in column-major (Fortran) order by CasADi
            expected_matrix_flat = np.array(
                [16.0, 18.0, 17.0, 19.0]
            )  # [[16,17],[18,19]] -> [16,18,17,19]
            np.testing.assert_array_equal(
                result[batch_idx, stage_idx, 3:7], expected_matrix_flat
            )

            # Check indicator values (last 9 elements for N_horizon=8)
            # indicator[stage_idx] should be 1.0, others should be 0.0
            expected_indicator = np.zeros(9)
            expected_indicator[stage_idx] = 1.0
            np.testing.assert_array_equal(
                result[batch_idx, stage_idx, 7:], expected_indicator
            )

    rng = np.random.default_rng(42)

    # Build random overwrites
    vector_non_learnable = rng.random(
        size=(
            batch_size,
            manager.N_horizon + 1,
            manager.non_learnable_parameters_default["vector_non_learnable"].shape[0],
        )
    )

    matrix_non_learnable = rng.random(
        size=(
            batch_size,
            manager.N_horizon + 1,
            manager.non_learnable_parameters_default["matrix_non_learnable"].shape[0],
            manager.non_learnable_parameters_default["matrix_non_learnable"].shape[1],
        )
    )

    result = manager.combine_non_learnable_parameter_values(
        matrix_non_learnable=matrix_non_learnable,
        vector_non_learnable=vector_non_learnable,
    )
