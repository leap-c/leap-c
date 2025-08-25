import numpy as np
import pytest
from leap_c.parameters import Parameter, ParameterManager


def test_parameter_manager_learnable_params():
    """Test ParameterManager with various parameter types and learnable functionality."""
    # Create test parameters with different shapes and interfaces
    params = [
        # Scalar parameters
        Parameter(name="scalar_fix", default=np.array([1.0]), interface="fix"),
        Parameter(
            name="scalar_learnable", default=np.array([2.0]), interface="learnable"
        ),
        Parameter(
            name="scalar_non_learnable",
            default=np.array([3.0]),
            interface="non-learnable",
        ),
        # Vector parameters
        Parameter(
            name="vector_learnable",
            default=np.array([4.0, 5.0, 6.0]),
            interface="learnable",
        ),
        Parameter(name="vector_fix", default=np.array([7.0, 8.0]), interface="fix"),
    ]

    # Initialize ParameterManager
    manager = ParameterManager(params)

    # Test that parameters are stored correctly
    assert len(manager.parameters) == 5
    assert "scalar_fix" in manager.parameters
    assert "scalar_learnable" in manager.parameters
    assert "vector_learnable" in manager.parameters

    # Test learnable parameter mapping
    expected_learnable_params = ["scalar_learnable", "vector_learnable"]
    assert len(manager.learnable_parameters) == len(expected_learnable_params)

    for param_name in expected_learnable_params:
        assert param_name in manager.learnable_parameters

    # Test learnable parameter indices and shapes
    assert manager.learnable_parameters["scalar_learnable"]["start_idx"] == 0
    assert manager.learnable_parameters["scalar_learnable"]["end_idx"] == 1
    assert manager.learnable_parameters["scalar_learnable"]["shape"] == (1,)

    assert manager.learnable_parameters["vector_learnable"]["start_idx"] == 1
    assert manager.learnable_parameters["vector_learnable"]["end_idx"] == 4
    assert manager.learnable_parameters["vector_learnable"]["shape"] == (3,)

    # Test flattened learnable array
    expected_learnable_array = np.array([2.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(manager.learnable_array, expected_learnable_array)


def test_parameter_manager_no_learnable_params():
    """Test ParameterManager when no parameters are learnable."""
    params = [
        Parameter(name="param1", default=np.array([1.0]), interface="fix"),
        Parameter(
            name="param2", default=np.array([2.0, 3.0]), interface="non-learnable"
        ),
    ]

    manager = ParameterManager(params)

    # Should have empty learnable structures
    assert len(manager.learnable_parameters) == 0
    assert manager.learnable_array.size == 0


def test_parameter_manager_matrix_support():
    """Test that matrix parameters (ndim > 1) are now supported."""
    params = [
        Parameter(
            name="matrix_param",
            default=np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2D matrix
            interface="fix",
        ),
        Parameter(
            name="learnable_matrix",
            default=np.array([[5.0, 6.0], [7.0, 8.0]]),  # 2D learnable matrix
            interface="learnable",
        ),
    ]

    manager = ParameterManager(params)
    assert len(manager.parameters) == 2

    # Test that learnable matrix is handled correctly
    assert "learnable_matrix" in manager.learnable_parameters
    assert manager.learnable_parameters["learnable_matrix"]["shape"] == (2, 2)

    # Should be flattened to [5.0, 6.0, 7.0, 8.0] in learnable_array
    expected_array = np.array([5.0, 6.0, 7.0, 8.0])
    np.testing.assert_array_equal(manager.learnable_array, expected_array)


def test_parameter_manager_get_method():
    """Test the get method for retrieving parameters."""
    params = [
        Parameter(
            name="test_param", default=np.array([1.0, 2.0]), interface="learnable"
        ),
    ]

    manager = ParameterManager(params)

    # Test successful retrieval
    retrieved_param = manager.get("test_param")
    assert retrieved_param.name == "test_param"
    np.testing.assert_array_equal(retrieved_param.default, np.array([1.0, 2.0]))
    assert retrieved_param.interface == "learnable"

    # Test retrieval of non-existent parameter
    with pytest.raises(KeyError, match="Parameter 'nonexistent' not found"):
        manager.get("nonexistent")


def test_parameter_manager_learnable_array_order():
    """Test that learnable parameters are ordered correctly in the flattened array."""
    params = [
        Parameter(name="c", default=np.array([3.0]), interface="learnable"),
        Parameter(name="a", default=np.array([1.0, 2.0]), interface="learnable"),
        Parameter(name="b", default=np.array([4.0, 5.0, 6.0]), interface="fix"),
        Parameter(name="d", default=np.array([7.0]), interface="learnable"),
    ]

    manager = ParameterManager(params)

    # Should only include learnable parameters in order they were added
    # c: [3.0] -> indices 0:1
    # a: [1.0, 2.0] -> indices 1:3
    # d: [7.0] -> indices 3:4
    expected_array = np.array([3.0, 1.0, 2.0, 7.0])
    np.testing.assert_array_equal(manager.learnable_array, expected_array)

    assert manager.learnable_parameters["c"]["start_idx"] == 0
    assert manager.learnable_parameters["c"]["end_idx"] == 1
    assert manager.learnable_parameters["a"]["start_idx"] == 1
    assert manager.learnable_parameters["a"]["end_idx"] == 3
    assert manager.learnable_parameters["d"]["start_idx"] == 3
    assert manager.learnable_parameters["d"]["end_idx"] == 4


def test_combine_learnable_parameter_values_default_only():
    """Test combine_learnable_parameter_values with default values only."""
    params = [
        Parameter(name="a", default=np.array([1.0]), interface="learnable"),
        Parameter(name="b", default=np.array([2.0, 3.0]), interface="learnable"),
        Parameter(name="c", default=np.array([4.0]), interface="fix"),
    ]

    manager = ParameterManager(params)

    # Test with default batch_size=1
    result = manager.combine_learnable_parameter_values()
    expected = np.array([[1.0, 2.0, 3.0]])  # Only learnable params
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (1, 3)

    # Test with specific batch_size
    result = manager.combine_learnable_parameter_values(batch_size=2)
    expected = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)


def test_combine_learnable_parameter_values_with_overwrites():
    """Test combine_learnable_parameter_values with parameter overwrites."""
    params = [
        Parameter(name="scalar", default=np.array([1.0]), interface="learnable"),
        Parameter(name="vector", default=np.array([2.0, 3.0]), interface="learnable"),
        Parameter(name="fixed", default=np.array([99.0]), interface="fix"),
    ]

    manager = ParameterManager(params)

    # Test overwriting scalar parameter
    overwrite_scalar = np.array([[10.0], [20.0]])
    result = manager.combine_learnable_parameter_values(scalar=overwrite_scalar)

    expected = np.array(
        [
            [10.0, 2.0, 3.0],  # scalar overwritten, vector default
            [20.0, 2.0, 3.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)

    # Test overwriting vector parameter
    overwrite_vector = np.array([[100.0, 200.0], [300.0, 400.0]])
    result = manager.combine_learnable_parameter_values(vector=overwrite_vector)

    expected = np.array(
        [
            [1.0, 100.0, 200.0],  # scalar default, vector overwritten
            [1.0, 300.0, 400.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)

    # Test overwriting both parameters
    result = manager.combine_learnable_parameter_values(
        scalar=overwrite_scalar, vector=overwrite_vector
    )

    expected = np.array(
        [
            [10.0, 100.0, 200.0],  # both overwritten
            [20.0, 300.0, 400.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)


def test_combine_learnable_parameter_values_batch_size_inference():
    """Test that batch_size is correctly inferred from overwrite parameters."""
    params = [
        Parameter(name="param", default=np.array([1.0]), interface="learnable"),
    ]

    manager = ParameterManager(params)

    # Batch size should be inferred from overwrite parameter
    overwrite_param = np.array([[10.0], [20.0], [30.0]])  # batch_size=3
    result = manager.combine_learnable_parameter_values(param=overwrite_param)

    expected = np.array([[10.0], [20.0], [30.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 1)


def test_combine_learnable_parameter_values_no_learnable_params():
    """Test combine_learnable_parameter_values when no parameters are learnable."""
    params = [
        Parameter(name="fixed1", default=np.array([1.0]), interface="fix"),
        Parameter(name="fixed2", default=np.array([2.0]), interface="non-learnable"),
    ]

    manager = ParameterManager(params)

    # Should return empty array with correct batch dimension
    result = manager.combine_learnable_parameter_values(batch_size=2)
    expected = np.empty((2, 0))
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 0)


def test_combine_learnable_parameter_values_error_cases():
    """Test error cases for combine_learnable_parameter_values."""
    params = [
        Parameter(
            name="learnable", default=np.array([1.0, 2.0]), interface="learnable"
        ),
        Parameter(name="fixed", default=np.array([3.0]), interface="fix"),
    ]

    manager = ParameterManager(params)

    # Test overwriting non-learnable parameter
    with pytest.raises(KeyError, match="Parameter 'fixed' is not learnable"):
        manager.combine_learnable_parameter_values(fixed=np.array([[99.0]]))

    # Test overwriting non-existent parameter
    with pytest.raises(KeyError, match="Parameter 'nonexistent' is not learnable"):
        manager.combine_learnable_parameter_values(nonexistent=np.array([[1.0]]))

    # Test shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch for parameter 'learnable'"):
        # learnable param expects 2 values, but providing 3
        wrong_shape = np.array([[1.0, 2.0, 3.0]])
        manager.combine_learnable_parameter_values(learnable=wrong_shape)


def test_combine_learnable_parameter_values_preserves_defaults():
    """Test that combine_learnable_parameter_values doesn't modify the original learnable_array."""
    params = [
        Parameter(name="param", default=np.array([1.0, 2.0]), interface="learnable"),
    ]

    manager = ParameterManager(params)
    original_learnable_array = manager.learnable_array.copy()

    # Perform operation with overwrite
    overwrite = np.array([[10.0, 20.0]])
    result = manager.combine_learnable_parameter_values(param=overwrite)

    # Original learnable_array should remain unchanged
    np.testing.assert_array_equal(manager.learnable_array, original_learnable_array)

    # Result should have the overwritten values
    expected = np.array([[10.0, 20.0]])
    np.testing.assert_array_equal(result, expected)


def test_combine_learnable_parameter_values_complex_scenario():
    """Test combine_learnable_parameter_values with a complex mix of parameters."""
    params = [
        Parameter(name="a", default=np.array([1.0]), interface="learnable"),
        Parameter(name="b", default=np.array([2.0, 3.0]), interface="fix"),
        Parameter(name="c", default=np.array([4.0, 5.0, 6.0]), interface="learnable"),
        Parameter(name="d", default=np.array([7.0]), interface="non-learnable"),
        Parameter(name="e", default=np.array([8.0, 9.0]), interface="learnable"),
    ]

    manager = ParameterManager(params)

    expected_default = np.array([1.0, 4.0, 5.0, 6.0, 8.0, 9.0])
    np.testing.assert_array_equal(manager.learnable_array, expected_default)

    # Test with partial overwrites
    overwrite_a = np.array([[100.0], [200.0], [300.0]])
    overwrite_e = np.array([[800.0, 900.0], [801.0, 901.0], [802.0, 902.0]])

    result = manager.combine_learnable_parameter_values(a=overwrite_a, e=overwrite_e)

    expected = np.array(
        [
            [100.0, 4.0, 5.0, 6.0, 800.0, 900.0],  # a and e overwritten
            [200.0, 4.0, 5.0, 6.0, 801.0, 901.0],  # c keeps defaults
            [300.0, 4.0, 5.0, 6.0, 802.0, 902.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 6)


def test_combine_learnable_parameter_values_with_matrices():
    """Test combine_learnable_parameter_values with matrix parameters."""
    params = [
        Parameter(name="scalar", default=np.array([1.0]), interface="learnable"),
        Parameter(
            name="matrix",
            default=np.array([[2.0, 3.0], [4.0, 5.0]]),
            interface="learnable",
        ),
        Parameter(
            name="fixed_matrix",
            default=np.array([[99.0, 98.0], [97.0, 96.0]]),
            interface="fix",
        ),
    ]

    manager = ParameterManager(params)

    # Test default values - matrix should be flattened
    expected_default = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(manager.learnable_array, expected_default)

    # Test shape tracking
    assert manager.learnable_parameters["scalar"]["shape"] == (1,)
    assert manager.learnable_parameters["matrix"]["shape"] == (2, 2)

    # Test with matrix overwrite
    overwrite_matrix = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],  # batch 1: 2x2 matrix
            [[50.0, 60.0], [70.0, 80.0]],  # batch 2: 2x2 matrix
        ]
    )

    result = manager.combine_learnable_parameter_values(matrix=overwrite_matrix)

    expected = np.array(
        [
            [1.0, 10.0, 20.0, 30.0, 40.0],  # scalar default, matrix overwritten
            [1.0, 50.0, 60.0, 70.0, 80.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 5)


def test_combine_learnable_parameter_values_mixed_dimensions():
    """Test combine_learnable_parameter_values with parameters of various dimensions."""
    params = [
        Parameter(name="scalar", default=np.array([1.0]), interface="learnable"),
        Parameter(name="vector", default=np.array([2.0, 3.0]), interface="learnable"),
        Parameter(
            name="matrix",
            default=np.array([[4.0, 5.0], [6.0, 7.0]]),
            interface="learnable",
        ),
        Parameter(
            name="tensor",
            default=np.array([[[8.0, 9.0], [10.0, 11.0]]]),
            interface="learnable",
        ),
    ]

    manager = ParameterManager(params)

    # Test that all parameters are correctly flattened
    expected_default = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    )
    np.testing.assert_array_equal(manager.learnable_array, expected_default)

    # Test shape tracking
    assert manager.learnable_parameters["scalar"]["shape"] == (1,)
    assert manager.learnable_parameters["vector"]["shape"] == (2,)
    assert manager.learnable_parameters["matrix"]["shape"] == (2, 2)
    assert manager.learnable_parameters["tensor"]["shape"] == (1, 2, 2)

    # Test overwriting different dimension parameters
    overwrite_matrix = np.array([[[100.0, 200.0], [300.0, 400.0]]])
    overwrite_tensor = np.array([[[[800.0, 900.0], [1000.0, 1100.0]]]])

    result = manager.combine_learnable_parameter_values(
        matrix=overwrite_matrix, tensor=overwrite_tensor
    )

    expected = np.array(
        [
            [
                1.0,
                2.0,
                3.0,  # scalar and vector defaults
                100.0,
                200.0,
                300.0,
                400.0,  # matrix overwritten
                800.0,
                900.0,
                1000.0,
                1100.0,  # tensor overwritten
            ]
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (1, 11)


def test_learnable_params_lower_bound():
    """Test learnable_params_lower_bound method with various parameter types."""
    params = [
        # Unbounded parameter (lower_bound=None)
        Parameter(name="unbounded", default=np.array([1.0]), interface="learnable"),
        # Bounded parameter with lower_bound
        Parameter(
            name="bounded",
            default=np.array([2.0, 3.0]),
            lower_bound=np.array([-1.0, -2.0]),
            interface="learnable",
        ),
        # Fixed parameter (should not appear in bounds)
        Parameter(
            name="fixed",
            default=np.array([4.0]),
            lower_bound=np.array([0.0]),
            interface="fix",
        ),
        # Matrix parameter with bounds
        Parameter(
            name="matrix",
            default=np.array([[5.0, 6.0], [7.0, 8.0]]),
            lower_bound=np.array([[0.0, 1.0], [2.0, 3.0]]),
            interface="learnable",
        ),
    ]

    manager = ParameterManager(params)
    bounds = manager.learnable_params_lower_bound()

    # unbounded uses -inf, bounded uses specified values, matrix is flattened
    expected = np.array([-np.inf, -1.0, -2.0, 0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(bounds, expected)


def test_learnable_params_upper_bound():
    """Test learnable_params_upper_bound method with various parameter types."""
    params = [
        # Unbounded parameter (upper_bound=None)
        Parameter(name="unbounded", default=np.array([1.0]), interface="learnable"),
        # Bounded parameter with upper_bound
        Parameter(
            name="bounded",
            default=np.array([2.0, 3.0]),
            upper_bound=np.array([10.0, 20.0]),
            interface="learnable",
        ),
        # Fixed parameter (should not appear in bounds)
        Parameter(
            name="fixed",
            default=np.array([4.0]),
            upper_bound=np.array([100.0]),
            interface="fix",
        ),
        # Matrix parameter with bounds
        Parameter(
            name="matrix",
            default=np.array([[5.0, 6.0], [7.0, 8.0]]),
            upper_bound=np.array([[50.0, 60.0], [70.0, 80.0]]),
            interface="learnable",
        ),
    ]

    manager = ParameterManager(params)
    bounds = manager.learnable_params_upper_bound()

    # unbounded uses +inf, bounded uses specified values, matrix is flattened
    expected = np.array([np.inf, 10.0, 20.0, 50.0, 60.0, 70.0, 80.0])
    np.testing.assert_array_equal(bounds, expected)


def test_learnable_params_bounds_no_learnable():
    """Test bounds methods when no parameters are learnable."""
    params = [
        Parameter(name="fixed1", default=np.array([1.0]), interface="fix"),
        Parameter(name="fixed2", default=np.array([2.0]), interface="non-learnable"),
    ]

    manager = ParameterManager(params)

    lower_bounds = manager.learnable_params_lower_bound()
    upper_bounds = manager.learnable_params_upper_bound()

    # Should return empty arrays
    assert lower_bounds.size == 0
    assert upper_bounds.size == 0
    np.testing.assert_array_equal(lower_bounds, np.array([]))
    np.testing.assert_array_equal(upper_bounds, np.array([]))


def test_learnable_params_bounds_consistency():
    """Test that bounds methods return arrays consistent with learnable_array order."""
    params = [
        Parameter(
            name="c",
            default=np.array([3.0]),
            lower_bound=np.array([-3.0]),
            upper_bound=np.array([30.0]),
            interface="learnable",
        ),
        Parameter(
            name="a",
            default=np.array([1.0, 2.0]),
            lower_bound=np.array([-1.0, -2.0]),
            upper_bound=np.array([10.0, 20.0]),
            interface="learnable",
        ),
        Parameter(
            name="fixed",
            default=np.array([99.0]),
            interface="fix",
        ),
        Parameter(
            name="b",
            default=np.array([4.0, 5.0, 6.0]),
            interface="learnable",  # No bounds specified
        ),
    ]

    manager = ParameterManager(params)

    # Check learnable_array order: c, a, b (order they were added)
    expected_values = np.array([3.0, 1.0, 2.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(manager.learnable_array, expected_values)

    # Check lower bounds: c, a, b (unbounded = -inf)
    expected_lower = np.array([-3.0, -1.0, -2.0, -np.inf, -np.inf, -np.inf])
    lower_bounds = manager.learnable_params_lower_bound()
    np.testing.assert_array_equal(lower_bounds, expected_lower)

    # Check upper bounds: c, a, b (unbounded = +inf)
    expected_upper = np.array([30.0, 10.0, 20.0, np.inf, np.inf, np.inf])
    upper_bounds = manager.learnable_params_upper_bound()
    np.testing.assert_array_equal(upper_bounds, expected_upper)


def test_learnable_params_bounds_mixed_bounded_unbounded():
    """Test bounds methods with mixed bounded and unbounded parameters."""
    params = [
        # Partially bounded (only lower bound)
        Parameter(
            name="lower_only",
            default=np.array([1.0, 2.0]),
            lower_bound=np.array([0.0, -1.0]),
            interface="learnable",
        ),
        # Partially bounded (only upper bound)
        Parameter(
            name="upper_only",
            default=np.array([3.0]),
            upper_bound=np.array([100.0]),
            interface="learnable",
        ),
        # Fully bounded
        Parameter(
            name="fully_bounded",
            default=np.array([4.0, 5.0]),
            lower_bound=np.array([-10.0, -20.0]),
            upper_bound=np.array([10.0, 20.0]),
            interface="learnable",
        ),
        # Unbounded
        Parameter(
            name="unbounded",
            default=np.array([6.0]),
            interface="learnable",
        ),
    ]

    manager = ParameterManager(params)

    lower_bounds = manager.learnable_params_lower_bound()
    upper_bounds = manager.learnable_params_upper_bound()

    expected_lower = np.array([0.0, -1.0, -np.inf, -10.0, -20.0, -np.inf])
    np.testing.assert_array_equal(lower_bounds, expected_lower)

    expected_upper = np.array([np.inf, np.inf, 100.0, 10.0, 20.0, np.inf])
    np.testing.assert_array_equal(upper_bounds, expected_upper)


def test_parameter_manager_non_learnable_params():
    """Test ParameterManager with various parameter types and non-learnable functionality."""
    # Create test parameters with different shapes and interfaces
    params = [
        # Scalar parameters
        Parameter(name="scalar_fix", default=np.array([1.0]), interface="fix"),
        Parameter(
            name="scalar_non_learnable",
            default=np.array([2.0]),
            interface="non-learnable",
        ),
        Parameter(
            name="scalar_learnable",
            default=np.array([3.0]),
            interface="learnable",
        ),
        # Vector parameters
        Parameter(
            name="vector_non_learnable",
            default=np.array([4.0, 5.0, 6.0]),
            interface="non-learnable",
        ),
        Parameter(name="vector_fix", default=np.array([7.0, 8.0]), interface="fix"),
    ]

    # Initialize ParameterManager
    manager = ParameterManager(params)

    # Test that parameters are stored correctly
    assert len(manager.parameters) == 5
    assert "scalar_fix" in manager.parameters
    assert "scalar_non_learnable" in manager.parameters
    assert "vector_non_learnable" in manager.parameters

    # Test non-learnable parameter mapping
    expected_non_learnable_params = ["scalar_non_learnable", "vector_non_learnable"]
    assert len(manager.non_learnable_parameters) == len(expected_non_learnable_params)

    for param_name in expected_non_learnable_params:
        assert param_name in manager.non_learnable_parameters

    # Test non-learnable parameter indices and shapes
    assert manager.non_learnable_parameters["scalar_non_learnable"]["start_idx"] == 0
    assert manager.non_learnable_parameters["scalar_non_learnable"]["end_idx"] == 1
    assert manager.non_learnable_parameters["scalar_non_learnable"]["shape"] == (1,)

    assert manager.non_learnable_parameters["vector_non_learnable"]["start_idx"] == 1
    assert manager.non_learnable_parameters["vector_non_learnable"]["end_idx"] == 4
    assert manager.non_learnable_parameters["vector_non_learnable"]["shape"] == (3,)

    # Test flattened non-learnable array
    expected_non_learnable_array = np.array([2.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(
        manager.non_learnable_array, expected_non_learnable_array
    )


def test_parameter_manager_no_non_learnable_params():
    """Test ParameterManager when no parameters are non-learnable."""
    params = [
        Parameter(name="param1", default=np.array([1.0]), interface="fix"),
        Parameter(name="param2", default=np.array([2.0, 3.0]), interface="learnable"),
    ]

    manager = ParameterManager(params)

    # Should have empty non-learnable structures
    assert len(manager.non_learnable_parameters) == 0
    assert manager.non_learnable_array.size == 0


def test_parameter_manager_non_learnable_matrix_support():
    """Test that matrix parameters (ndim > 1) are supported for non-learnable."""
    params = [
        Parameter(
            name="matrix_param",
            default=np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2D matrix
            interface="fix",
        ),
        Parameter(
            name="non_learnable_matrix",
            default=np.array([[5.0, 6.0], [7.0, 8.0]]),  # 2D non-learnable matrix
            interface="non-learnable",
        ),
    ]

    manager = ParameterManager(params)
    assert len(manager.parameters) == 2

    # Test that non-learnable matrix is handled correctly
    assert "non_learnable_matrix" in manager.non_learnable_parameters
    assert manager.non_learnable_parameters["non_learnable_matrix"]["shape"] == (2, 2)

    # Should be flattened to [5.0, 6.0, 7.0, 8.0] in non_learnable_array
    expected_array = np.array([5.0, 6.0, 7.0, 8.0])
    np.testing.assert_array_equal(manager.non_learnable_array, expected_array)


def test_parameter_manager_non_learnable_array_order():
    """Test that non-learnable parameters are ordered correctly in the flattened array."""
    params = [
        Parameter(name="c", default=np.array([3.0]), interface="non-learnable"),
        Parameter(name="a", default=np.array([1.0, 2.0]), interface="non-learnable"),
        Parameter(name="b", default=np.array([4.0, 5.0, 6.0]), interface="fix"),
        Parameter(name="d", default=np.array([7.0]), interface="non-learnable"),
    ]

    manager = ParameterManager(params)

    # Should only include non-learnable parameters in order they were added
    # c: [3.0] -> indices 0:1
    # a: [1.0, 2.0] -> indices 1:3
    # d: [7.0] -> indices 3:4
    expected_array = np.array([3.0, 1.0, 2.0, 7.0])
    np.testing.assert_array_equal(manager.non_learnable_array, expected_array)

    assert manager.non_learnable_parameters["c"]["start_idx"] == 0
    assert manager.non_learnable_parameters["c"]["end_idx"] == 1
    assert manager.non_learnable_parameters["a"]["start_idx"] == 1
    assert manager.non_learnable_parameters["a"]["end_idx"] == 3
    assert manager.non_learnable_parameters["d"]["start_idx"] == 3
    assert manager.non_learnable_parameters["d"]["end_idx"] == 4


def test_combine_non_learnable_parameter_values_default_only():
    """Test combine_non_learnable_parameter_values with default values only."""
    params = [
        Parameter(name="a", default=np.array([1.0]), interface="non-learnable"),
        Parameter(name="b", default=np.array([2.0, 3.0]), interface="non-learnable"),
        Parameter(name="c", default=np.array([4.0]), interface="fix"),
    ]

    manager = ParameterManager(params)

    # Test with default batch_size=1
    result = manager.combine_non_learnable_parameter_values()
    expected = np.array([[1.0, 2.0, 3.0]])  # Only non-learnable params
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (1, 3)

    # Test with specific batch_size
    result = manager.combine_non_learnable_parameter_values(batch_size=2)
    expected = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)


def test_combine_non_learnable_parameter_values_with_overwrites():
    """Test combine_non_learnable_parameter_values with parameter overwrites."""
    params = [
        Parameter(name="scalar", default=np.array([1.0]), interface="non-learnable"),
        Parameter(
            name="vector", default=np.array([2.0, 3.0]), interface="non-learnable"
        ),
        Parameter(name="fixed", default=np.array([99.0]), interface="fix"),
    ]

    manager = ParameterManager(params)

    # Test overwriting scalar parameter
    overwrite_scalar = np.array([[10.0], [20.0]])
    result = manager.combine_non_learnable_parameter_values(scalar=overwrite_scalar)

    expected = np.array(
        [
            [10.0, 2.0, 3.0],  # scalar overwritten, vector default
            [20.0, 2.0, 3.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)

    # Test overwriting vector parameter
    overwrite_vector = np.array([[100.0, 200.0], [300.0, 400.0]])
    result = manager.combine_non_learnable_parameter_values(vector=overwrite_vector)

    expected = np.array(
        [
            [1.0, 100.0, 200.0],  # scalar default, vector overwritten
            [1.0, 300.0, 400.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)

    # Test overwriting both parameters
    result = manager.combine_non_learnable_parameter_values(
        scalar=overwrite_scalar, vector=overwrite_vector
    )

    expected = np.array(
        [
            [10.0, 100.0, 200.0],  # both overwritten
            [20.0, 300.0, 400.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)


def test_combine_non_learnable_parameter_values_batch_size_inference():
    """Test that batch_size is correctly inferred from overwrite parameters."""
    params = [
        Parameter(name="param", default=np.array([1.0]), interface="non-learnable"),
    ]

    manager = ParameterManager(params)

    # Batch size should be inferred from overwrite parameter
    overwrite_param = np.array([[10.0], [20.0], [30.0]])  # batch_size=3
    result = manager.combine_non_learnable_parameter_values(param=overwrite_param)

    expected = np.array([[10.0], [20.0], [30.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 1)


def test_combine_non_learnable_parameter_values_no_non_learnable_params():
    """Test combine_non_learnable_parameter_values when no parameters are non-learnable."""
    params = [
        Parameter(name="fixed1", default=np.array([1.0]), interface="fix"),
        Parameter(name="learnable2", default=np.array([2.0]), interface="learnable"),
    ]

    manager = ParameterManager(params)

    # Should return empty array with correct batch dimension
    result = manager.combine_non_learnable_parameter_values(batch_size=2)
    expected = np.empty((2, 0))
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 0)


def test_combine_non_learnable_parameter_values_error_cases():
    """Test error cases for combine_non_learnable_parameter_values."""
    params = [
        Parameter(
            name="non_learnable",
            default=np.array([1.0, 2.0]),
            interface="non-learnable",
        ),
        Parameter(name="fixed", default=np.array([3.0]), interface="fix"),
    ]

    manager = ParameterManager(params)

    # Test overwriting fixed parameter
    with pytest.raises(KeyError, match="Parameter 'fixed' is not non-learnable"):
        manager.combine_non_learnable_parameter_values(fixed=np.array([[99.0]]))

    # Test overwriting non-existent parameter
    with pytest.raises(KeyError, match="Parameter 'nonexistent' is not non-learnable"):
        manager.combine_non_learnable_parameter_values(nonexistent=np.array([[1.0]]))

    # Test shape mismatch
    with pytest.raises(
        ValueError, match="Shape mismatch for parameter 'non_learnable'"
    ):
        # non_learnable param expects 2 values, but providing 3
        wrong_shape = np.array([[1.0, 2.0, 3.0]])
        manager.combine_non_learnable_parameter_values(non_learnable=wrong_shape)


def test_combine_non_learnable_parameter_values_preserves_defaults():
    """Test that combine_non_learnable_parameter_values doesn't modify the original non_learnable_array."""
    params = [
        Parameter(
            name="param", default=np.array([1.0, 2.0]), interface="non-learnable"
        ),
    ]

    manager = ParameterManager(params)
    original_non_learnable_array = manager.non_learnable_array.copy()

    # Perform operation with overwrite
    overwrite = np.array([[10.0, 20.0]])
    result = manager.combine_non_learnable_parameter_values(param=overwrite)

    # Original non_learnable_array should remain unchanged
    np.testing.assert_array_equal(
        manager.non_learnable_array, original_non_learnable_array
    )

    # Result should have the overwritten values
    expected = np.array([[10.0, 20.0]])
    np.testing.assert_array_equal(result, expected)


def test_combine_non_learnable_parameter_values_complex_scenario():
    """Test combine_non_learnable_parameter_values with a complex mix of parameters."""
    params = [
        Parameter(name="a", default=np.array([1.0]), interface="non-learnable"),
        Parameter(name="b", default=np.array([2.0, 3.0]), interface="fix"),
        Parameter(
            name="c", default=np.array([4.0, 5.0, 6.0]), interface="non-learnable"
        ),
        Parameter(name="d", default=np.array([7.0]), interface="learnable"),
        Parameter(name="e", default=np.array([8.0, 9.0]), interface="non-learnable"),
    ]

    manager = ParameterManager(params)

    expected_default = np.array([1.0, 4.0, 5.0, 6.0, 8.0, 9.0])
    np.testing.assert_array_equal(manager.non_learnable_array, expected_default)

    # Test with partial overwrites
    overwrite_a = np.array([[100.0], [200.0], [300.0]])
    overwrite_e = np.array([[800.0, 900.0], [801.0, 901.0], [802.0, 902.0]])

    result = manager.combine_non_learnable_parameter_values(
        a=overwrite_a, e=overwrite_e
    )

    expected = np.array(
        [
            [100.0, 4.0, 5.0, 6.0, 800.0, 900.0],  # a and e overwritten
            [200.0, 4.0, 5.0, 6.0, 801.0, 901.0],  # c keeps defaults
            [300.0, 4.0, 5.0, 6.0, 802.0, 902.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 6)


def test_combine_non_learnable_parameter_values_with_matrices():
    """Test combine_non_learnable_parameter_values with matrix parameters."""
    params = [
        Parameter(name="scalar", default=np.array([1.0]), interface="non-learnable"),
        Parameter(
            name="matrix",
            default=np.array([[2.0, 3.0], [4.0, 5.0]]),
            interface="non-learnable",
        ),
        Parameter(
            name="fixed_matrix",
            default=np.array([[99.0, 98.0], [97.0, 96.0]]),
            interface="fix",
        ),
    ]

    manager = ParameterManager(params)

    # Test default values - matrix should be flattened
    expected_default = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(manager.non_learnable_array, expected_default)

    # Test shape tracking
    assert manager.non_learnable_parameters["scalar"]["shape"] == (1,)
    assert manager.non_learnable_parameters["matrix"]["shape"] == (2, 2)

    # Test with matrix overwrite
    overwrite_matrix = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],  # batch 1: 2x2 matrix
            [[50.0, 60.0], [70.0, 80.0]],  # batch 2: 2x2 matrix
        ]
    )

    result = manager.combine_non_learnable_parameter_values(matrix=overwrite_matrix)

    expected = np.array(
        [
            [1.0, 10.0, 20.0, 30.0, 40.0],  # scalar default, matrix overwritten
            [1.0, 50.0, 60.0, 70.0, 80.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 5)


def test_combine_non_learnable_parameter_values_mixed_dimensions():
    """Test combine_non_learnable_parameter_values with parameters of various dimensions."""
    params = [
        Parameter(name="scalar", default=np.array([1.0]), interface="non-learnable"),
        Parameter(
            name="vector", default=np.array([2.0, 3.0]), interface="non-learnable"
        ),
        Parameter(
            name="matrix",
            default=np.array([[4.0, 5.0], [6.0, 7.0]]),
            interface="non-learnable",
        ),
        Parameter(
            name="tensor",
            default=np.array([[[8.0, 9.0], [10.0, 11.0]]]),
            interface="non-learnable",
        ),
    ]

    manager = ParameterManager(params)

    # Test that all parameters are correctly flattened
    expected_default = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    )
    np.testing.assert_array_equal(manager.non_learnable_array, expected_default)

    # Test shape tracking
    assert manager.non_learnable_parameters["scalar"]["shape"] == (1,)
    assert manager.non_learnable_parameters["vector"]["shape"] == (2,)
    assert manager.non_learnable_parameters["matrix"]["shape"] == (2, 2)
    assert manager.non_learnable_parameters["tensor"]["shape"] == (1, 2, 2)

    # Test overwriting different dimension parameters
    overwrite_matrix = np.array([[[100.0, 200.0], [300.0, 400.0]]])
    overwrite_tensor = np.array([[[[800.0, 900.0], [1000.0, 1100.0]]]])

    result = manager.combine_non_learnable_parameter_values(
        matrix=overwrite_matrix, tensor=overwrite_tensor
    )

    expected = np.array(
        [
            [
                1.0,
                2.0,
                3.0,  # scalar and vector defaults
                100.0,
                200.0,
                300.0,
                400.0,  # matrix overwritten
                800.0,
                900.0,
                1000.0,
                1100.0,  # tensor overwritten
            ]
        ]
    )
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (1, 11)
