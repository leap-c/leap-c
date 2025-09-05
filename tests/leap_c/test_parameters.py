import gymnasium as gym
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


def test_learnable_params_lower_bound():
    """Test learnable_params_lower_bound method with various parameter types."""
    params = [
        # Unbounded parameter (bounds=None)
        Parameter(name="unbounded", default=np.array([1.0]), interface="learnable"),
        # Bounded parameter with bounds
        Parameter(
            name="bounded",
            default=np.array([2.0, 3.0]),
            space=gym.spaces.Box(
                low=np.array([-1.0, -2.0]),
                high=np.array([10.0, 20.0]),
                dtype=np.float32,
            ),
            interface="learnable",
        ),
        # Fixed parameter (should not appear in bounds)
        Parameter(
            name="fixed",
            default=np.array([4.0]),
            space=gym.spaces.Box(
                low=np.array([0.0]), high=np.array([100.0]), dtype=np.float32
            ),
            interface="fix",
        ),
        # Matrix parameter with bounds
        Parameter(
            name="matrix",
            default=np.array([[5.0, 6.0], [7.0, 8.0]]),
            space=gym.spaces.Box(
                low=np.array([[0.0, 1.0], [2.0, 3.0]]),
                high=np.array([[50.0, 60.0], [70.0, 80.0]]),
                dtype=np.float32,
            ),
            interface="learnable",
        ),
    ]

    manager = ParameterManager(params)

    # unbounded uses -inf, bounded uses specified values, matrix is flattened
    expected = np.array([-np.inf, -1.0, -2.0, 0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(manager.get_param_space().low, expected)


def test_learnable_params_upper_bound():
    """Test learnable_params_upper_bound method with various parameter types."""
    params = [
        # Unbounded parameter (bounds=None)
        Parameter(name="unbounded", default=np.array([1.0]), interface="learnable"),
        # Bounded parameter with bounds
        Parameter(
            name="bounded",
            default=np.array([2.0, 3.0]),
            space=gym.spaces.Box(
                low=np.array([-1.0, -2.0]),
                high=np.array([10.0, 20.0]),
                dtype=np.float32,
            ),
            interface="learnable",
        ),
        # Fixed parameter (should not appear in bounds)
        Parameter(
            name="fixed",
            default=np.array([4.0]),
            space=gym.spaces.Box(
                low=np.array([0.0]), high=np.array([100.0]), dtype=np.float32
            ),
            interface="fix",
        ),
        # Matrix parameter with bounds
        Parameter(
            name="matrix",
            default=np.array([[5.0, 6.0], [7.0, 8.0]]),
            space=gym.spaces.Box(
                low=np.array([[0.0, 1.0], [2.0, 3.0]]),
                high=np.array([[50.0, 60.0], [70.0, 80.0]]),
                dtype=np.float32,
            ),
            interface="learnable",
        ),
    ]

    manager = ParameterManager(params)

    # unbounded uses +inf, bounded uses specified values, matrix is flattened
    expected = np.array([np.inf, 10.0, 20.0, 50.0, 60.0, 70.0, 80.0])
    np.testing.assert_array_equal(manager.get_param_space().high, expected)


def test_learnable_params_bounds_no_learnable():
    """Test bounds methods when no parameters are learnable."""
    params = [
        Parameter(name="fixed1", default=np.array([1.0]), interface="fix"),
        Parameter(name="fixed2", default=np.array([2.0]), interface="non-learnable"),
    ]

    manager = ParameterManager(params)

    # Should return empty arrays
    np.testing.assert_array_equal(manager.get_param_space().low, np.array([]))
    np.testing.assert_array_equal(manager.get_param_space().high, np.array([]))


def test_learnable_params_bounds_consistency():
    """Test that bounds methods return arrays consistent with learnable_array order."""
    params = [
        Parameter(
            name="c",
            default=np.array([3.0]),
            space=gym.spaces.Box(
                low=np.array([-3.0]), high=np.array([30.0]), dtype=np.float32
            ),
            interface="learnable",
        ),
        Parameter(
            name="a",
            default=np.array([1.0, 2.0]),
            space=gym.spaces.Box(
                low=np.array([-1.0, -2.0]),
                high=np.array([10.0, 20.0]),
                dtype=np.float32,
            ),
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
    np.testing.assert_array_equal(manager.get_param_space().low, expected_lower)

    # Check upper bounds: c, a, b (unbounded = +inf)
    expected_upper = np.array([30.0, 10.0, 20.0, np.inf, np.inf, np.inf])
    np.testing.assert_array_equal(manager.get_param_space().high, expected_upper)


def test_learnable_params_bounds_mixed_bounded_unbounded():
    """Test bounds methods with mixed bounded and unbounded parameters."""
    params = [
        # Partially bounded (only lower bound)
        Parameter(
            name="lower_only",
            default=np.array([1.0, 2.0]),
            space=gym.spaces.Box(
                low=np.array([0.0, -1.0]),
                high=np.array([np.inf, np.inf]),
                dtype=np.float32,
            ),
            interface="learnable",
        ),
        # Partially bounded (only upper bound)
        Parameter(
            name="upper_only",
            default=np.array([3.0]),
            space=gym.spaces.Box(
                low=np.array([-np.inf]), high=np.array([100.0]), dtype=np.float32
            ),
            interface="learnable",
        ),
        # Fully bounded
        Parameter(
            name="fully_bounded",
            default=np.array([4.0, 5.0]),
            space=gym.spaces.Box(
                low=np.array([-10.0, -20.0]),
                high=np.array([10.0, 20.0]),
                dtype=np.float32,
            ),
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

    expected_lower = np.array([0.0, -1.0, -np.inf, -10.0, -20.0, -np.inf])
    np.testing.assert_array_equal(manager.get_param_space().low, expected_lower)

    expected_upper = np.array([np.inf, np.inf, 100.0, 10.0, 20.0, np.inf])
    np.testing.assert_array_equal(manager.get_param_space().high, expected_upper)
