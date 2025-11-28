import re

import casadi as ca
import gymnasium as gym
import numpy as np
import pytest
import torch
from acados_template import AcadosOcp, AcadosOcpSolver

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch


def test_acados_param_manager_basic_initialization():
    """Test basic initialization of AcadosParamManager."""
    params = [
        AcadosParameter(name="scalar", default=np.array([1.0]), interface="fix"),
        AcadosParameter(name="vector", default=np.array([2.0, 3.0]), interface="learnable"),
    ]

    manager = AcadosParameterManager(params, N_horizon=10)

    assert len(manager.parameters) == 2
    assert "scalar" in manager.parameters
    assert "vector" in manager.parameters
    assert manager.N_horizon == 10


def test_parameter_interface_fix():
    """Test fixed parameters (interface='fix')."""
    params = [
        AcadosParameter(name="scalar_fix", default=np.array([1.0]), interface="fix"),
        AcadosParameter(name="vector_fix", default=np.array([2.0, 3.0]), interface="fix"),
        AcadosParameter(
            name="matrix_fix",
            default=np.array([[4.0, 5.0], [6.0, 7.0]]),
            interface="fix",
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=5)

    # Fixed parameters should not appear in learnable or non-learnable structures
    assert len(manager.learnable_parameters.keys()) == 0
    assert len(manager.non_learnable_parameters.keys()) == 0


def test_parameter_interface_learnable_no_vary_stages():
    """Test learnable parameters without vary_stages."""
    params = [
        AcadosParameter(name="scalar_learnable", default=np.array([1.0]), interface="learnable"),
        AcadosParameter(
            name="vector_learnable", default=np.array([2.0, 3.0]), interface="learnable"
        ),
        AcadosParameter(
            name="matrix_learnable",
            default=np.array([[4.0, 5.0], [6.0, 7.0]]),
            interface="learnable",
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=5)

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
    N_horizon = 10
    params = [
        AcadosParameter(
            name="price",
            default=np.array([10.0]),
            interface="learnable",
            end_stages=[3, 7, N_horizon],  # Ends at stages 3 and 7, and horizon (10)
        ),
        AcadosParameter(
            name="demand",
            default=np.array([5.0, 6.0]),
            interface="learnable",
            end_stages=[2, 5, 8, N_horizon],  # Changes at stages 2, 5, 8, and horizon (10)
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=N_horizon)

    # Should create staged parameters with {name}_{start}_{end} template
    learnable_keys = list(manager.learnable_parameters.keys())

    # price changes at [3, 7], so we expect: price_0_2, price_3_6, price_7_10
    price_keys = [k for k in learnable_keys if k.startswith("price_")]
    assert len(price_keys) == 3
    assert "price_0_3" in price_keys
    assert "price_4_7" in price_keys
    assert "price_8_10" in price_keys

    # demand changes at [2, 5, 8], so we expect: demand_0_1, demand_2_4, demand_5_7, demand_8_10
    demand_keys = [k for k in learnable_keys if k.startswith("demand_")]
    assert len(demand_keys) == 4
    assert "demand_0_2" in demand_keys
    assert "demand_3_5" in demand_keys
    assert "demand_6_8" in demand_keys
    assert "demand_9_10" in demand_keys

    # Check that values are set correctly for each stage (CasADi format)
    for key in price_keys:
        np.testing.assert_array_equal(manager.learnable_parameters_default[key], np.array([[10.0]]))

    for key in demand_keys:
        np.testing.assert_array_equal(
            manager.learnable_parameters_default[key], np.array([[5.0], [6.0]])
        )


def test_parameter_interface_non_learnable_no_vary_stages():
    """Test non-learnable parameters without vary_stages."""
    params = [
        AcadosParameter(
            name="scalar_non_learnable",
            default=np.array([1.0]),
            interface="non-learnable",
        ),
        AcadosParameter(
            name="vector_non_learnable",
            default=np.array([2.0, 3.0]),
            interface="non-learnable",
        ),
        AcadosParameter(
            name="matrix_non_learnable",
            default=np.array([[4.0, 5.0], [6.0, 7.0]]),
            interface="non-learnable",
        ),
    ]

    N_horizon = 5

    manager = AcadosParameterManager(params, N_horizon=N_horizon)

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
        AcadosParameter(
            name="unbounded",
            default=np.array([1.0]),
            interface="learnable",
        ),
        # Parameter with bounds
        AcadosParameter(
            name="lower_bounded",
            default=np.array([2.0]),
            space=gym.spaces.Box(low=np.array([0.0]), high=np.array([np.inf])),
            interface="learnable",
        ),
        # Parameter with upper bound only
        AcadosParameter(
            name="upper_bounded",
            default=np.array([3.0]),
            space=gym.spaces.Box(low=np.array([-np.inf]), high=np.array([10.0])),
            interface="learnable",
        ),
        # Fully bounded parameter
        AcadosParameter(
            name="fully_bounded",
            default=np.array([4.0, 5.0]),
            space=gym.spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([10.0, 20.0])),
            interface="learnable",
        ),
        # Matrix with lower bounds
        AcadosParameter(
            name="matrix_lower_bounded",
            default=np.array([[1.0, 2.0], [3.0, 4.0]]),
            space=gym.spaces.Box(
                low=np.array([[0.0, 0.0], [0.0, 0.0]]),
                high=np.array([[np.inf, np.inf], [np.inf, np.inf]]),
            ),
            interface="learnable",
        ),
        # Matrix with upper bounds
        AcadosParameter(
            name="matrix_upper_bounded",
            default=np.array([[1.0, 2.0], [3.0, 4.0]]),
            space=gym.spaces.Box(
                low=np.array([[-np.inf, -np.inf], [-np.inf, -np.inf]]),
                high=np.array([[10.0, 20.0], [30.0, 40.0]]),
            ),
            interface="learnable",
        ),
        # Matrix with bounds
        AcadosParameter(
            name="matrix_bounded",
            default=np.array([[1.0, 2.0], [3.0, 4.0]]),
            space=gym.spaces.Box(
                low=np.array([[0.0, 0.0], [0.0, 0.0]]),
                high=np.array([[10.0, 20.0], [30.0, 40.0]]),
            ),
            interface="learnable",
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=5)

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
    N_horizon = 5
    params = [
        AcadosParameter(
            name="bounded_staged",
            default=np.array([5.0]),
            space=gym.spaces.Box(low=np.array([0.0]), high=np.array([10.0])),
            interface="learnable",
            end_stages=[3, N_horizon],  # Ends at stage 3, and horizon (5)
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=N_horizon)

    # Should have bounds set for both staged parameters
    assert "bounded_staged_0_3" in manager.learnable_parameters_lb.keys()
    assert "bounded_staged_4_5" in manager.learnable_parameters_lb.keys()

    np.testing.assert_array_equal(
        manager.learnable_parameters_lb["bounded_staged_0_3"], np.array([[0.0]])
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_ub["bounded_staged_0_3"], np.array([[10.0]])
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_lb["bounded_staged_4_5"], np.array([[0.0]])
    )
    np.testing.assert_array_equal(
        manager.learnable_parameters_ub["bounded_staged_4_5"], np.array([[10.0]])
    )


def test_vary_stages_last_element_not_valid():
    """Test that ValueError is raised when vary_stages last element is invalid."""
    N_horizon = 10
    params = [
        AcadosParameter(
            name="exceed_horizon",
            default=np.array([1.0]),
            interface="learnable",
            end_stages=[5],  # N_horizon is 10, but last vary_stages is 5
        ),
    ]

    with pytest.raises(
        ValueError,
        match=r"Parameter 'exceed_horizon' has end_stages \[5\] "
        r"but the last element must be either 9 or 10.",
    ):
        AcadosParameterManager(params, N_horizon=N_horizon)


def test_indicator_creation():
    """Test that indicator is created when vary_stages are used."""
    N_horizon = 5
    params_no_vary = [
        AcadosParameter(name="no_vary", default=np.array([1.0]), interface="learnable"),
    ]

    params_with_vary = [
        AcadosParameter(
            name="with_vary",
            default=np.array([1.0]),
            interface="learnable",
            end_stages=[3, N_horizon],  # Ends at stage 3, and horizon (5)
        ),
    ]

    manager_no_vary = AcadosParameterManager(params_no_vary, N_horizon=N_horizon)
    manager_with_vary = AcadosParameterManager(params_with_vary, N_horizon=N_horizon)

    # No vary_stages should not have indicator
    assert "indicator" not in manager_no_vary.non_learnable_parameters.keys()

    # With vary_stages should have indicator
    assert "indicator" in manager_with_vary.non_learnable_parameters.keys()


def test_mixed_parameter_types_and_interfaces():
    """Test complex scenario with mixed parameter types and interfaces."""
    N_horizon = 8
    params = [
        # Fixed parameters
        AcadosParameter(name="fix_scalar", default=np.array([1.0]), interface="fix"),
        AcadosParameter(
            name="fix_matrix",
            default=np.array([[2.0, 3.0], [4.0, 5.0]]),
            interface="fix",
        ),
        # Learnable parameters without vary_stages
        AcadosParameter(name="learn_scalar", default=np.array([6.0]), interface="learnable"),
        AcadosParameter(name="learn_vector", default=np.array([7.0, 8.0]), interface="learnable"),
        # Learnable parameters with vary_stages
        AcadosParameter(
            name="learn_staged",
            default=np.array([9.0]),
            interface="learnable",
            end_stages=[2, 6, N_horizon],
        ),
        # Non-learnable parameters without vary_stages
        AcadosParameter(
            name="non_learn_scalar", default=np.array([10.0]), interface="non-learnable"
        ),
        AcadosParameter(
            name="non_learn_vector",
            default=np.array([11.0, 12.0]),
            interface="non-learnable",
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=N_horizon)

    # Check learnable parameters
    learnable_keys = list(manager.learnable_parameters.keys())
    expected_learnable = [
        "learn_scalar",
        "learn_vector",
        "learn_staged_0_2",
        "learn_staged_3_6",
        "learn_staged_7_8",
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


def test_get_param_space():
    """Test get_param_space method."""
    params = [
        AcadosParameter(
            name="bounded",
            default=np.array([1.0, 2.0]),
            space=gym.spaces.Box(low=np.array([0.0, -1.0]), high=np.array([10.0, 20.0])),
            interface="learnable",
        ),
        AcadosParameter(name="unbounded", default=np.array([3.0]), interface="learnable"),
    ]

    manager = AcadosParameterManager(params, N_horizon=5)

    # Should return flattened arrays
    expected_lb = np.array([0.0, -1.0, -np.inf], dtype=np.float32)  # unbounded gets default -inf
    expected_ub = np.array([10.0, 20.0, +np.inf], dtype=np.float32)  # unbounded gets default +inf

    np.testing.assert_array_equal(manager.get_param_space().low, expected_lb)
    np.testing.assert_array_equal(manager.get_param_space().high, expected_ub)


def test_get_param_space_with_variable_end_stages():
    """Test get_param_space method with parameters that have variable end_stages.

    The parameter space should scale up according to the number of stage variations
    and dimensions of each parameter.
    Each stage variation should create a separate entry in the parameter space.
    """
    N_horizon = 10
    params = [
        # Scalar parameter with 3 stage variations
        AcadosParameter(
            name="scalar",
            default=np.array([5.0]),
            space=gym.spaces.Box(low=np.array([0.0]), high=np.array([20.0])),
            interface="learnable",
            end_stages=[3, 7, N_horizon],
        ),
        # Vector parameter with 4 stage variations
        AcadosParameter(
            name="vector",
            default=np.array([10.0, 15.0]),
            space=gym.spaces.Box(low=np.array([0.0, 5.0]), high=np.array([50.0, 100.0])),
            interface="learnable",
            end_stages=[2, 5, 8, N_horizon],
        ),
        # Scalar parameter with 5 stage variations but no bounds (should get -inf/+inf)
        AcadosParameter(
            name="scalar_unbounded",
            default=np.array([2.5]),
            interface="learnable",
            end_stages=[1, 3, 6, 8, N_horizon],
        ),
        # Matrix parameter with 2 stage variations
        AcadosParameter(
            name="matrix",
            default=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),  # 3x3 matrix
            space=gym.spaces.Box(
                low=np.array([[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]),
                high=np.array([[15.0, 15.0, 15.0], [15.0, 15.0, 15.0], [15.0, 15.0, 15.0]]),
            ),
            interface="learnable",
            end_stages=[4, N_horizon],
        ),
        # Regular parameter without end_stages
        AcadosParameter(
            name="regular_param",
            default=np.array([1.0]),
            space=gym.spaces.Box(low=np.array([-10.0]), high=np.array([10.0])),
            interface="learnable",
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=N_horizon)
    param_space = manager.get_param_space()

    # Expected space dimensions:
    # - scalar: 3 stage variations × 1 dimension = 3 elements
    # - vector: 4 stage variations × 2 dimensions = 8 elements
    # - scalar_unbounded: 5 stage variations × 1 dimension = 5 elements
    # - matrix: 2 stage variations × 9 dimensions (3x3) = 18 elements
    # - regular_param: 1 × 1 dimension = 1 element
    # Total: 3 + 8 + 5 + 18 + 1 = 35 elements
    expected_total_dims = 35

    assert isinstance(param_space, gym.spaces.Box)
    assert param_space.shape == (expected_total_dims,)

    # Verify bounds are replicated correctly for staged parameters
    expected_low = np.array(
        [
            # scalar (3 variations): [0.0, 0.0, 0.0]
            0.0,
            0.0,
            0.0,
            # vector (4 variations × 2 dims): [0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0]
            0.0,
            5.0,
            0.0,
            5.0,
            0.0,
            5.0,
            0.0,
            5.0,
            # scalar_unbounded (5 variations): [-inf, -inf, -inf, -inf, -inf]
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            # matrix (2 variations × 9 dims): [-5.0 repeated 18 times]
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,  # First variation
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,  # Second variation
            # regular_param: [-10.0]
            -10.0,
        ],
        dtype=np.float32,
    )

    expected_high = np.array(
        [
            # scalar (3 variations): [20.0, 20.0, 20.0]
            20.0,
            20.0,
            20.0,
            # vector (4 variations × 2 dims): [50.0, 100.0, ...]
            50.0,
            100.0,
            50.0,
            100.0,
            50.0,
            100.0,
            50.0,
            100.0,
            # scalar_unbounded (5 variations): [+inf, +inf, +inf, +inf, +inf]
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            # matrix (2 variations × 9 dims): [15.0 repeated 18 times]
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,  # First variation
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,  # Second variation
            # regular_param: [10.0]
            10.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(param_space.low, expected_low)
    np.testing.assert_array_equal(param_space.high, expected_high)

    # Verify learnable parameter keys match expected staged parameter names
    learnable_keys = list(manager.learnable_parameters.keys())

    # Check scalar variations (but not scalar_unbounded)
    scalar_keys = [
        k
        for k in learnable_keys
        if k.startswith("scalar_") and not k.startswith("scalar_unbounded_")
    ]
    assert len(scalar_keys) == 3
    assert "scalar_0_3" in scalar_keys
    assert "scalar_4_7" in scalar_keys
    assert "scalar_8_10" in scalar_keys

    # Check vector variations
    vector_keys = [k for k in learnable_keys if k.startswith("vector_")]
    assert len(vector_keys) == 4
    assert "vector_0_2" in vector_keys
    assert "vector_3_5" in vector_keys
    assert "vector_6_8" in vector_keys
    assert "vector_9_10" in vector_keys

    # Check scalar_unbounded variations
    scalar_unbounded_keys = [k for k in learnable_keys if k.startswith("scalar_unbounded_")]
    assert len(scalar_unbounded_keys) == 5
    assert "scalar_unbounded_0_1" in scalar_unbounded_keys
    assert "scalar_unbounded_2_3" in scalar_unbounded_keys
    assert "scalar_unbounded_4_6" in scalar_unbounded_keys
    assert "scalar_unbounded_7_8" in scalar_unbounded_keys
    assert "scalar_unbounded_9_10" in scalar_unbounded_keys

    # Check matrix variations
    matrix_keys = [k for k in learnable_keys if k.startswith("matrix_")]
    assert len(matrix_keys) == 2
    assert "matrix_0_4" in matrix_keys
    assert "matrix_5_10" in matrix_keys

    # Check regular parameter
    assert "regular_param" in learnable_keys

    # Total learnable parameters: 3 + 4 + 5 + 2 + 1 = 15 distinct parameter names
    assert len(learnable_keys) == 15


def test_get_method_fix_parameters():
    """Test get method for fixed parameters."""
    params = [
        AcadosParameter(name="fixed_param", default=np.array([42.0]), interface="fix"),
    ]

    manager = AcadosParameterManager(params, N_horizon=5)

    # Should return the parameter value directly
    result = manager.get("fixed_param")
    np.testing.assert_array_equal(result, np.array([42.0]))


def test_get_method_learnable_parameters():
    """Test get method for learnable parameters."""
    params = [
        AcadosParameter(name="learnable_param", default=np.array([1.0]), interface="learnable"),
    ]

    manager = AcadosParameterManager(params, N_horizon=5)

    # Should return the symbolic variable
    result = manager.get("learnable_param")

    # Check that result has type ca.SX and shape (1,1) and that its name is "learnable_param"
    assert isinstance(result, ca.SX)
    assert result.shape == (1, 1)
    assert result.str() == "learnable_param"


def test_get_method_non_learnable_parameters():
    """Test get method for non-learnable parameters."""
    params = [
        AcadosParameter(
            name="non_learnable_param",
            default=np.array([1.0]),
            interface="non-learnable",
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=5)

    # Should return the symbolic variable
    result = manager.get("non_learnable_param")

    # Check that result has type ca.SX and shape (1,1) and that its name is "learnable_param"
    assert isinstance(result, ca.SX)
    assert result.shape == (1, 1)
    assert result.str() == "non_learnable_param"


def test_get_method_vary_stages():
    """Test get method for parameters with vary_stages."""
    N_horizon = 5
    params = [
        AcadosParameter(
            name="staged_param",
            default=np.array([1.0]),
            interface="learnable",
            end_stages=[3, N_horizon],
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=N_horizon)

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
    params = [
        AcadosParameter(name="existing_param", default=np.array([1.0]), interface="fix"),
    ]

    manager = AcadosParameterManager(params, N_horizon=5)

    with pytest.raises(ValueError, match="Unknown name: nonexistent"):
        manager.get("nonexistent")


def test_empty_parameter_list():
    """Test AcadosParamManager with empty parameter list."""
    params = []

    with pytest.warns(UserWarning, match="Empty parameter list provided to AcadosParamManager"):
        manager = AcadosParameterManager(params, N_horizon=5)

    assert len(manager.parameters) == 0
    assert len(manager.learnable_parameters.keys()) == 0
    assert len(manager.non_learnable_parameters.keys()) == 0


def test_parameter_name_with_underscores():
    """Test parameters with underscores in their names.

    Test is due to a potential conflict with template for stages: {name}_{start}_{end}).
    """
    N_horizon = 5
    params = [
        AcadosParameter(
            name="param_with_underscores",
            default=np.array([1.0]),
            interface="learnable",
            end_stages=[3, N_horizon],
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=N_horizon)

    # Should properly handle names with underscores
    learnable_keys = list(manager.learnable_parameters.keys())
    staged_keys = [k for k in learnable_keys if k.startswith("param_with_underscores_")]

    assert len(staged_keys) == 2
    assert "param_with_underscores_0_3" in staged_keys
    assert "param_with_underscores_4_5" in staged_keys

    # Values should be set correctly (CasADi format)
    for key in staged_keys:
        np.testing.assert_array_equal(manager.learnable_parameters_default[key], np.array([[1.0]]))


def test_large_dimension_parameters():
    """Test that CasADi limitation with >2D arrays is handled gracefully."""
    # CasADi only supports up to 2D arrays, test that 2D arrays are accepted and work as expected.
    params_2d = [
        AcadosParameter(
            name="matrix_param",
            default=np.array([[1.0, 2.0], [3.0, 4.0]]),
            interface="learnable",
            space=gym.spaces.Box(
                low=np.array([[0.0, 0.0], [0.0, 0.0]]),
                high=np.array([[10.0, 10.0], [10.0, 10.0]]),
            ),
        ),
    ]

    manager = AcadosParameterManager(params_2d, N_horizon=5)

    # Should handle 2D arrays correctly (flattened in CasADi)
    assert "matrix_param" in manager.learnable_parameters.keys()

    # CasADi preserves matrix shapes
    expected_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_lb = np.array([[0.0, 0.0], [0.0, 0.0]])
    expected_ub = np.array([[10.0, 10.0], [10.0, 10.0]])

    np.testing.assert_array_equal(
        manager.learnable_parameters_default["matrix_param"], expected_value
    )
    np.testing.assert_array_equal(manager.learnable_parameters_lb["matrix_param"], expected_lb)
    np.testing.assert_array_equal(manager.learnable_parameters_ub["matrix_param"], expected_ub)

    # Test that 3D arrays raise an error
    params_3d = [
        AcadosParameter(
            name="tensor_param",
            default=np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            interface="learnable",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Parameter 'tensor_param' has 3 dimensions."
        "*CasADi only supports arrays up to 2 dimensions",
    ):
        AcadosParameterManager(params_3d, N_horizon=5)

    # Test that 3D space bounds raise an error
    params_3d_bounds = [
        AcadosParameter(
            name="tensor_bounds_param",
            default=np.array([[1.0, 2.0], [3.0, 4.0]]),
            space=gym.spaces.Box(
                low=np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
                high=np.array([[[10.0, 10.0], [10.0, 10.0]], [[10.0, 10.0], [10.0, 10.0]]]),
            ),
            interface="learnable",
        ),
    ]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Parameter 'tensor_bounds_param' space has 3 dimensions, but CasADi only"
            " supports arrays up to 2 dimensions. Space shape: (2, 2, 2)"
        ),
    ):
        AcadosParameterManager(params_3d_bounds, N_horizon=5)


def test_combine_parameter_values():
    """Test combining non-learnable parameter values across multiple batches and time stages.

    Verifies that AcadosParameterManager.combine_non_learnable_parameter_values()
    correctly combines parameter values into a (batch_size, N_horizon+1, param_dim) array.
    """
    params = [
        AcadosParameter(name="test_param", default=np.array([1.0]), interface="non-learnable"),
    ]

    manager = AcadosParameterManager(params, N_horizon=5)

    expected = np.ones((2, 6, 1))
    result = manager.combine_non_learnable_parameter_values(batch_size=2)
    np.testing.assert_array_equal(result, expected)


def test_combine_parameter_values_complex():
    """Test combine_parameter_values with mixed parameter types, interfaces, and vary_stages."""
    N_horizon = 8
    params = [
        # Scalar parameters
        AcadosParameter(name="scalar_fix", default=np.array([2.0]), interface="fix"),
        AcadosParameter(name="scalar_learnable", default=np.array([3.0]), interface="learnable"),
        AcadosParameter(
            name="scalar_non_learnable",
            default=np.array([4.0]),
            interface="non-learnable",
        ),
        AcadosParameter(
            name="scalar_staged",
            default=np.array([5.0]),
            interface="learnable",
            end_stages=[2, 6, N_horizon],
        ),
        # Vector parameters
        AcadosParameter(name="vector_fix", default=np.array([1.0, 2.0]), interface="fix"),
        AcadosParameter(
            name="vector_learnable",
            default=np.array([6.0, 7.0]),
            interface="learnable",
        ),
        AcadosParameter(
            name="vector_non_learnable",
            default=np.array([8.0, 9.0]),
            interface="non-learnable",
        ),
        AcadosParameter(
            name="vector_staged",
            default=np.array([10.0, 11.0]),
            interface="learnable",
            end_stages=[3, N_horizon],
        ),
        # Matrix parameters
        AcadosParameter(
            name="matrix_fix",
            default=np.array([[1.0, 2.0], [3.0, 4.0]]),
            interface="fix",
        ),
        AcadosParameter(
            name="matrix_learnable",
            default=np.array([[12.0, 13.0], [14.0, 15.0]]),
            interface="learnable",
        ),
        AcadosParameter(
            name="matrix_non_learnable",
            default=np.array([[16.0, 17.0], [18.0, 19.0]]),
            interface="non-learnable",
        ),
        AcadosParameter(
            name="matrix_staged",
            default=np.array([[20.0, 21.0], [22.0, 23.0]]),
            interface="learnable",
            end_stages=[1, 4, 7],
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=8)

    # Test with batch_size=3
    batch_size = 3
    result = manager.combine_non_learnable_parameter_values(batch_size=batch_size)

    # Verify result shape: (batch_size, N_horizon + 1, total_non_learnable_params)
    # Non-learnable params: scalar_non_learnable(1) + vector_non_learnable(2) +
    # matrix_non_learnable(4) + indicator(9) = 16
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
            np.testing.assert_array_equal(result[batch_idx, stage_idx, 3:7], expected_matrix_flat)

            # Check indicator values (last 9 elements for N_horizon=8)
            # indicator[stage_idx] should be 1.0, others should be 0.0
            expected_indicator = np.zeros(9)
            expected_indicator[stage_idx] = 1.0
            np.testing.assert_array_equal(result[batch_idx, stage_idx, 7:], expected_indicator)

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

    # Verify the result shape remains the same
    assert result.shape == expected_shape

    # Verify that the overwritten parameters are correctly incorporated
    for batch_idx in range(batch_size):
        for stage_idx in range(manager.N_horizon + 1):
            # scalar_non_learnable should still be the default value (not overwritten)
            assert result[batch_idx, stage_idx, 0] == 4.0

            # vector_non_learnable should use the random overwrite values
            np.testing.assert_array_equal(
                result[batch_idx, stage_idx, 1:3],
                vector_non_learnable[batch_idx, stage_idx, :],
            )

            # matrix_non_learnable should use the random overwrite values (flattened)
            # Note: overwrite values use C-order flattening, unlike default values which use F-order
            expected_matrix_flat = matrix_non_learnable[batch_idx, stage_idx, :, :].flatten(
                order="C"
            )
            np.testing.assert_array_equal(result[batch_idx, stage_idx, 3:7], expected_matrix_flat)

            # indicator values should remain unchanged
            expected_indicator = np.zeros(9)
            expected_indicator[stage_idx] = 1.0
            np.testing.assert_array_equal(result[batch_idx, stage_idx, 7:], expected_indicator)


def test_param_manager_combine_parameter_values(
    acados_test_ocp_with_stagewise_varying_params: AcadosOcp,
    nominal_stagewise_params: tuple[AcadosParameter, ...],
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

    acados_param_manager = AcadosParameterManager(
        parameters=nominal_stagewise_params,
        N_horizon=N_horizon,
    )

    keys = [
        key
        for key in list(acados_param_manager.non_learnable_parameters_default.keys())
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
                acados_param_manager.non_learnable_parameters_default[key].shape[0],
            )
        )

    res = acados_param_manager.combine_non_learnable_parameter_values(**overwrite)

    assert res.shape == (
        batch_size,
        N_horizon + 1,
        acados_param_manager.non_learnable_parameters_default.cat.shape[0],
    ), "The shape of the combined parameter values does not match the expected shape."

    # Verify that the overwritten parameter values are correctly incorporated
    param_start_idx = 0
    for key in keys:
        param_dim = acados_param_manager.non_learnable_parameters_default[key].shape[0]
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
    diff_mpc: AcadosDiffMpcTorch,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpcTorch,
    nominal_stagewise_params: tuple[AcadosParameter, ...],
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

    N_horizon = (
        mpc["global"]
        .diff_mpc_fun.forward_batch_solver.ocp_solvers[0]
        .acados_ocp.solver_options.N_horizon
    )

    # Create a parameter manager for the stagewise varying parameters.
    parameter_manager = AcadosParameterManager(
        parameters=nominal_stagewise_params,
        N_horizon=N_horizon,
    )
    p_stagewise = parameter_manager.combine_non_learnable_parameter_values()

    x0 = np.array([1.0, 1.0, 0.0, 0.0])

    sol_forward = {}
    sol_forward["global"] = mpc["global"].forward(
        x0=torch.tensor(x0, dtype=torch.float32).reshape(1, -1)
    )
    sol_forward["stagewise"] = mpc["stagewise"].forward(
        x0=torch.tensor(x0, dtype=torch.float32).reshape(1, -1),
        p_stagewise=p_stagewise,
    )

    for key, val in sol_forward.items():
        print(f"sol_forward_{key} u:", val[3])

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
    for interface in ["learnable", "non-learnable"]:
        default_a = np.array([2.0])
        default_b = np.array([3.0, 4.0])
        params = [
            AcadosParameter(name="param_a", default=default_a, interface=interface),
            AcadosParameter(name="param_b", default=default_b, interface=interface),
        ]

        manager = AcadosParameterManager(params, N_horizon=5)

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


def test_combine_default_learnable_parameter_values_basic():
    """Test combine_default_learnable_parameter_values with basic parameters."""
    N_horizon = 5
    params = [
        AcadosParameter(name="scalar", default=np.array([1.0]), interface="learnable"),
        AcadosParameter(name="vector", default=np.array([2.0, 3.0]), interface="learnable"),
    ]

    manager = AcadosParameterManager(params, N_horizon=N_horizon)

    # Test default values without overwrites
    batch_size = 3
    result = manager.combine_default_learnable_parameter_values(batch_size=batch_size)

    # Expected: tiled default values
    default_flat = manager.learnable_parameters_default.cat.full().flatten()
    expected = np.tile(default_flat, (batch_size, 1))

    np.testing.assert_array_equal(result, expected)
    assert result.shape == (batch_size, len(default_flat))


def test_combine_default_learnable_parameter_values_with_overwrites():
    """Test combine_default_learnable_parameter_values with overwrites for non-stagewise params."""
    N_horizon = 5
    params = [
        AcadosParameter(name="scalar", default=np.array([1.0]), interface="learnable"),
        AcadosParameter(name="vector", default=np.array([2.0, 3.0]), interface="learnable"),
    ]

    manager = AcadosParameterManager(params, N_horizon=N_horizon)

    batch_size = 3
    # Overwrite scalar with custom values
    scalar_values = np.array([[10.0], [20.0], [30.0]])

    result = manager.combine_default_learnable_parameter_values(
        batch_size=batch_size, scalar=scalar_values
    )

    # Check that scalar was overwritten
    scalar_idx = manager.learnable_parameters.f["scalar"]
    np.testing.assert_array_equal(result[:, scalar_idx], scalar_values)

    # Check that vector kept default values
    vector_idx = manager.learnable_parameters.f["vector"]
    expected_vector = np.tile([[2.0], [3.0]], (1, batch_size)).T
    np.testing.assert_array_equal(result[:, vector_idx], expected_vector)


def test_combine_default_learnable_parameter_values_stagewise():
    """Test combine_default_learnable_parameter_values with stagewise parameters."""
    N_horizon = 5
    params = [
        AcadosParameter(
            name="temperature",
            default=np.array([20.0]),
            interface="learnable",
            end_stages=[2, N_horizon],
        ),
        AcadosParameter(
            name="price",
            default=np.array([10.0]),
            interface="learnable",
            end_stages=[N_horizon],
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=N_horizon)

    batch_size = 2
    # Provide stage-varying forecasts: shape (batch_size, N_horizon + 1)
    temperature_forecast = np.array(
        [[15.0, 16.0, 17.0, 18.0, 19.0, 20.0], [25.0, 26.0, 27.0, 28.0, 29.0, 30.0]]
    )

    price_forecast = np.array(
        [[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [15.0, 16.0, 17.0, 18.0, 19.0, 20.0]]
    )

    result = manager.combine_default_learnable_parameter_values(
        batch_size=batch_size, temperature=temperature_forecast, price=price_forecast
    )

    # Verify temperature stages
    temp_0_2_idx = manager.learnable_parameters.f["temperature_0_2"]
    temp_3_5_idx = manager.learnable_parameters.f["temperature_3_5"]

    # For batch 0: stages 0-2 should all have first block value, stages 3-5 second block
    np.testing.assert_array_equal(result[0, temp_0_2_idx], temperature_forecast[0, 0])
    np.testing.assert_array_equal(result[0, temp_3_5_idx], temperature_forecast[0, 3])

    # For batch 1
    np.testing.assert_array_equal(result[1, temp_0_2_idx], temperature_forecast[1, 0])
    np.testing.assert_array_equal(result[1, temp_3_5_idx], temperature_forecast[1, 3])

    # Verify price (single stage block 0-5)
    price_0_5_idx = manager.learnable_parameters.f["price_0_5"]
    np.testing.assert_array_equal(result[0, price_0_5_idx], price_forecast[0, 0])
    np.testing.assert_array_equal(result[1, price_0_5_idx], price_forecast[1, 0])


def test_combine_default_learnable_parameter_values_errors():
    """Test error handling in combine_default_learnable_parameter_values."""
    N_horizon = 5
    params = [
        AcadosParameter(name="scalar", default=np.array([1.0]), interface="learnable"),
        AcadosParameter(
            name="temperature",
            default=np.array([20.0]),
            interface="learnable",
            end_stages=[2, N_horizon],
        ),
    ]

    manager = AcadosParameterManager(params, N_horizon=N_horizon)

    # Test error for unknown parameter
    with pytest.raises(ValueError, match="Parameter 'unknown' not found"):
        manager.combine_default_learnable_parameter_values(
            batch_size=2, unknown=np.array([[1.0], [2.0]])
        )

    # Test error for non-learnable parameter
    params_non_learnable = [
        AcadosParameter(name="non_learn", default=np.array([1.0]), interface="non-learnable")
    ]
    manager2 = AcadosParameterManager(params_non_learnable, N_horizon=N_horizon)

    with pytest.raises(ValueError, match="has interface 'non-learnable'"):
        manager2.combine_default_learnable_parameter_values(
            batch_size=2, non_learn=np.array([[1.0], [2.0]])
        )

    # Test error for wrong batch size
    with pytest.raises(ValueError, match="batch_size=2 does not match.*batch_size=3"):
        manager.combine_default_learnable_parameter_values(
            batch_size=2, scalar=np.array([[1.0], [2.0], [3.0]])
        )

    # Test error for wrong shape in stagewise parameter
    with pytest.raises(ValueError, match="requires shape \\(batch_size, 6"):
        manager.combine_default_learnable_parameter_values(
            batch_size=2, temperature=np.array([[1.0], [2.0]])  # Should be (2, 6)
        )


def test_stagewise_solution_matches_global_solver_for_initial_reference_change(
    nominal_stagewise_params: tuple[AcadosParameter, ...],
    acados_test_ocp_no_p_global: AcadosOcp,
    diff_mpc_with_stagewise_varying_params: AcadosDiffMpcTorch,
    diff_mpc: AcadosDiffMpcTorch,
    rng: np.random.Generator,
) -> None:
    """Test stagewise solution matches global solver for initial reference change.

    Test that setting parameters stagewise has the expected effect by comparing it to
    an ocp_solver with global parameters and nonlinear_ls cost.
    """
    global_solver = AcadosOcpSolver(acados_test_ocp_no_p_global)

    ocp = diff_mpc_with_stagewise_varying_params.diff_mpc_fun.ocp
    pm = AcadosParameterManager(
        parameters=nominal_stagewise_params,
        N_horizon=ocp.solver_options.N_horizon,
    )

    p_global_values = pm.learnable_parameters_default
    p_stagewise = pm.combine_non_learnable_parameter_values()

    xref_0 = rng.random(size=4)
    uref_0 = rng.random(size=2)
    yref_0 = np.concatenate((xref_0, uref_0))

    p_global_values["xref_0_0"] = xref_0
    p_global_values["uref_0_0"] = uref_0

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

    p_global = p_global_values.cat.full().flatten().reshape(1, ocp.dims.np_global)
    x0 = torch.tensor(x0, dtype=torch.float32).reshape(1, -1)

    sol_pert = diff_mpc_with_stagewise_varying_params.forward(
        x0=x0, p_global=p_global, p_stagewise=p_stagewise
    )

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
        "The control trajectory matches between nominal and stagewise diff MPC \
            despite different initial reference."
    )

    assert not np.allclose(
        x_stagewise_nom,
        x_stagewise,
        atol=1e-3,
        rtol=1e-3,
    ), (
        "The state trajectory matches between nominal and stagewise diff MPC \
            despite different initial reference."
    )
