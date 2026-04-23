import numpy as np

from leap_c.ocp.acados.diff_ocp import AcadosDiffOcp


def test_register_parameter_returns_symbol():
    """Test that register_param returns a CasADi symbolic expression and registers parameters."""
    diff_ocp = AcadosDiffOcp(N_horizon=5)
    non_learnable = diff_ocp.register_param(
        name="non_learnable",
        default=np.array([1.0, 2.0]),
        differentiable=False,
    )

    # Non-learnable should appear in non_learnable_parameter_store with original name
    assert len(diff_ocp.parameter_manager._non_learnable_parameter_store.symbols) == 1
    assert "non_learnable" in diff_ocp.parameter_manager._non_learnable_parameter_store.symbols
    assert (
        non_learnable
        is diff_ocp.parameter_manager._non_learnable_parameter_store.symbols["non_learnable"]
    )

    learnable = diff_ocp.register_param(
        name="learnable",
        default=np.array([3.0, 4.0]),
        differentiable=True,
    )

    # Learnable should appear in learnable_parameter_store with original name
    assert len(diff_ocp.parameter_manager._learnable_parameter_store.symbols) == 1
    assert "learnable" in diff_ocp.parameter_manager._learnable_parameter_store.symbols
    assert learnable is diff_ocp.parameter_manager._learnable_parameter_store.symbols["learnable"]

    diff_ocp.register_param(
        name="learnable_stagewise",
        default=np.array([5.0, 6.0]),
        differentiable=True,
        end_stages=[2, 4],
    )

    assert len(diff_ocp.parameter_manager._non_learnable_parameter_store.symbols) == 2
    assert len(diff_ocp.parameter_manager._learnable_parameter_store.symbols) == 3
    assert (
        "learnable_stagewise_0_2" in diff_ocp.parameter_manager._learnable_parameter_store.symbols
    )
    assert (
        "learnable_stagewise_3_4" in diff_ocp.parameter_manager._learnable_parameter_store.symbols
    )


def test_finalize_assigns_to_ocp():
    """Test that finalize() assigns registered parameters to the OCP."""
    diff_ocp = AcadosDiffOcp(N_horizon=5)
    diff_ocp.register_param(
        name="param1",
        default=np.array([1.0]),
        differentiable=False,
    )
    diff_ocp.register_param(
        name="param2",
        default=np.array([2.0]),
        differentiable=True,
    )

    diff_ocp.finalize()

    # Check non-learnable parameters are assigned to ocp.parameter_values
    assert np.array_equal(diff_ocp.parameter_values, np.array([1.0]))

    # Check learnable parameters are assigned to ocp.p_global_values
    assert np.array_equal(diff_ocp.p_global_values, np.array([2.0]))


def test_register_parameter_after_finalize_raises():
    """Test that registering parameters after finalization raises an error."""
    diff_ocp = AcadosDiffOcp(N_horizon=5)
    diff_ocp.finalize()

    try:
        diff_ocp.register_param(
            name="late_param",
            default=np.array([0.0]),
            differentiable=False,
        )
        assert False, "Expected RuntimeError when registering param after finalize()"
    except RuntimeError as e:
        assert str(e) == "Cannot register params after finalize()."
