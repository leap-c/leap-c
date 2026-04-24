import numpy as np

from leap_c.ocp.acados.diff_ocp import AcadosDiffOcp


def test_register_parameter_returns_symbol():
    """Test that register_param returns a CasADi symbolic expression and registers parameters."""
    N_HORIZON = 5

    diff_ocp = AcadosDiffOcp(N_horizon=N_HORIZON)
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
    # test parameter_values being updated automatically with the new non-learnable parameter
    assert np.allclose(diff_ocp.parameter_values, np.array([1.0, 2.0]))

    learnable = diff_ocp.register_param(
        name="learnable",
        default=np.array([3.0, 4.0]),
        differentiable=True,
    )

    # Learnable should appear in learnable_parameter_store with original name
    assert len(diff_ocp.parameter_manager._learnable_parameter_store.symbols) == 1
    assert "learnable" in diff_ocp.parameter_manager._learnable_parameter_store.symbols
    assert learnable is diff_ocp.parameter_manager._learnable_parameter_store.symbols["learnable"]
    # test p_global_values being updated automatically with the new learnable parameter
    assert np.allclose(diff_ocp.p_global_values, np.array([3.0, 4.0]))

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
    # test p_global_values being updated automatically with the new learnable parameters
    assert np.allclose(
        diff_ocp.p_global_values,
        np.array([3.0, 4.0, 5.0, 6.0, 5.0, 6.0]),
    )
    # test indicator being added to parameter_values automatically
    assert diff_ocp.parameter_values.shape == (non_learnable.shape[0] + N_HORIZON + 1,)
