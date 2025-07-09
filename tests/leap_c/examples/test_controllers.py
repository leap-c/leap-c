from leap_c.examples.cartpole.controller import CartPoleController
from leap_c.examples.cartpole.mpc import CartPoleMPC
from leap_c.examples.pointmass.controller import PointMassController
from leap_c.examples.pointmass.mpc import PointMassMpc
from leap_c.examples.chain.controller import ChainController
from leap_c.examples.chain.mpc import ChainMpc
from leap_c.examples.chain.env import ChainEnv
from leap_c.ocp.acados.layer import MpcSolutionModule
from leap_c.ocp.acados.torch import AcadosDiffMpc
import numpy as np
import torch

from leap_c.ocp.acados.mpc import MpcInput, MpcParameter


def get_default_action(mpc_layer, controller, diff_mpc, obs, state, param):

    # MPC layer output
    state_torch = torch.as_tensor(state, dtype=torch.float32)
    param_torch = torch.as_tensor(param, dtype=torch.float32)
    mpc_param = MpcParameter(p_global=param_torch)  # type: ignore
    mpc_input = MpcInput(x0=state_torch, parameters=mpc_param)
    mpc_output, mpc_state, mpc_stats = mpc_layer(mpc_input)
    mpc_action = mpc_output.u0.detach().cpu().numpy()

    # Diff MPC output
    _, diff_mpc_action, *_ = diff_mpc(x0=state_torch, p_global=param_torch, ctx=None)


    # Controller output
    ctx, ctrl_action = controller(obs, param)

    return mpc_action, ctrl_action.numpy(), diff_mpc_action.numpy()


def test_pointmass_controller_matches_mpc():
    learnable_params = ["q_diag", "xref", "uref"]

    mpc = PointMassMpc(
        learnable_params=learnable_params,
    )
    mpc_layer = MpcSolutionModule(mpc)

    controller = PointMassController()
    
    diff_mpc = AcadosDiffMpc(mpc.ocp)


    obs = np.array(
        [
            [0.2, 0.2, 0.0, 0.0, 0, 0.0],
            [0.3, 1.3, 0.2, 0.0, 0, 0.0],
            [0.9, 4.2, 0.0, 0.1, 0, 0.0],
        ]
    )
    state = obs[:, :4]  # Extract the state part of the observation
    epsilon = 1e-5
    low_param = controller.param_space.low + epsilon  # type: ignore
    high_param = controller.param_space.high - epsilon  # type: ignore
    mid_param = (low_param + high_param) / 2
    param = np.stack([low_param, mid_param, high_param], axis=0)

    # Get default action from both MPC layer and controller
    mpc_layer, ctrl_action, diff_mpc = get_default_action(mpc_layer, controller, diff_mpc, obs, state, param)

    # Compare outputs
    np.testing.assert_allclose(mpc_layer, diff_mpc, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(mpc_layer, ctrl_action, rtol=1e-3, atol=1e-3)


def test_cartpole_controller_matches_mpc():
    learnable_params = ["xref2"]

    mpc = CartPoleMPC(
        learnable_params=learnable_params,
    )
    mpc_layer = MpcSolutionModule(mpc)

    controller = CartPoleController()
    
    diff_mpc = AcadosDiffMpc(mpc.ocp)

    # CartPole observations: [x, theta, x_dot, theta_dot]
    obs = np.array(
        [
            [0.0, np.pi + 0.1, 0.0, 0.0],  # Near upright position
            [0.5, np.pi - 0.2, 0.1, 0.0],  # Slightly off-center
            [-0.3, np.pi + 0.3, -0.1, 0.1],  # Different starting position
        ]
    )
    state = obs  # For cartpole, obs is the state
    epsilon = 1e-5
    low_param = controller.param_space.low + epsilon  # type: ignore
    high_param = controller.param_space.high - epsilon  # type: ignore
    mid_param = (low_param + high_param) / 2
    param = np.stack([low_param, mid_param, high_param], axis=0)

    # Get default action from both MPC layer and controller
    mpc_layer, ctrl_action, diff_mpc = get_default_action(mpc_layer, controller, diff_mpc, obs, state, param)

    # Compare outputs
    np.testing.assert_allclose(mpc_layer, diff_mpc, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(mpc_layer, ctrl_action, rtol=1e-3, atol=1e-3)


# def test_chain_controller_matches_mpc():
#     learnable_params = ["m", "D", "L", "C", "w"]
#     n_mass = 4
# 
#     mpc = ChainMpc(learnable_params=learnable_params, n_mass=n_mass)
#     mpc_layer = MpcSolutionModule(mpc)
#     controller = ChainController()
#     diff_mpc = AcadosDiffMpc(mpc.ocp)
# 
#     env = ChainEnv()
#     obs, _ = env.reset()
# 
#     obs = np.array(obs).reshape(1, -1)  # Reshape to match expected input shape
#     state = obs
#     param = controller.default_param.reshape(1, -1)  # Default parameters
# 
# 
#     # Get default action from both MPC layer and controller
#     mpc_layer, ctrl_action, diff_mpc = get_default_action(mpc_layer, controller, diff_mpc, obs, state, param)
# 
#     # TODO: tolerance was increased due to numerical errors, possibly need to investigate
#     # Compare outputs
#     np.testing.assert_allclose(mpc_layer, diff_mpc, rtol=1e-3, atol=1e-3)
#     np.testing.assert_allclose(mpc_layer, ctrl_action, rtol=1e-3, atol=1e-3)



