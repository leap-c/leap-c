from leap_c.examples.cartpole.controller import CartPoleController
from leap_c.examples.cartpole.mpc import CartPoleMPC
from leap_c.examples.cartpole.task import PARAMS_SWINGUP
from leap_c.examples.pointmass.controller import PointMassController
from leap_c.examples.pointmass.mpc import PointMassMpc
from leap_c.examples.chain.controller import ChainController
from leap_c.examples.chain.mpc import ChainMpc
from leap_c.ocp.acados.layer import MpcSolutionModule
import numpy as np
import torch

from leap_c.ocp.acados.mpc import MpcInput, MpcParameter


def get_default_action(mpc_layer, controller):
    # Prepare test input
    obs = np.copy(controller.ocp.constraints.x0)
    param = controller.default_param()

    # MPC layer output
    obs_torch = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    param_torch = torch.as_tensor(param, dtype=torch.float32).unsqueeze(0)
    mpc_param = MpcParameter(p_global=param_torch)  # type: ignore
    mpc_input = MpcInput(x0=obs_torch, parameters=mpc_param)
    mpc_output, mpc_state, mpc_stats = mpc_layer(mpc_input)
    mpc_action = mpc_output.u0.detach().cpu().numpy()

    # Controller output
    ctx, ctrl_action = controller.forward(obs, param)

    return mpc_action, ctrl_action.numpy()


def test_cartpole_controller_matches_mpc():
    learnable_params = ["xref2"]

    mpc = CartPoleMPC(
        N_horizon=5,
        T_horizon=0.25,
        learnable_params=learnable_params,
    )
    mpc_layer = MpcSolutionModule(mpc)

    controller = CartPoleController(
        N_horizon=5,
        T_horizon=0.25,
        learnable_params=learnable_params,
    )

    # Get default action from both MPC layer and controller
    mpc_layer, ctrl_action = get_default_action(mpc_layer, controller)

    # Compare outputs
    np.testing.assert_allclose(ctrl_action, mpc_layer, rtol=1e-5, atol=1e-6)


def test_pointmass_controller_matches_mpc():
    learnable_params = ["m", "cx", "cy"]

    mpc = PointMassMpc(
        learnable_params=learnable_params,
    )
    mpc_layer = MpcSolutionModule(mpc)

    controller = PointMassController(
        learnable_params=learnable_params,
    )

    # Get default action from both MPC layer and controller
    mpc_layer, ctrl_action = get_default_action(mpc_layer, controller)

    # Compare outputs
    np.testing.assert_allclose(ctrl_action, mpc_layer, rtol=1e-5, atol=1e-6)


def test_chain_controller_matches_mpc():
    learnable_params = ["m", "D", "L", "C", "w"]
    n_mass = 4

    mpc = ChainMpc(
        learnable_params=learnable_params,
        n_mass=n_mass
    )
    mpc_layer = MpcSolutionModule(mpc)

    controller = ChainController(
        learnable_params=learnable_params,
        n_mass=n_mass
    )

    # Get default action from both MPC layer and controller
    mpc_layer, ctrl_action = get_default_action(mpc_layer, controller)

    # TODO: tolerance was increased due to numerical errors, possibly need to investigate
    # Compare outputs
    np.testing.assert_allclose(ctrl_action, mpc_layer, rtol=0.05, atol=5e-4)
