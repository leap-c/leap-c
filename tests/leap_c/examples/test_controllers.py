from leap_c.examples.cartpole.controller import CartPoleController
from leap_c.examples.cartpole.mpc import CartPoleMPC
from leap_c.examples.cartpole.task import PARAMS_SWINGUP, CartPoleSwingup
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
    task = CartPoleSwingup()
    mpc_layer = task.mpc
    mpc = mpc_layer.mpc  # type: ignore

    controller = CartPoleController()
    
    diff_mpc = AcadosDiffMpc(mpc.ocp)  # type: ignore

    # Print OCP information for debugging
    print_acados_ocp(mpc.ocp, "CartPoleMPC")
    print_acados_ocp(controller.ocp, "CartPoleController")

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

    print_acados_ocp(mpc.ocp, "CartPole MPC")
    print_acados_ocp(controller.ocp, "CartPole Controller")

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

def print_acados_ocp(ocp, name="AcadosOcp"):
    """
    Print the most important aspects of an AcadosOcp object for debugging.
    
    Args:
        ocp: AcadosOcp object to inspect
        name: Name identifier for the OCP (for logging purposes)
    """
    print(f"\n=== {name} Info ===")
    
    # Basic dimensions
    print(f"Dimensions: nx={ocp.dims.nx}, nu={ocp.dims.nu}")
    print(f"Horizon: N={ocp.solver_options.N_horizon}, tf={ocp.solver_options.tf}")
    
    # Parameters
    print("\nParameters:")
    if hasattr(ocp.model, 'p') and ocp.model.p is not None:
        print(f"  model.p: {ocp.model.p}")
        if hasattr(ocp, 'parameter_values') and ocp.parameter_values is not None:
            print(f"  parameter_values: {ocp.parameter_values}")
    else:
        print("  model.p: None")
    
    if hasattr(ocp.model, 'p_global') and ocp.model.p_global is not None:
        print(f"  model.p_global: {ocp.model.p_global}")
        if hasattr(ocp, 'p_global_values') and ocp.p_global_values is not None:
            print(f"  p_global_values: {ocp.p_global_values}")
    else:
        print("  model.p_global: None")
    
    # Cost information
    print("\nCost:")
    print(f"  cost_type: {getattr(ocp.cost, 'cost_type', 'None')}")
    print(f"  cost_type_e: {getattr(ocp.cost, 'cost_type_e', 'None')}")
    
    # Cost matrices and references
    if hasattr(ocp.cost, 'W') and ocp.cost.W is not None:
        print(f"  W shape: {ocp.cost.W.shape if hasattr(ocp.cost.W, 'shape') else 'CasADi expression'}")
        if hasattr(ocp.cost.W, 'shape') and ocp.cost.W.shape[0] <= 10:  # Only print small matrices
            print(f"  W:\n{ocp.cost.W}")
    
    if hasattr(ocp.cost, 'W_e') and ocp.cost.W_e is not None:
        print(f"  W_e shape: {ocp.cost.W_e.shape if hasattr(ocp.cost.W_e, 'shape') else 'CasADi expression'}")
        if hasattr(ocp.cost.W_e, 'shape') and ocp.cost.W_e.shape[0] <= 10:
            print(f"  W_e:\n{ocp.cost.W_e}")
    
    if hasattr(ocp.cost, 'yref') and ocp.cost.yref is not None:
        print(f"  yref: {ocp.cost.yref}")
    
    if hasattr(ocp.cost, 'yref_e') and ocp.cost.yref_e is not None:
        print(f"  yref_e: {ocp.cost.yref_e}")
    
    # Cost expressions
    if hasattr(ocp.model, 'cost_y_expr') and ocp.model.cost_y_expr is not None:
        print(f"  cost_y_expr: {ocp.model.cost_y_expr}")
    
    if hasattr(ocp.model, 'cost_y_expr_e') and ocp.model.cost_y_expr_e is not None:
        print(f"  cost_y_expr_e: {ocp.model.cost_y_expr_e}")
    
    # External cost expressions
    if hasattr(ocp.model, 'cost_expr_ext_cost') and ocp.model.cost_expr_ext_cost is not None:
        print("  cost_expr_ext_cost: Present")
    
    if hasattr(ocp.model, 'cost_expr_ext_cost_e') and ocp.model.cost_expr_ext_cost_e is not None:
        print("  cost_expr_ext_cost_e: Present")
    
    # Constraints
    print("\nConstraints:")
    if hasattr(ocp.constraints, 'x0') and ocp.constraints.x0 is not None:
        print(f"  x0: {ocp.constraints.x0}")
    
    if hasattr(ocp.constraints, 'lbu') and ocp.constraints.lbu is not None:
        print(f"  lbu: {ocp.constraints.lbu}")
        print(f"  ubu: {ocp.constraints.ubu}")
    
    if hasattr(ocp.constraints, 'lbx') and ocp.constraints.lbx is not None:
        print(f"  lbx: {ocp.constraints.lbx}")
        print(f"  ubx: {ocp.constraints.ubx}")
    
    # Dynamics
    print("\nDynamics:")
    if hasattr(ocp.model, 'disc_dyn_expr') and ocp.model.disc_dyn_expr is not None:
        print("  disc_dyn_expr: Present")
    elif hasattr(ocp.model, 'f_expl_expr') and ocp.model.f_expl_expr is not None:
        print("  f_expl_expr: Present")
    elif hasattr(ocp.model, 'f_impl_expr') and ocp.model.f_impl_expr is not None:
        print("  f_impl_expr: Present")
    else:
        print("  No dynamics expressions found")
    
    # Solver options
    print("\nSolver Options:")
    print(f"  integrator_type: {ocp.solver_options.integrator_type}")
    print(f"  nlp_solver_type: {ocp.solver_options.nlp_solver_type}")
    print(f"  qp_solver: {ocp.solver_options.qp_solver}")
    print(f"  hessian_approx: {ocp.solver_options.hessian_approx}")
    
    print(f"=== End {name} Info ===\n")



