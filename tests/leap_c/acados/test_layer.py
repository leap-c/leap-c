import numpy as np
import torch
from leap_c.examples.chain.mpc import ChainMpc
from leap_c.examples.cartpole.mpc import CartPoleMPC
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.ocp.acados.mpc import MpcInput, MpcParameter
from leap_c.ocp.acados.layer import MpcSolutionModule


def test_MPCSolutionModule_on_PointMassMPC(
    learnable_point_mass_mpc_m: PointMassMPC,
    point_mass_mpc_p_global: np.ndarray,
    x0: np.ndarray = np.array([0.5, 0.5, 0.0, 0.0]),
    u0: np.ndarray = np.array([0.5, 0.5]),
):
    batch_size = point_mass_mpc_p_global.shape[0]
    assert batch_size <= 10, "Using batch_sizes too large will make the test very slow."

    varying_params_to_test = [0]
    chosen_samples = []
    for i in range(batch_size):
        vary_idx = varying_params_to_test[i % len(varying_params_to_test)]
        chosen_samples.append(point_mass_mpc_p_global[i, :, vary_idx].squeeze())
    test_param = np.stack(chosen_samples, axis=0)

    if len(varying_params_to_test) == 1:
        test_param = test_param.reshape(-1, 1)
    assert test_param.shape == (batch_size, 1)  # Sanity check

    p_rests = None

    mpc_module = MpcSolutionModule(learnable_point_mass_mpc_m)
    x0_torch = torch.tensor(x0, dtype=torch.float64)
    x0_torch = torch.tile(x0_torch, (batch_size, 1))
    p = torch.tensor(test_param, dtype=torch.float64)
    u0 = torch.tensor(u0, dtype=torch.float64)
    u0 = torch.tile(u0, (batch_size, 1))

    u0.requires_grad = True
    x0_torch.requires_grad = True
    p.requires_grad = True

    def only_du0dx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0,
            u0=None,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.u0, mpc_output.status

    torch.autograd.gradcheck(
        only_du0dx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0,
            u0=None,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.V, mpc_output.status

    torch.autograd.gradcheck(
        only_dVdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  #
        mpc_input = MpcInput(
            x0=x0,
            u0=u0,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(
        only_dQdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_du0dp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.u0, mpc_output.status

    torch.autograd.gradcheck(
        only_du0dp_global, p, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.V, mpc_output.status

    # NOTE: A higher tolerance than in the other checks is used here.
    torch.autograd.gradcheck(
        only_dVdp_global, p, atol=5 * 1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            u0=u0,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(
        only_dQdp_global, p, atol=1e-2, eps=1e-6, raise_exception=True
    )

    def only_dQdu0(u0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            u0=u0,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(only_dQdu0, u0, atol=1e-2, eps=1e-4, raise_exception=True)


def test_MPCSolutionModule_on_PendulumOnCart(
    learnable_pendulum_on_cart_mpc: CartPoleMPC,
    pendulum_on_cart_p_global: np.ndarray,
    x0: np.ndarray = np.array([0.0, (179.0 / 180.0) * np.pi, 0.0, 0.0]),
    u0: np.ndarray = np.array([0.0]),
):
    batch_size = pendulum_on_cart_p_global.shape[0]
    assert batch_size <= 10, "Using batch_sizes too large will make the test very slow."
    assert (
        pendulum_on_cart_p_global.shape[2]
        == learnable_pendulum_on_cart_mpc.default_p_global.shape[0]
    )

    varying_params_to_test = np.arange(
        learnable_pendulum_on_cart_mpc.default_p_global.shape[0]
    )

    chosen_samples = []
    for i in range(batch_size):
        vary_idx = varying_params_to_test[i % len(varying_params_to_test)]
        chosen_samples.append(pendulum_on_cart_p_global[i, :, vary_idx].squeeze())
    test_param = np.stack(chosen_samples, axis=0)

    if len(varying_params_to_test) == 1:
        test_param = test_param.reshape(-1, 1)
    assert test_param.shape == (
        batch_size,
        learnable_pendulum_on_cart_mpc.default_p_global.shape[0],
    )  # Sanity check

    p_rests = None

    mpc_module = MpcSolutionModule(learnable_pendulum_on_cart_mpc)
    x0_torch = torch.tensor(x0, dtype=torch.float64)
    x0_torch = torch.tile(x0_torch, (batch_size, 1))
    p = torch.tensor(test_param, dtype=torch.float64)
    u0 = torch.tensor(u0, dtype=torch.float64)
    u0 = torch.tile(u0, (batch_size, 1))

    u0.requires_grad = True
    x0_torch.requires_grad = True
    p.requires_grad = True

    def only_du0dx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0,
            u0=None,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.u0, mpc_output.status

    torch.autograd.gradcheck(
        only_du0dx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0,
            u0=None,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.V, mpc_output.status

    torch.autograd.gradcheck(
        only_dVdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  #
        mpc_input = MpcInput(
            x0=x0,
            u0=u0,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(
        only_dQdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_du0dp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.u0, mpc_output.status

    torch.autograd.gradcheck(
        only_du0dp_global, p, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        return mpc_output.V, mpc_output.status

    # NOTE: A higher tolerance than in the other checks is used here.
    torch.autograd.gradcheck(
        only_dVdp_global, p, atol=5 * 1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            u0=u0,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(
        only_dQdp_global, p, atol=1e-2, eps=1e-6, raise_exception=True
    )

    def only_dQdu0(u0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            u0=u0,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(only_dQdu0, u0, atol=1e-2, eps=1e-4, raise_exception=True)


def test_MPCSolutionModule_on_PendulumOnCart_ext_cost(
    learnable_pendulum_on_cart_mpc_ext_cost: CartPoleMPC,
    pendulum_on_cart_ext_cost_p_global: np.ndarray,
    x0: np.ndarray = np.array([0.0, (179.0 / 180.0) * np.pi, 0.0, 0.0]),
    u0: np.ndarray = np.array([79.0]),
    # NOTE: u0 is different for easier convergence in the Q cases.
    # It needs to be slightly below 80 because else finite differences would exceed 80
    # which we do not allow in the ocp formulation and we raise an error then
):
    batch_size = pendulum_on_cart_ext_cost_p_global.shape[0]
    assert batch_size <= 10, "Using batch_sizes too large will make the test very slow."
    assert (
        pendulum_on_cart_ext_cost_p_global.shape[2]
        == learnable_pendulum_on_cart_mpc_ext_cost.default_p_global.shape[0]
    )

    varying_params_to_test = np.arange(
        learnable_pendulum_on_cart_mpc_ext_cost.default_p_global.shape[0]
    )

    chosen_samples = []
    for i in range(batch_size):
        vary_idx = varying_params_to_test[i % len(varying_params_to_test)]
        chosen_samples.append(
            pendulum_on_cart_ext_cost_p_global[i, :, vary_idx].squeeze()
        )
    test_param = np.stack(chosen_samples, axis=0)

    if len(varying_params_to_test) == 1:
        test_param = test_param.reshape(-1, 1)
    assert test_param.shape == (
        batch_size,
        learnable_pendulum_on_cart_mpc_ext_cost.default_p_global.shape[0],
    )  # Sanity check

    p_rests = None

    mpc_module = MpcSolutionModule(learnable_pendulum_on_cart_mpc_ext_cost)
    x0_torch = torch.tensor(x0, dtype=torch.float64)
    x0_torch = torch.tile(x0_torch, (batch_size, 1))
    p = torch.tensor(test_param, dtype=torch.float64)
    u0 = torch.tensor(u0, dtype=torch.float64)
    u0 = torch.tile(u0, (batch_size, 1))

    u0.requires_grad = True
    x0_torch.requires_grad = True
    p.requires_grad = True

    def only_du0dx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0,
            u0=None,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.u0, mpc_output.status

    torch.autograd.gradcheck(
        only_du0dx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0,
            u0=None,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.V, mpc_output.status

    torch.autograd.gradcheck(
        only_dVdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  #
        mpc_input = MpcInput(
            x0=x0,
            u0=u0,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(
        only_dQdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_du0dp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.u0, mpc_output.status

    torch.autograd.gradcheck(
        only_du0dp_global, p, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.V, mpc_output.status

    # NOTE: A higher tolerance than in the other checks is used here.
    torch.autograd.gradcheck(
        only_dVdp_global, p, atol=5 * 1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            u0=u0,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(
        only_dQdp_global, p, atol=1e-2, eps=1e-6, raise_exception=True
    )

    def only_dQdu0(u0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            u0=u0,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(only_dQdu0, u0, atol=1e-2, eps=1e-4, raise_exception=True)


def test_MPCSolutionModule_on_ChainCost(
    learnable_chain_cost_mpc: ChainMpc,
    chain_cost_p_global: np.ndarray,
    u0: np.ndarray = np.array([0.0, 0.0, 0.0]),
):
    x0 = learnable_chain_cost_mpc.ocp_solver.acados_ocp.constraints.x0
    batch_size = chain_cost_p_global.shape[0]
    assert batch_size <= 10, "Using batch_sizes too large will make the test very slow."
    assert (
        chain_cost_p_global.shape[2]
        == learnable_chain_cost_mpc.default_p_global.shape[0]
    )

    varying_params_to_test = np.arange(
        learnable_chain_cost_mpc.default_p_global.shape[0]
    )

    chosen_samples = []
    for i in range(batch_size):
        vary_idx = varying_params_to_test[i % len(varying_params_to_test)]
        chosen_samples.append(chain_cost_p_global[i, :, vary_idx].squeeze())
    test_param = np.stack(chosen_samples, axis=0)

    if len(varying_params_to_test) == 1:
        test_param = test_param.reshape(-1, 1)
    assert test_param.shape == (
        batch_size,
        learnable_chain_cost_mpc.default_p_global.shape[0],
    )  # Sanity check

    p_rests = None

    mpc_module = MpcSolutionModule(learnable_chain_cost_mpc)
    x0_torch = torch.tensor(x0, dtype=torch.float64)
    x0_torch = torch.tile(x0_torch, (batch_size, 1))
    p = torch.tensor(test_param, dtype=torch.float64)
    u0 = torch.tensor(u0, dtype=torch.float64)
    u0 = torch.tile(u0, (batch_size, 1))

    u0.requires_grad = True
    x0_torch.requires_grad = True
    p.requires_grad = True

    def only_du0dx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0,
            u0=None,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.u0, mpc_output.status

    torch.autograd.gradcheck(
        only_du0dx0, x0_torch, atol=1e-1, eps=1e-4, raise_exception=True
    )

    def only_dVdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0,
            u0=None,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.V, mpc_output.status

    torch.autograd.gradcheck(
        only_dVdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdx0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  #
        mpc_input = MpcInput(
            x0=x0,
            u0=u0,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(
        only_dQdx0, x0_torch, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_du0dp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.u0, mpc_output.status

    torch.autograd.gradcheck(
        only_du0dp_global, p, atol=1e-2, eps=1e-4, raise_exception=True
    )

    def only_dVdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.V, mpc_output.status

    # NOTE: A higher tolerance than in the other checks is used here.
    torch.autograd.gradcheck(
        only_dVdp_global, p, atol=5 * 1e-2, eps=1e-4, raise_exception=True
    )

    def only_dQdp_global(p_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            u0=u0,
            parameters=MpcParameter(p_global=p_global, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(
        only_dQdp_global, p, atol=1e-2, eps=1e-6, raise_exception=True
    )

    def only_dQdu0(u0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mpc_input = MpcInput(
            x0=x0_torch,
            u0=u0,
            parameters=MpcParameter(p_global=p, p_stagewise=p_rests),
        )
        mpc_output, _, _ = mpc_module.forward(
            mpc_input=mpc_input,
            mpc_state=None,
        )
        assert torch.all(
            torch.isnan(mpc_output.u0)
        ), "u_star should be nan, since u0 is given."
        assert np.all(
            mpc_output.status.cpu().detach().numpy() == 0
        ), "MPC should converge."
        return mpc_output.Q, mpc_output.status

    torch.autograd.gradcheck(only_dQdu0, u0, atol=1e-2, eps=1e-4, raise_exception=True)
