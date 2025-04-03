import numpy as np
import pytest
import torch
from leap_c.examples.pendulum_on_a_cart.mpc import PendulumOnCartMPC
from leap_c.examples.pointmass.mpc import PointMassMPC
from leap_c.mpc import MpcInput, MpcParameter
from leap_c.nn.modules import CleanseAndReducePerSampleLoss, MpcSolutionModule


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
    learnable_pendulum_on_cart_mpc: PendulumOnCartMPC,
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
    learnable_pendulum_on_cart_mpc_ext_cost: PendulumOnCartMPC,
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


def test_CleanseAndReduce():
    cleansed_loss = CleanseAndReducePerSampleLoss(
        reduction="mean",
        num_batch_dimensions=1,
        n_nonconvergences_allowed=2,
        throw_exception_if_exceeded=False,
    )

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    status = torch.tensor([[0], [1], [0], [0]], dtype=torch.int8)
    loss, _ = cleansed_loss(x, status)
    assert loss.item() == 8 / 3

    status = torch.tensor([[0], [0], [0], [0]], dtype=torch.int8)
    loss, _ = cleansed_loss(x, status)
    assert loss.item() == 10 / 4

    status = torch.tensor([[2], [0], [1], [0]], dtype=torch.int8)
    loss, _ = cleansed_loss(x, status)
    assert loss.item() == 6 / 2

    status = torch.tensor([[2], [2], [1], [0]], dtype=torch.int8)
    loss, _ = cleansed_loss(x, status)
    assert loss.item() == 0.0

    status = torch.tensor([[1], [0], [1]], dtype=torch.int8)
    try:
        loss, _ = cleansed_loss(x, status)
        assert False
    except ValueError:
        assert True


def test_CleanseAndReduceMultipleBatchAndSampleDims():
    cleansed_loss = CleanseAndReducePerSampleLoss(
        reduction="mean",
        num_batch_dimensions=2,
        n_nonconvergences_allowed=4,
        throw_exception_if_exceeded=False,
    )

    x = torch.ones((3, 3, 3, 3))
    x[0, 0] = 2
    x[0, 1] = 100
    status = torch.zeros((3, 3, 1), dtype=torch.int8)
    status[0, 0] = 1
    status[0, 1] = 2
    loss, _ = cleansed_loss(x, status)
    assert loss.item() == 1.0

    x = torch.ones((3, 3, 3, 3))
    status = torch.zeros((3, 3, 1), dtype=torch.int8)
    status[0, 0] = 1
    status[0, 1] = 2
    status[0, 2] = 4
    status[1, 2] = 5
    loss, _ = cleansed_loss(x, status)
    assert loss.item() == 1.0

    status = torch.zeros((3, 3, 1), dtype=torch.int8)
    status[0, 0] = 1
    status[0, 1] = 2
    status[0, 2] = 1
    status[1, 2] = 2
    status[2, 2] = 2
    loss, _ = cleansed_loss(x, status)
    assert loss.item() == 0.0

    status = torch.zeros((3, 3, 3, 1), dtype=torch.int8)
    try:
        loss, _ = cleansed_loss(x, status)
        assert False
    except ValueError:
        assert True

    status = torch.zeros((3, 1), dtype=torch.int8)
    try:
        loss, _ = cleansed_loss(x, status)
        assert False
    except ValueError:
        assert True

    status = torch.ones((3, 3, 1), dtype=torch.int8)
    status[0, 0] = 1
    status[0, 1] = 2
    status[0, 2] = 1
    status[1, 2] = 2
    status[2, 2] = 2
    cleansed_loss = CleanseAndReducePerSampleLoss(
        reduction="mean",
        num_batch_dimensions=2,
        n_nonconvergences_allowed=4,
        throw_exception_if_exceeded=True,
    )
    try:
        loss, _ = cleansed_loss(x, status)
        assert False
    except ValueError:
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
