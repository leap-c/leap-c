from dataclasses import fields
from typing import Any, NamedTuple

import numpy as np
import torch
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
)
from leap_c.ocp.acados.data import (
    AcadosOcpSolverInput,
    collate_acados_ocp_solver_input,
    collate_acados_flattened_iterate_fn,
)
from leap_c.torch.rl.buffer import ReplayBuffer


class ForNesting(NamedTuple):
    a: Any
    b: Any


def test_sample_collation_and_dtype_and_device():
    # NOTE: Only tests moving to devices if cuda is available, i.e., github probably won't test it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    collate_fn_map = {
        AcadosOcpSolverInput: collate_acados_ocp_solver_input,
        AcadosOcpFlattenedIterate: collate_acados_flattened_iterate_fn,
    }

    buffer = ReplayBuffer(
        buffer_limit=10,
        device=device,
        tensor_dtype=dtype,
        collate_fn_map=collate_fn_map,
    )
    data_one = (
        1,
        np.array([2], dtype=np.float64),
        torch.tensor([3], device="cpu", dtype=torch.float64),
        AcadosOcpSolverInput(
            x0=np.array([[1, 2, 3]], dtype=np.float32),
            p_global=np.array([1, 2, 3], dtype=np.float32),
            p_stagewise=np.ones((2, 2), dtype=np.float32),
            p_stagewise_sparse_idx=None,
        ),
        ForNesting(a=np.array([1, 2, 3], dtype=np.float64), b=True),
        AcadosOcpFlattenedIterate(
            x=np.ones(1, dtype=np.float64),
            u=np.ones(2, dtype=np.float64),
            z=np.ones(3, dtype=np.float64),
            sl=np.ones(4, dtype=np.float64),
            su=np.ones(5, dtype=np.float64),
            pi=np.ones(6, dtype=np.float64),
            lam=np.ones(7, dtype=np.float64),
        ),
    )
    data_two = (
        1,
        np.array([2], dtype=np.float64),
        torch.tensor([3], device="cpu", dtype=torch.float64),
        AcadosOcpSolverInput(
            x0=np.array([[1, 2, 3]], dtype=np.float32),
            p_global=np.array([1, 2, 3], dtype=np.float32),
            p_stagewise=np.ones((2, 2), dtype=np.float32),
            p_stagewise_sparse_idx=None,
        ),
        ForNesting(a=np.array([1, 2, 3], dtype=np.float64), b=True),
        AcadosOcpFlattenedIterate(
            x=np.ones(1, dtype=np.float64),
            u=np.ones(2, dtype=np.float64),
            z=np.ones(3, dtype=np.float64),
            sl=np.ones(4, dtype=np.float64),
            su=np.ones(5, dtype=np.float64),
            pi=np.ones(6, dtype=np.float64),
            lam=np.ones(7, dtype=np.float64),
        ),
    )
    buffer.put(data_one)
    buffer.put(data_two)
    batch = buffer.sample(2)
    torch.testing.assert_close(
        batch[0], torch.tensor([1, 1], device=device, dtype=dtype)
    )
    torch.testing.assert_close(
        batch[1], torch.tensor([[2], [2]], device=device, dtype=dtype)
    )
    torch.testing.assert_close(
        batch[2], torch.tensor([[3], [3]], device=device, dtype=dtype)
    )

    test_param = AcadosOcpSolverInput(
        x0=np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32),
        p_global=np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32),
        p_stagewise=np.ones((2, 2, 2), dtype=np.float32),
        p_stagewise_sparse_idx=None,
    )
    assert (
        np.allclose(batch[3].p_global, test_param.p_global)  # type:ignore
        and batch[3].p_global.dtype == test_param.p_global.dtype  # type:ignore
    )
    assert (
        np.allclose(batch[3].p_stagewise, test_param.p_stagewise)  # type:ignore
        and batch[3].p_stagewise.dtype == test_param.p_stagewise.dtype  # type:ignore
    )
    assert batch[3].p_stagewise_sparse_idx is None

    torch.testing.assert_close(
        batch[4].a, torch.tensor([[1, 2, 3], [1, 2, 3]], device=device, dtype=dtype)
    )
    torch.testing.assert_close(
        batch[4].b, torch.tensor([1.0, 1.0], device=device, dtype=dtype)
    )
    assert isinstance(batch[5], AcadosOcpFlattenedBatchIterate)
    for i, field in enumerate(fields(AcadosOcpFlattenedBatchIterate)):
        if field.name == "N_batch":
            assert getattr(batch[5], field.name) == 2
            continue
        arr = getattr(batch[5], field.name)
        assert np.array_equal(arr, np.ones((2, i + 1), dtype=np.float64))


def test_sample_order_consistency():
    # NOTE: It should be enough to test preservation of order here for the types
    # for which we have custom rules
    collate_fn_map = {
        AcadosOcpSolverInput: collate_acados_ocp_solver_input,
        AcadosOcpFlattenedIterate: collate_acados_flattened_iterate_fn,
    }
    buffer = ReplayBuffer(
        buffer_limit=10,
        device="cpu",
        tensor_dtype=torch.float32,
        collate_fn_map=collate_fn_map,
    )
    data_one = (
        AcadosOcpSolverInput(
            x0=np.array([[1, 2, 3]], dtype=np.float32),
            p_global=np.array([1, 1, 1]),
            p_stagewise=np.ones((2, 2)),
            p_stagewise_sparse_idx=np.ones((2, 2)),
        ),
        AcadosOcpFlattenedIterate(
            x=np.ones(1),
            u=np.ones(1),
            z=np.ones(1),
            sl=np.ones(1),
            su=np.ones(1),
            pi=np.ones(1),
            lam=np.ones(1),
        ),
    )
    data_two = (
        AcadosOcpSolverInput(
            x0=np.array([[1, 2, 3]], dtype=np.float32),
            p_global=np.array([0, 0, 0]),
            p_stagewise=np.zeros((2, 2)),
            p_stagewise_sparse_idx=np.zeros((2, 2)),
        ),
        AcadosOcpFlattenedIterate(
            x=np.zeros(1),
            u=np.zeros(1),
            z=np.zeros(1),
            sl=np.zeros(1),
            su=np.zeros(1),
            pi=np.zeros(1),
            lam=np.zeros(1),
        ),
    )
    buffer.put(data_one)
    buffer.put(data_two)
    batch = buffer.sample(2)

    def sample_is_consistent(sample_idx):
        if np.array_equal(batch[0].p_global[sample_idx], np.array([1, 1, 1])):
            assert np.array_equal(batch[0].p_stagewise[sample_idx], np.ones((2, 2)))
            assert np.array_equal(
                batch[0].p_stagewise_sparse_idx[sample_idx], np.ones((2, 2))
            )
            for field in fields(AcadosOcpFlattenedIterate):
                if field.name == "N_batch":
                    continue
                assert np.array_equal(
                    getattr(batch[1], field.name)[sample_idx], np.ones(1)
                )

        elif np.array_equal(batch[0].p_global[sample_idx], np.array([0, 0, 0])):
            assert np.array_equal(batch[0].p_stagewise[sample_idx], np.zeros((2, 2)))
            assert np.array_equal(
                batch[0].p_stagewise_sparse_idx[sample_idx], np.zeros((2, 2))
            )
            for field in fields(AcadosOcpFlattenedIterate):
                if field.name == "N_batch":
                    continue
                assert np.array_equal(
                    getattr(batch[1], field.name)[sample_idx], np.zeros(1)
                )
        else:
            assert False

    sample_is_consistent(0)
    sample_is_consistent(1)
    assert not np.array_equal(batch[0].p_global[0], batch[0].p_global[1])


def test_length():
    buffer = ReplayBuffer(buffer_limit=2, device="cpu", tensor_dtype=torch.float32)
    dummy1 = (1, 2, 3)
    dummy2 = (1, 2, 3)
    dummy3 = (1, 2, 3)
    buffer.put(dummy1)
    buffer.put(dummy2)
    buffer.put(dummy3)
    assert len(buffer) == 2
    try:
        buffer.sample(3)
        assert False
    except ValueError:
        pass


def test_state_dict():
    buffer = ReplayBuffer(buffer_limit=3, device="cpu", tensor_dtype=torch.float32)

    dummy = 1
    dummy_tensor = torch.tensor([1, 1], device="cpu", dtype=torch.float32)

    for _ in range(3):
        buffer.put(dummy)

    state_dict = buffer.state_dict()
    assert torch.equal(buffer.sample(2), dummy_tensor)

    buffer = ReplayBuffer(buffer_limit=2, device="cpu", tensor_dtype=torch.float32)

    buffer.load_state_dict(state_dict)

    assert len(buffer) == 2
    assert torch.equal(buffer.sample(2), dummy_tensor)
