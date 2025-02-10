from dataclasses import fields
from typing import Any, NamedTuple

import numpy as np
import torch
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
    AcadosOcpIterate,
)
from leap_c.collate import safe_collate_possible_nones
from leap_c.mpc import MPCParameter
from leap_c.rl.replay_buffer import ReplayBuffer


class ForNesting(NamedTuple):
    a: Any
    b: Any


def test_sample_collation_and_dtype_and_device():
    # NOTE: Only tests moving to devices if cuda is available, i.e., github probably won't test it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    buffer = ReplayBuffer(buffer_limit=10, device=device, tensor_dtype=dtype)
    data_one = (
        1,
        np.array([2], dtype=np.float64),
        torch.tensor([3], device="cpu", dtype=torch.float64),
        MPCParameter(
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
        AcadosOcpIterate(
            x_traj=[np.ones(3, dtype=np.float64)],
            u_traj=[np.ones(2, dtype=np.float64)],
            z_traj=[np.ones(1, dtype=np.float64)],
            sl_traj=[np.ones(1, dtype=np.float64)],
            su_traj=[np.ones(1, dtype=np.float64)],
            pi_traj=[np.ones(1, dtype=np.float64)],
            lam_traj=[np.ones(1, dtype=np.float64)],
        ),
    )
    data_two = (
        1,
        np.array([2], dtype=np.float64),
        torch.tensor([3], device="cpu", dtype=torch.float64),
        MPCParameter(
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
        AcadosOcpIterate(
            x_traj=[np.ones(1, dtype=np.float64)],
            u_traj=[np.ones(2, dtype=np.float64)],
            z_traj=[np.ones(3, dtype=np.float64)],
            sl_traj=[np.ones(4, dtype=np.float64)],
            su_traj=[np.ones(5, dtype=np.float64)],
            pi_traj=[np.ones(6, dtype=np.float64)],
            lam_traj=[np.ones(7, dtype=np.float64)],
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

    test_param = MPCParameter(
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
    assert len(batch[6]) == 2
    for iter in batch[6]:
        assert iter is data_one[6] or iter is data_two[6]  # type:ignore


def test_sample_order_consistency():
    # NOTE: It should be enough to test preservation of order here for the types
    # for which we have custom rules
    buffer = ReplayBuffer(buffer_limit=10, device="cpu", tensor_dtype=torch.float32)
    data_one = (
        MPCParameter(
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
        AcadosOcpIterate(
            x_traj=[np.ones(1)],
            u_traj=[np.ones(1)],
            z_traj=[np.ones(1)],
            sl_traj=[np.ones(1)],
            su_traj=[np.ones(1)],
            pi_traj=[np.ones(1)],
            lam_traj=[np.ones(1)],
        ),
    )
    data_two = (
        MPCParameter(
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
        AcadosOcpIterate(
            x_traj=[np.zeros(1)],
            u_traj=[np.zeros(1)],
            z_traj=[np.zeros(1)],
            sl_traj=[np.zeros(1)],
            su_traj=[np.zeros(1)],
            pi_traj=[np.zeros(1)],
            lam_traj=[np.zeros(1)],
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
            for field in fields(AcadosOcpIterate):
                assert np.array_equal(
                    getattr(batch[2][sample_idx], field.name)[0], np.ones(1)
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
            for field in fields(AcadosOcpIterate):
                assert np.array_equal(
                    getattr(batch[2][sample_idx], field.name)[0], np.zeros(1)
                )
        else:
            assert False

    sample_is_consistent(0)
    sample_is_consistent(1)
    assert not np.array_equal(batch[0].p_global[0], batch[0].p_global[1])


def test_safe_collate_possible_nones():
    data = [
        np.array([1, 2, 3], dtype=np.float64),
        None,
        np.array([4, 5, 6], dtype=np.float64),
    ]
    try:
        safe_collate_possible_nones(data)
        assert False
    except ValueError:
        pass
    data = [
        np.array([1, 2, 3], dtype=np.float64),
        np.array([4, 5, 6], dtype=np.float64),
    ]
    assert np.array_equal(
        safe_collate_possible_nones(data),  # type:ignore
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
    )
    data = [None, None, None]
    assert safe_collate_possible_nones(data) is None


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
