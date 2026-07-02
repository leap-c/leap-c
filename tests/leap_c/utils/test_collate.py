"""Tests for leap_c.utils.collate."""

import numpy as np
import pytest
import torch
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate

from leap_c.diff_mpc.data import AcadosOcpSolverInput
from leap_c.diff_mpc.function import AcadosDiffMpcCtx
from leap_c.utils import ACADOS_DIFF_MPC_COLLATE_FN_MAP, collate_acados_diff_mpc_ctx, collate_torch


def _make_ctx(status: int = 0, x0: np.ndarray | None = None) -> AcadosDiffMpcCtx:
    """Build a minimal single-sample AcadosDiffMpcCtx for testing."""
    if x0 is None:
        x0 = np.array([[1.0, 2.0]], dtype=np.float64)
    return AcadosDiffMpcCtx(
        iterate=AcadosOcpFlattenedBatchIterate(
            x=np.ones((1, 2), dtype=np.float64),
            u=np.ones((1, 1), dtype=np.float64),
            z=np.ones((1, 0), dtype=np.float64),
            sl=np.ones((1, 0), dtype=np.float64),
            su=np.ones((1, 0), dtype=np.float64),
            pi=np.ones((1, 0), dtype=np.float64),
            lam=np.ones((1, 0), dtype=np.float64),
            N_batch=1,
        ),
        status=np.array(status),
        log=None,
        solver_input=AcadosOcpSolverInput(x0=x0),
    )


@pytest.fixture
def ctx_batch() -> list[AcadosDiffMpcCtx]:
    return [
        _make_ctx(status=0, x0=np.array([[1.0, 2.0]])),
        _make_ctx(status=1, x0=np.array([[3.0, 4.0]])),
    ]


def test_collate_acados_diff_mpc_ctx(ctx_batch: list[AcadosDiffMpcCtx]) -> None:
    """Direct stacking of contexts produces a batched context."""
    result = collate_acados_diff_mpc_ctx(ctx_batch)

    assert isinstance(result, AcadosDiffMpcCtx)
    assert result.iterate.N_batch == 2
    assert result.status.tolist() == [0, 1]
    np.testing.assert_allclose(result.solver_input.x0, np.array([[[1.0, 2.0]], [[3.0, 4.0]]]))
    assert result.log is None


def test_acados_diff_mpc_collate_fn_map_keys() -> None:
    """The map exposes the ctx type as its sole key."""
    assert AcadosDiffMpcCtx in ACADOS_DIFF_MPC_COLLATE_FN_MAP
    assert ACADOS_DIFF_MPC_COLLATE_FN_MAP[AcadosDiffMpcCtx] is collate_acados_diff_mpc_ctx


def test_collate_torch_with_ctx(ctx_batch: list[AcadosDiffMpcCtx]) -> None:
    """collate_torch handles dicts containing both tensors and ctx objects."""
    samples = [
        {"x0": torch.tensor([1.0, 2.0]), "ctx": ctx_batch[0]},
        {"x0": torch.tensor([3.0, 4.0]), "ctx": ctx_batch[1]},
    ]

    batch = collate_torch(samples)

    assert isinstance(batch["x0"], torch.Tensor)
    assert batch["x0"].shape == (2, 2)
    torch.testing.assert_close(batch["x0"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    assert isinstance(batch["ctx"], AcadosDiffMpcCtx)
    assert batch["ctx"].iterate.N_batch == 2
    assert batch["ctx"].status.tolist() == [0, 1]


def test_collate_torch_plain_tensors() -> None:
    """collate_torch still handles plain tensors without ctx objects."""
    samples = [torch.tensor([1.0]), torch.tensor([2.0])]
    batch = collate_torch(samples)
    torch.testing.assert_close(batch, torch.tensor([[1.0], [2.0]]))
