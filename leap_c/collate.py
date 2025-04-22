from typing import Any

import numpy as np
import torch
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
    AcadosOcpIterate,
)
from torch.utils._pytree import tree_map_only
from torch.utils.data._utils.collate import default_collate_fn_map

from leap_c.mpc import MpcParameter


def safe_collate_possible_nones(
    field_data: list[None] | list[np.ndarray],
) -> None | np.ndarray:
    """Checks whether the given list contains only Nones or only non-Nones.
    If it contains only Nones, it returns None, otherwise it uses np.stack on the given list.
    If a mixture of Nones and non-Nones is detected, a ValueError is raised."""
    any_none = False
    all_none = True
    for data in field_data:
        if data is None:
            any_none = True
        else:
            all_none = False
        if any_none and not all_none:
            raise ValueError("All or none of the data must be None.")
    if all_none:
        return None
    else:
        return np.stack(field_data, axis=0)  # type:ignore


def _collate_mpc_param_fn(batch, *, collate_fn_map=None):
    # Collate MPCParameters by stacking the p_global and p_stagewise parts, but do not convert them to tensors.

    glob_data = [x.p_global for x in batch]
    stag_data = [x.p_stagewise for x in batch]
    idx_data = [x.p_stagewise_sparse_idx for x in batch]

    return MpcParameter(
        p_global=safe_collate_possible_nones(glob_data),
        p_stagewise=safe_collate_possible_nones(stag_data),
        p_stagewise_sparse_idx=safe_collate_possible_nones(idx_data),
    )


def _collate_acados_flattened_iterate_fn(batch, *, collate_fn_map=None):
    return AcadosOcpFlattenedBatchIterate(
        x=np.stack([x.x for x in batch], axis=0),
        u=np.stack([x.u for x in batch], axis=0),
        z=np.stack([x.z for x in batch], axis=0),
        sl=np.stack([x.sl for x in batch], axis=0),
        su=np.stack([x.su for x in batch], axis=0),
        pi=np.stack([x.pi for x in batch], axis=0),
        lam=np.stack([x.lam for x in batch], axis=0),
        N_batch=len(batch),
    )


def _collate_acados_flattened_batch_iterate_fn(batch, *, collate_fn_map=None):
    return AcadosOcpFlattenedBatchIterate(
        x=np.concat([x.x for x in batch], axis=0),
        u=np.concat([x.u for x in batch], axis=0),
        z=np.concat([x.z for x in batch], axis=0),
        sl=np.concat([x.sl for x in batch], axis=0),
        su=np.concat([x.su for x in batch], axis=0),
        pi=np.concat([x.pi for x in batch], axis=0),
        lam=np.concat([x.lam for x in batch], axis=0),
        N_batch=sum([x.N_batch for x in batch]),
    )


def _collate_acados_iterate_fn(batch, *, collate_fn_map=None):
    # NOTE: Could also be a FlattenedBatchIterate (which has a parallelized set in the batch solver),
    # but this seems more intuitive. If the user wants to have a flattened batch iterate, he can
    # just put in AcadosOcpIterate.flatten into the buffer.
    return list(batch)


def create_collate_fn_map():
    """Create the collate function map for the collate function.
    By default, this is the default_collate_fn_map in torch.utils.data._utils.collate, with an additional
    rule for MpcParameter and AcadosOcpFlattenedIterate."""
    custom_collate_map = default_collate_fn_map.copy()

    # NOTE: If MpcParameter should also be tensorified, you can turn mpcparam_fn off
    # and use the following code for handling the Nones
    # def none_fn(batch, *, collate_fn_map=None):
    #     # Collate nones into one none but throws an error if batch contains something else than none.
    #     if any(x is not None for x in batch):
    #         raise ValueError("None collate function can only collate Nones.")
    #     return None
    # custom_collate_map[type_(None)]=none_fn

    # Keeps MPCParameter as np.ndarray

    custom_collate_map[MpcParameter] = _collate_mpc_param_fn
    custom_collate_map[AcadosOcpFlattenedIterate] = _collate_acados_flattened_iterate_fn
    custom_collate_map[AcadosOcpFlattenedBatchIterate] = (
        _collate_acados_flattened_batch_iterate_fn
    )
    custom_collate_map[AcadosOcpIterate] = _collate_acados_iterate_fn

    return custom_collate_map


def pytree_tensor_to(pytree: Any, device: str, tensor_dtype: torch.dtype) -> Any:
    """Convert all tensors in the pytree to tensor_dtype and
    move them to device."""
    return tree_map_only(
        torch.Tensor,
        lambda t: t.to(device=device, dtype=tensor_dtype),
        pytree,
    )
