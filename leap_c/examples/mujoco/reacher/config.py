from dataclasses import dataclass

import numpy as np
from pinocchio import Model


@dataclass(kw_only=True)
class ReacherParams:
    xy_ee_ref: np.ndarray
    q_sqrt_diag: np.ndarray
    r_sqrt_diag: np.ndarray


def make_default_reacher_params(pinocchio_model: Model) -> ReacherParams:
    return ReacherParams(
        xy_ee_ref=np.array([0.21, 0.0]),
        q_sqrt_diag=np.array([10.0, 10.0]),
        r_sqrt_diag=np.array([0.5] * pinocchio_model.nq),
    )
