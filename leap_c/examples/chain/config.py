from dataclasses import dataclass

import numpy as np


@dataclass(kw_only=True)
class ChainParams:
    L: np.ndarray  # rest length of spring
    D: np.ndarray  # spring constant
    C: np.ndarray  # damping constant
    m: np.ndarray  # mass of the balls
    w: np.ndarray  # disturbance on intermediate balls
    q_sqrt_diag: np.ndarray  # weight on state
    r_sqrt_diag: np.ndarray  # weight on control inputs


def make_default_chain_params(n_mass: int) -> ChainParams:
    return ChainParams(
        L=np.repeat([0.033, 0.033, 0.033], n_mass - 1),
        D=np.repeat([1.0, 1.0, 1.0], n_mass - 1),
        C=np.repeat([0.1, 0.1, 0.1], n_mass - 1),
        m=np.repeat([0.033], n_mass - 1),
        w=np.repeat([0.0, 0.0, 0.0], n_mass - 2),
        q_sqrt_diag=np.ones(3 * (n_mass - 1) + 3 * (n_mass - 2)),
        r_sqrt_diag=1e-1 * np.ones(3),
    )
