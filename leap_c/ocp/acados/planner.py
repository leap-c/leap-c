from typing import get_args

import numpy as np

from leap_c.controller import CtxType
from leap_c.ocp.acados.torch import AcadosDiffMpcSensitivityOptions
from leap_c.planner import SensitivityOptions

# TODO Jasper: Needs to be updated if we go to a dictionary based sensitivity derivation.
TO_ACADOS_DIFFMPC_SENSOPTS: dict[SensitivityOptions, AcadosDiffMpcSensitivityOptions] = {
    "du0_dp": "du0_dp_global",
    "dx_dp": "dx_dp_global",
    "du_dp": "du_dp_global",
    "dvalue_dp": "dvalue_dp_global",
    "dvalue_daction": "dvalue_du0",
    "du0_dx0": "du0_dx0",
    "dvalue_dx0": "dvalue_dx0",
}


def acados_sensitivity(diff_mpc, ctx: CtxType, name: SensitivityOptions) -> np.ndarray:
    if name not in TO_ACADOS_DIFFMPC_SENSOPTS:
        raise ValueError(
            f"Unknown sensitivity option `{name}`; available options: "
            + ", ".join(get_args(SensitivityOptions))
        )
    return diff_mpc.sensitivity(ctx, TO_ACADOS_DIFFMPC_SENSOPTS[name])
