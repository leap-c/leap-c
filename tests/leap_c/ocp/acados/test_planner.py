from typing import get_args

from leap_c.ocp.acados.planner import (
    TO_ACADOS_DIFFMPC_SENSOPTS,
    AcadosDiffMpcSensitivityOptions,
    SensitivityOptions,
)


def test_TO_ACADOS_DIFFMPC_SENSOPTS_is_bijective():
    """Test that we did not forget any sensitivity option in `TO_ACADOS_DIFFMPC_SENSOPTS`.

    Also test that mappings are one-to-one without duplicates.
    """
    this = TO_ACADOS_DIFFMPC_SENSOPTS
    reverse = {v: k for k, v in TO_ACADOS_DIFFMPC_SENSOPTS.items()}

    for dictionary, options_type in [
        (this, SensitivityOptions),
        (reverse, AcadosDiffMpcSensitivityOptions),
    ]:
        # test completeness
        assert all(opt in dictionary for opt in get_args(options_type)), (
            "Some sensitivity are missing"
        )

        # test bijectivity
        values = list(dictionary.values())
        assert len(values) == len(set(values)), "Two or more keys map to the same value"
