from acados_template import AcadosOcp
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
)

from leap_c.mpc import (
    MPC,
    MPCInput,
    MPCOutput,
    MPCSingleState,
    MPCBatchedState,
)


class LinearMPC(MPC):
    """docstring for LinearMPC."""

    def __init__(self, ocp: AcadosOcp):
        super().__init__()

        self.ocp = ocp

    def __call__(
        self,
        mpc_input: MPCInput,
        mpc_state: MPCSingleState | MPCBatchedState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MPCOutput, MPCSingleState | MPCBatchedState]:
        pass

    def _solve(
        self,
        mpc_input: MPCInput,
        mpc_state: MPCSingleState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MPCOutput, AcadosOcpFlattenedIterate]:
        pass

    def _batch_solve(
        self,
        mpc_input: MPCInput,
        mpc_state: MPCBatchedState | None = None,
        dudx: bool = False,
        dudp: bool = False,
        dvdx: bool = False,
        dvdu: bool = False,
        dvdp: bool = False,
        use_adj_sens: bool = True,
    ) -> tuple[MPCOutput, AcadosOcpFlattenedBatchIterate]:
        pass
