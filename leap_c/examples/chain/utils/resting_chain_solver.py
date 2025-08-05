from typing import Callable

import casadi as ca
import numpy as np
from casadi import vertcat
from casadi.tools import struct_symSX, entry

from leap_c.examples.chain.config import ChainParams


def _define_param_struct(n_mass: int) -> struct_symSX:
    return struct_symSX(
        [
            entry("m", shape=(1,), repeat=n_mass - 1),
            entry("D", shape=(3,), repeat=n_mass - 1),
            entry("L", shape=(3,), repeat=n_mass - 1),
            entry("C", shape=(3,), repeat=n_mass - 1),
            entry("w", shape=(3,), repeat=n_mass - 2),
            entry("fix_point", shape=(3,)),
            entry("p_last", shape=(3,)),
        ]
    )


def _define_nlp_solver(n_mass: int, f_expl: Callable) -> Callable:
    x = struct_symSX(
        [
            entry("pos", shape=(3, 1), repeat=n_mass - 1),
            entry("vel", shape=(3, 1), repeat=n_mass - 2),
        ]
    )

    xdot = ca.SX.sym("xdot", x.cat.shape)

    u = ca.SX.sym("u", 3, 1)

    p = _define_param_struct(n_mass=n_mass)
    # decision variables
    w = vertcat(*[x.cat, xdot, u])

    g = vertcat(
        *[
            xdot
            - f_expl(
                x=x,
                u=u,
                p={key: vertcat(*p[key]) for key in ["m", "D", "L", "C", "w"]},
                x0=p["fix_point"],
            ),
            x["pos", -1] - p["p_last"],
            u,
        ]
    )

    nlp = {"x": w, "f": 0, "g": g, "p": p.cat}

    return ca.nlpsol("solver", "ipopt", nlp), x(0), p(0)


class RestingChainSolver:
    def __init__(
        self,
        n_mass: int,
        f_expl: Callable,
        params: ChainParams,
    ):
        self.n_mass = n_mass
        self.f_expl = f_expl
        self.nlp_solver, x0, p0 = _define_nlp_solver(n_mass=n_mass, f_expl=f_expl)

        p0["fix_point"] = params.fix_point.value  # Anchor point of the chain.

        # Extract parameter values from ChainParams
        for i_mass in range(n_mass - 1):
            p0["m", i_mass] = params.m.value[i_mass]
            p0["D", i_mass] = params.D.value[3 * i_mass : 3 * (i_mass + 1)]
            p0["C", i_mass] = params.C.value[3 * i_mass : 3 * (i_mass + 1)]
            p0["L", i_mass] = params.L.value[3 * i_mass : 3 * (i_mass + 1)]

        for i_pos in range(len(x0["pos"])):
            x0["pos", i_pos] = x0["pos", 0] + p0["L", i_pos] * (i_pos + 1)

        self.x0 = x0
        self.p0 = p0

    def __call__(self, p_last: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.p0["p_last"] = p_last

        self.w0 = np.concatenate(
            [
                self.x0.cat.full().flatten(),
                0 * self.x0.cat.full().flatten(),
                np.zeros(3),
            ]
        )
        sol = self.nlp_solver(x0=self.w0, lbg=0, ubg=0, p=self.p0.cat)

        nx = self.x0.cat.shape[0]

        x_ss = sol["x"].full()[:nx].flatten()
        u_ss = sol["x"].full()[-3:].flatten()

        return x_ss, u_ss
