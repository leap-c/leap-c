import numpy as np
from leap_c.examples.point_mass import PointMassMPC, PointMassOcpEnv
from acados_template import AcadosOcpSolver


def main():
    mpc = PointMassMPC(learnable_params=["m", "c"])

    # policy = mpc.policy(state=np.array([1.0, 1.0, 0.0, 0.0]), p_global=None)[0]
    env = PointMassOcpEnv(mpc)

    (x, p), _ = env.reset(seed=0)

    x0 = np.array([1.0, 1.0, 0.0, 0.0])

    _ = env.step(mpc.policy(state=x0, p_global=None)[0])


if __name__ == "__main__":
    main()
