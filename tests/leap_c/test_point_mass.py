import numpy as np
from leap_c.examples.pointmass.mpc import PointMassMPC


def main():
    mpc = PointMassMPC(learnable_params=["m", "c", "q_diag"])

    s = np.array([1.0, 0.0, 0.0, 0.0])
    a = np.array([0.0, 0.0])

    policy = mpc.policy(state=s, p_global=None)[0]
    print(policy)

    state_value = mpc.state_value(state=s, p_global=None)[0]
    print(state_value)

    state_action_value = mpc.state_action_value(state=s, action=a, p_global=None)[0]
    print(state_action_value)

    # env = PointMassOcpEnv(mpc)

    # (x, p), _ = env.reset(seed=0)

    # x0 = np.array([1.0, 1.0, 0.0, 0.0])

    # _ = env.step(mpc.policy(state=x0, p_global=None)[0])


if __name__ == "__main__":
    main()
