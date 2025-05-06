from pathlib import Path
import numpy as np

from leap_c.examples.mujoco.reacher.env import ReacherEnv
from leap_c.examples.mujoco.reacher.mpc import ReacherMpc
from leap_c.examples.mujoco.reacher.task import ReacherTask
from leap_c.examples.mujoco.reacher.task import (
    prepare_mpc_input_q,
    prepare_mpc_input_cosq_sinq,
)
from leap_c.mpc import MpcInput

import pinocchio as pin
import matplotlib.pyplot as plt
import torch
import casadi as ca


def main_env(mjcf_path: Path):
    env = ReacherEnv(train=False, xml_file=mjcf_path.as_posix(), render_mode="human")

    state_representation = "q"
    mpc = ReacherMpc(
        learnable_params=[
            "xy_ee_ref",
            "q_sqrt_diag",
            "r_sqrt_diag",
        ],
        params={
            "xy_ee_ref": np.array([0.2, 0.0]),
            "q_sqrt_diag": np.array([np.sqrt(10.0)] * 2),
            "r_sqrt_diag": np.array([np.sqrt(0.05)] * 2),
        },
        mjcf_path=mjcf_path,
        # N_horizon=200,
        # T_horizon=2.0,
        state_representation=state_representation,
    )
    model = pin.buildModelFromMJCF(mjcf_path)
    data = model.createData()

    ocp_solver = mpc.ocp_solver

    observation = torch.Tensor([1.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    if state_representation == "q":
        prepare_mpc_input = prepare_mpc_input_q
    elif state_representation == "sin_cos":
        prepare_mpc_input = prepare_mpc_input_cosq_sinq

    x0 = prepare_mpc_input(observation).x0.detach().numpy()
    ocp_solver.solve_for_x0(x0_bar=x0)
    print("status", ocp_solver.status)

    policy, _, status = mpc.policy(
        state=prepare_mpc_input(observation).x0.detach().numpy(),
        p_global=mpc.ocp.p_global_values,
    )
    print("status", status)

    exit(0)
    # p_global = mpc.ocp.p_global_values

    # observation, info = env.reset()
    # env.render()
    # solver_state = None
    p_global = mpc.ocp.p_global_values
    for _ in range(10000):
        p_global[0:2] = observation[4:6]
        # p_global[0:2] = np.random.uniform(-0.2, 0.2, size=2)

        mpc_output = mpc.policy(
            state=prepare_mpc_input(observation).x0.detach().numpy(),
            p_global=p_global,
            # solver_state=solver_state,
        )

        if True:
            X = np.vstack(
                [
                    ocp_solver.get(stage, "x")
                    for stage in range(
                        ocp_solver.acados_ocp.solver_options.N_horizon + 1
                    )
                ]
            )

            U = np.vstack(
                [
                    ocp_solver.get(stage, "u")
                    for stage in range(ocp_solver.acados_ocp.solver_options.N_horizon)
                ]
            )

            q0_mpc = np.atan2(X[:, 2], X[:, 0])
            q1_mpc = np.atan2(X[:, 3], X[:, 1])
            q0_mpc_casadi = ca.atan2(X[:, 2], X[:, 0])
            q1_mpc_casadi = ca.atan2(X[:, 3], X[:, 1])
            dq0_mpc = X[:, 4]
            dq1_mpc = X[:, 5]

            q_mpc = np.vstack([q0_mpc, q1_mpc]).T
            dq_mpc = np.vstack([dq0_mpc, dq1_mpc]).T

            print("q0_mpc: ", q0_mpc)
            print("q0_mpc_casadi: ", q0_mpc_casadi)

            xy_ee = []
            for q, dq in zip(q_mpc, dq_mpc, strict=True):
                data = model.createData()
                pin.forwardKinematics(model, data, q, dq)
                pin.updateFramePlacements(model, data)
                xy_ee.append(data.oMf[model.getFrameId("fingertip")].translation[:2])

            # Roll out environment on mpc plan
            # O = np.vstack([env.step(u)[0] for u in U])
            # q0 = np.arctan2(O[:, 2], O[:, 0])
            # q1 = np.arctan2(O[:, 3], O[:, 1])
            # dq0 = O[:, 6]
            # dq1 = O[:, 7]

            plt.figure()
            plt.subplot(4, 1, 1)
            plt.plot(q0_mpc, label="q0_mpc")
            # plt.plot(q0, label="q0")
            plt.legend()
            plt.subplot(4, 1, 2)
            plt.plot(q1_mpc, label="q1_mpc")
            # plt.plot(q1, label="q1")
            plt.legend()
            plt.subplot(4, 1, 3)
            plt.plot(dq0_mpc, label="dq0_mpc")
            # plt.plot(dq0, label="dq0")
            plt.legend()
            plt.subplot(4, 1, 4)
            plt.plot(dq1_mpc, label="dq1_mpc")
            # plt.plot(dq1, label="dq1")
            plt.legend()
            plt.xlabel("k")

            plt.figure()
            plt.plot(U[:, 0], label="u0")
            plt.plot(U[:, 1], label="u1")
            plt.legend()
            plt.title("U")
            plt.xlabel("k")
            plt.ylabel("U")

            xy_ee = np.array(xy_ee)

            k = np.arange(0, xy_ee.shape[0])

            print("X: ", X)
            print("U: ", U)
            print("Observation[4:6]: ", observation[4:6])
            print("")

            plt.figure()
            plt.plot(k, xy_ee[:, 0], label="x")
            plt.plot(k, xy_ee[:, 1], label="y")
            plt.plot(k, p_global[0] * np.ones_like(k), label="x_ref")
            plt.plot(k, p_global[1] * np.ones_like(k), label="y_ref")
            plt.legend()
            plt.title("xy_ee")
            plt.xlabel("k")
            plt.ylabel("xy_ee")
            plt.show()

            exit(0)

        # action = mpc_output[0]
        # solver_state = mpc_output[1]

        # observation, reward, terminated, truncated, info = env.step(action)

        # data = model.createData()
        # pin.forwardKinematics(
        #     model,
        #     data,
        #     np.arcsin(observation[0:2]),
        #     observation[6:8],
        # )
        # pin.updateFramePlacements(model, data)
        # xy_ee = data.oMf[model.getFrameId("fingertip")].translation[:2]
        # print("xy_ee: ", xy_ee)
        # print("xy_ee_ref: ", p_global[0:2])
        # env.render()
        # if terminated or truncated:
        #     observation, info = env.reset()
        #     print("Resetting environment")
        # else:
        #     print(f"Reward: {reward}, Action: {action}")

    env.close()


if __name__ == "__main__":
    mjcf_path = Path(
        "/home/dirk/cybernetics/leap-c/leap_c/examples/mujoco/reacher/reacher"
    ).with_suffix(".xml")

    main_env(mjcf_path=mjcf_path)
