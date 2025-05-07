from pathlib import Path
import numpy as np

from leap_c.examples.mujoco.reacher.env import ReacherEnv
from leap_c.examples.mujoco.reacher.mpc import ReacherMpc
from leap_c.examples.mujoco.reacher.task import ReacherTask
from leap_c.examples.mujoco.reacher.util import InverseKinematicsSolver
from leap_c.examples.mujoco.reacher.task import (
    prepare_mpc_input_q,
    prepare_mpc_input_cosq_sinq,
)
from leap_c.mpc import MpcInput

import pinocchio as pin
import matplotlib.pyplot as plt
import torch
import casadi as ca

from numpy.linalg import norm, solve


def main_env(mjcf_path: Path):
    env = ReacherEnv(train=False, xml_file=mjcf_path.as_posix(), render_mode="human")

    model = pin.buildModelFromMJCF(mjcf_path)
    data = model.createData()

    q_target = np.array([np.deg2rad(60.0), np.deg2rad(20.0)])
    dq_target = np.array([0.0, 0.0])
    pin.forwardKinematics(
        model,
        data,
        q_target,
        dq_target,
    )
    pin.updateFramePlacements(model, data)
    xy_ee_target = data.oMf[model.getFrameId("fingertip")].translation[:2]

    ik_solver = InverseKinematicsSolver(
        pinocchio_model=model,
        step_size=0.2,
        max_iter=1000,
        tol=1e-4,
        print_level=0,
        plot_level=1,
    )

    state_representation = "sin_cos" if False else "q"

    params = {
        "xy_ee_ref": xy_ee_target,
        "q_sqrt_diag": np.array([np.sqrt(10.0)] * 2),
        "r_sqrt_diag": np.array([np.sqrt(0.05)] * 2),
    }

    q0 = np.deg2rad(np.array([0.0, 0.0]))
    dq0 = np.deg2rad(np.array([0.0, 0.0]))

    x0 = np.concatenate([q0, dq0])

    q_ref, dq_ref, _, _, _ = ik_solver(
        q=np.deg2rad(np.array([45.0, 45.0])),
        # q=q0,
        dq=dq0,
        target_position=np.concatenate([xy_ee_target, np.array([0.01])]),
    )
    x_ref = np.concatenate([q_ref, dq_ref])

    print("q_target: ", q_target)
    print("q_ref: ", q_ref)

    # Wrap q_ref to [-pi, pi]
    # q_ref_wrapped = np.arctan2(np.sin(q_ref), np.cos(q_ref))
    # print("q_ref_wrapped: ", q_ref_wrapped)

    mpc = ReacherMpc(
        learnable_params=[
            "xy_ee_ref",
            "q_sqrt_diag",
            "r_sqrt_diag",
        ],
        params=params,
        mjcf_path=mjcf_path,
        N_horizon=200,
        T_horizon=2.0,
        state_representation=state_representation,
    )

    ocp_solver = mpc.ocp_solver

    if state_representation == "q":
        for stage in range(ocp_solver.acados_ocp.solver_options.N_horizon + 1):
            ocp_solver.set(stage, "x", x_ref)

    observation = torch.Tensor(
        np.concatenate(
            [
                np.cos(q0),
                np.sin(q0),
                params["xy_ee_ref"],
                dq0,
            ]
        )
    )

    if state_representation == "q":
        prepare_mpc_input = prepare_mpc_input_q
    elif state_representation == "sin_cos":
        prepare_mpc_input = prepare_mpc_input_cosq_sinq

    x0 = prepare_mpc_input(observation).x0.detach().numpy()
    print("x0: ", x0)
    print("x_ref: ", x_ref)
    print("xy_ee_target: ", xy_ee_target)
    print("xy_ee_ref: ", params["xy_ee_ref"])

    ocp_solver.solve_for_x0(x0_bar=x0, fail_on_nonzero_status=False)
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
    # pin_warmstart(mjcf_path=mjcf_path)
