from pathlib import Path
import numpy as np

from leap_c.examples.mujoco.reacher.env import ReacherEnv
from leap_c.examples.mujoco.reacher.mpc import ReacherMpc
from leap_c.examples.mujoco.reacher.task import ReacherTask
from leap_c.examples.mujoco.reacher.util import (
    InverseKinematicsSolver,
    PathGeometry,
    ReferencePath,
)
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


def main_inverse_kinematics(
    pinocchio_model: pin.Model,
    ik_solver: InverseKinematicsSolver,
    q: np.ndarray,
    dq: np.ndarray,
    target_position: np.ndarray,
) -> None:
    q_ref, dq_ref, _, _, _ = ik_solver(
        q=q,
        dq=dq,
        target_position=target_position,
    )

    print("q_ref: ", q_ref)
    print("dq_ref: ", dq_ref)

    data = pinocchio_model.createData()
    pin.forwardKinematics(
        pinocchio_model,
        data,
        q_ref,
        dq_ref,
    )
    pin.updateFramePlacements(pinocchio_model, data)

    xy_ee = data.oMf[pinocchio_model.getFrameId("fingertip")].translation[:2]
    print("xy_ee: ", xy_ee)
    print("target_position: ", target_position)
    print("norm(xy_ee - target_position): ", norm(xy_ee - target_position))


def main_warmstart(
    env: ReacherEnv,
    mpc: ReacherMpc,
    pinocchio_model: pin.Model,
    ik_solver: InverseKinematicsSolver,
    state_representation: str = "q",
):
    p_global = mpc.ocp.p_global_values
    ocp_solver = mpc.ocp_solver

    data = pinocchio_model.createData()

    # Create a target position
    q_target = np.array([np.deg2rad(10.0), np.deg2rad(90.0)])
    dq_target = np.array([0.0, 0.0])
    pin.forwardKinematics(
        pinocchio_model,
        data,
        q_target,
        dq_target,
    )
    pin.updateFramePlacements(pinocchio_model, data)

    xy_ee_target = data.oMf[pinocchio_model.getFrameId("fingertip")].translation[:2]
    # xy_ee_target = np.array([0.1, 0.1])

    # q0 = np.deg2rad(np.array([0.0, 0.0]))
    # dq0 = np.deg2rad(np.array([0.0, 0.0]))
    # x0 = np.concatenate([q0, dq0])

    q_ref, dq_ref, _, _, _ = ik_solver(
        q=np.array([np.arctan2(xy_ee_target[1], xy_ee_target[0])] * pinocchio_model.nq),
        dq=np.zeros(pinocchio_model.nv),
        target_position=np.concatenate([xy_ee_target, np.array([0.01])]),
    )

    ik_solver.plot_solver_iterations()

    x_ref = np.concatenate([q_ref, dq_ref])

    if state_representation == "q":
        for stage in range(ocp_solver.acados_ocp.solver_options.N_horizon + 1):
            ocp_solver.set(stage, "x", x_ref)

    q0 = np.deg2rad(np.array([10.0, 10.0]))
    observation = torch.Tensor(
        np.concatenate(
            [
                np.cos(q0),
                np.sin(q0),
                np.array([0.1, 0.1]),
                np.zeros(2),
                np.zeros(2),
            ]
        )
    )

    param_nn = mpc.ocp_solver.acados_ocp.p_global_values
    param_nn[0:2] = np.array([0.0, 0.0])

    x0 = prepare_mpc_input(obs=observation, param_nn=param_nn).x0.detach().numpy()

    print("q_target: ", q_target)
    print("q_ref: ", q_ref)
    print("norm(q_target - q_ref): ", np.linalg.norm(q_target - q_ref))
    # print("x0: ", x0)
    # print("x_ref: ", x_ref)
    print("xy_ee_target: ", xy_ee_target)
    # print("xy_ee_ref: ", params["xy_ee_ref"])

    # p_global[0:2] = observation[4:6]

    ocp_solver.set_p_global_and_precompute_dependencies(data_=p_global)

    ocp_solver.solve_for_x0(x0_bar=x0, fail_on_nonzero_status=False)
    print("status", ocp_solver.status)

    exit(0)

    policy, _, status = mpc.policy(
        state=prepare_mpc_input(observation).x0.detach().numpy(),
        p_global=mpc.ocp.p_global_values,
    )
    print("status", status)

    p_global = mpc.ocp.p_global_values
    for _ in range(10000):
        p_global[0:2] = observation[4:6]

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
                data = pinocchio_model.createData()
                pin.forwardKinematics(pinocchio_model, data, q, dq)
                pin.updateFramePlacements(pinocchio_model, data)
                xy_ee.append(
                    data.oMf[pinocchio_model.getFrameId("fingertip")].translation[:2]
                )

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


def main_ocp_solver_closed_loop(
    env: ReacherEnv,
    mpc: ReacherMpc,
    pinocchio_model: pin.Model,
    ik_solver: InverseKinematicsSolver,
    state_representation: str = "q",
    use_ik_solver: bool = True,
):
    ocp_solver = mpc.ocp_solver

    p_global = ocp_solver.acados_ocp.p_global_values

    o = []
    observation, info = env.reset()
    while True:
        p_global[0:2] = observation[4:6]
        ocp_solver.set_p_global_and_precompute_dependencies(data_=p_global)

        if use_ik_solver:
            target_position = np.concatenate([observation[4:6], np.array([0.01])])
            target_angle = np.arctan2(target_position[1], target_position[0])

            q_ref, dq_ref, _, _, _ = ik_solver(
                q=np.array([target_angle] * pinocchio_model.nq),
                dq=np.zeros(pinocchio_model.nv),
                target_position=np.concatenate([observation[4:6], np.array([0.01])]),
            )
            x_ref = np.concatenate([q_ref, dq_ref])

            for stage in range(ocp_solver.acados_ocp.solver_options.N_horizon + 1):
                ocp_solver.set(stage, "x", x_ref)

        x0 = prepare_mpc_input(observation, offset_target=False).x0.detach().numpy()
        action = ocp_solver.solve_for_x0(x0_bar=x0, fail_on_nonzero_status=False)
        error_norm = np.linalg.norm(observation[-2:])
        print(
            f"status: {ocp_solver.status}; norm(p - p_ref): {error_norm}; q: {x0[:2]}; dq: {x0[2:]}; p_ref: {p_global[0:2]}; p: {observation[4:6]}; action: {action}",
        )

        o.append(observation)

        observation, reward, terminated, truncated, info = env.step(
            action=action,
        )
        env.render()

    # Stack o
    o = np.vstack(o)

    env.close()


def main_mpc_closed_loop(
    env: ReacherEnv,
    mpc: ReacherMpc,
    pinocchio_model: pin.Model,
    ik_solver: InverseKinematicsSolver | None = None,
) -> None:
    o = []
    observation, info = env.reset()
    param_nn = np.concatenate(
        [np.array([0.0, 0.0]), mpc.ocp_solver.acados_ocp.p_global_values[2:]]
    )

    solver_state = None
    while True:
        mpc_input = prepare_mpc_input(
            obs=observation,
            param_nn=param_nn,
        )

        state = mpc_input.x0.detach().numpy()
        p_global = mpc_input.parameters.p_global

        action, solver_state, status = mpc.policy(
            state=state,
            p_global=p_global,
            solver_state=solver_state,
        )

        error_norm = np.linalg.norm(observation[-2:])
        print(
            f"status: {status}; norm(p - p_ref): {error_norm}; q: {mpc_input.x0[:2]}; dq: {mpc_input.x0[2:]}; p_ref: {p_global[0:2]}; p: {observation[4:6]}; action: {action}",
        )

        o.append(observation)

        observation, reward, terminated, truncated, info = env.step(
            action=action,
        )
        env.render()

    # Stack o
    o = np.vstack(o)

    env.close()


RUN_ONLY_IK_SOLVER = False
RUN_MAIN_WARMSTART = False
RUN_MAIN_OCP_SOLVER_CLOSED_LOOP = False
RUN_MAIN_MPC_CLOSED_LOOP = True

if __name__ == "__main__":
    mjcf_path = Path(
        "/home/dirk/cybernetics/leap-c/leap_c/examples/mujoco/reacher/reacher"
    ).with_suffix(".xml")

    pinocchio_model = pin.buildModelFromMJCF(mjcf_path)

    ik_solver = InverseKinematicsSolver(
        pinocchio_model=pinocchio_model,
        step_size=0.1,
        max_iter=1000,
        tol=1e-6,
        print_level=0,
        plot_level=0,
    )

    reference_path = ReferencePath(
        geometry=PathGeometry(
            type="ellipse",
            origin=(0, 0.1, 0.01),
            orientation=(0.0, 0.0, 0.0),
            length=0.1,
            width=0.1,
            direction=+1,
        ),
        max_reach=ik_solver.max_reach,
    )

    env = ReacherEnv(
        # train=False, xml_file=mjcf_path.as_posix(), render_mode="rgb_array"
        train=False,
        xml_file=mjcf_path.as_posix(),
        # render_mode="human",
        render_mode="rgb_array",
        reference_path=reference_path,
    )
    state_representation = "q"

    if state_representation == "q":
        prepare_mpc_input = prepare_mpc_input_q
    elif state_representation == "sin_cos":
        prepare_mpc_input = prepare_mpc_input_cosq_sinq

    if RUN_ONLY_IK_SOLVER:
        q = np.array([[0.0700362, 0.0199048, 0.1603086, 1.9847226]])
        xy_ee_ref = np.array([0.15683472, 0.17357419])
        main_inverse_kinematics(
            pinocchio_model=pinocchio_model,
            ik_solver=ik_solver,
            q=np.array([np.deg2rad(10.0), np.deg2rad(90.0)]),
            dq=np.array([0.0, 0.0]),
            target_position=np.array([0.1, 0.1]),
        )

    dt = 0.1
    T_horizon = 1.0

    mpc = ReacherMpc(
        learnable_params=[
            "xy_ee_ref",
            "q_sqrt_diag",
            "r_sqrt_diag",
        ],
        params={
            "xy_ee_ref": np.array([0.21, 0.0]),
            "q_sqrt_diag": np.array([np.sqrt(100.0)] * 2),
            "r_sqrt_diag": np.array([np.sqrt(0.1)] * 2),
        },
        mjcf_path=mjcf_path,
        N_horizon=int(T_horizon / dt),
        T_horizon=T_horizon,
        state_representation=state_representation,
    )

    if RUN_MAIN_WARMSTART:
        main_warmstart(
            env=env,
            mpc=mpc,
            pinocchio_model=pinocchio_model,
            ik_solver=ik_solver,
            state_representation=state_representation,
        )
    if RUN_MAIN_OCP_SOLVER_CLOSED_LOOP:
        main_ocp_solver_closed_loop(
            env=env,
            mpc=mpc,
            pinocchio_model=pinocchio_model,
            ik_solver=ik_solver,
            state_representation=state_representation,
        )

    if RUN_MAIN_MPC_CLOSED_LOOP:
        main_mpc_closed_loop(
            env=env,
            mpc=mpc,
            pinocchio_model=pinocchio_model,
            ik_solver=ik_solver,
        )
