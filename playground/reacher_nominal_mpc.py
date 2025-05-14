import numpy as np
import pinocchio as pin
import torch

from leap_c.examples.mujoco.reacher.env import ReacherEnv
from leap_c.examples.mujoco.reacher.mpc import ReacherMpc
from leap_c.examples.mujoco.reacher.task import prepare_mpc_input_q as prepare_mpc_input
from leap_c.examples.mujoco.reacher.util import (
    InverseKinematicsSolver,
    PathGeometry,
    ReferencePath,
    get_mjcf_path,
)


def main_mpc_closed_loop(
    env: ReacherEnv,
    mpc: ReacherMpc,
    n_iter: int = 200,
) -> None:
    observation, info = env.reset()

    observation = torch.from_numpy(observation).to(device="cpu", dtype=torch.float32)

    param_nn = torch.tensor(
        [[0.0, 0.0, *mpc.ocp_solver.acados_ocp.p_global_values[2:]]],
        dtype=torch.float32,
    )

    solver_state = None
    for _ in range(n_iter):
        mpc_input = prepare_mpc_input(
            obs=observation,
            param_nn=param_nn,
        )

        p_global = (
            mpc_input.parameters.p_global.detach().numpy().astype(np.float64).flatten()
        )

        action, solver_state, status = mpc.policy(
            state=mpc_input.x0.detach().numpy().flatten(),
            p_global=p_global,
            solver_state=solver_state,
        )

        error_norm = np.linalg.norm(observation[-2:])
        print(
            f"status: {status}; norm(p - p_ref): {error_norm:>8.4f}; "
            f"q: [{mpc_input.x0[0]:>9.4f}, {mpc_input.x0[1]:>9.4f}]; "
            f"dq: [{mpc_input.x0[2]:>9.4f}, {mpc_input.x0[3]:>9.4f}]; "
            f"p_ref: [{p_global[0]:>9.4f}, {p_global[1]:>9.4f}]; "
            f"p: [{observation[4]:>9.4f}, {observation[5]:>9.4f}]; "
            f"action: [{action[0]:>9.4f}, {action[1]:>9.4f}]",
        )

        observation, _, terminated, truncated, _ = env.step(
            action=action,
        )

        if terminated or truncated:
            observation, info = env.reset()
            print("Resetting environment")

        observation = torch.from_numpy(observation).to(
            device="cpu", dtype=torch.float32
        )

    env.close()


if __name__ == "__main__":
    mjcf_path = get_mjcf_path("reacher")

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
            direction=-1,
        ),
        max_reach=ik_solver.max_reach,
    )

    env = ReacherEnv(
        train=False,
        xml_file=mjcf_path.as_posix(),
        render_mode="human",  # "rgb_array"
        reference_path=reference_path,
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
        state_representation="q",
    )

    main_mpc_closed_loop(env=env, mpc=mpc)
