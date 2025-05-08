import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin


class InverseKinematicsSolver:
    def __init__(
        self,
        pinocchio_model: pin.Model,
        step_size: float = 0.2,
        max_iter: int = 1000,
        tol: float = 1e-4,
        plot_level: int = 0,
        print_level: int = 0,
    ) -> None:
        self.model = pinocchio_model
        self.data = self.model.createData()
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = tol
        self.plot_level = plot_level
        self.print_level = print_level

        # TODO: Pass frame_id as a parameter
        self.FRAME_ID = self.model.getFrameId("fingertip")

    def __call__(
        self, q: np.ndarray, dq: np.ndarray, target_position: np.ndarray
    ) -> np.ndarray:
        q_data = [q]
        dq_data = [dq]
        position_data = []
        self.target_position = target_position
        for i in range(self.max_iter):
            # Compute current position
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_position = self.data.oMf[self.FRAME_ID].translation

            # Compute position error
            error = target_position - current_position

            # Check if we've reached desired precision
            if np.linalg.norm(error) < self.eps:
                if self.print_level > 0:
                    print("Initialization complete.")
                    print(f"Convergence achieved at iteration {i}")
                    print("Final joint configuration:", q)
                    print("Final end effector position:", current_position)
                break

            # Compute the Jacobian for the end effector
            J = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.FRAME_ID,
                pin.LOCAL_WORLD_ALIGNED,
            )

            # Extract position part of the Jacobian (first 3 rows)
            J_position = J[:3, :]

            # Update joint velocities using pseudo-inverse of the Jacobian
            dq = np.linalg.pinv(J_position) @ error

            # Update joint configuration (simple integration)
            q = pin.integrate(self.model, q, self.step_size * dq)

            q_data.append(q)
            dq_data.append(dq)
            position_data.append(current_position.copy())

            if i == self.max_iter - 1 and self.print_level > 0:
                print("Warning: Maximum iterations reached without convergence")

        q_data = np.array(q_data)
        dq_data = np.array(dq_data)
        position_data = np.array(position_data)

        # Compute numerical gradient of dq_data
        ddq_data = np.gradient(dq_data, self.step_size, axis=0)

        # Compute joint torques using rnea
        tau = []
        for q_k, dq_k, ddq_k in zip(q_data, dq_data, ddq_data, strict=False):
            tau.append(pin.rnea(self.model, self.data, q_k, dq_k, ddq_k).copy())

        tau = np.array(tau)

        if self.plot_level > 0:
            self.plot_solver_iterations(target_position, q_data, dq_data, position_data)

        # Convert to the desired state representation
        if self.print_level > 1:
            print("Joint angles (q):", q_data)
            print("Joint velocities (dq):", dq_data)
            print("End effector positions:", current_position)
            print("Target position:", target_position)
            print("Error:", target_position - current_position)

        self.q_data = q_data
        self.position_data = position_data

        return (
            q_data[-1, :],
            dq_data[-1, :],
            ddq_data[-1, :],
            position_data[-1, :],
            tau[-1, :],
        )

    def plot_solver_iterations(
        self,
    ) -> None:
        q_data = self.q_data
        position_data = self.position_data
        target_position = self.target_position
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(q_data)
        plt.legend(["Joint 1", "Joint 2"])
        plt.ylabel("Joint Angles (radians)")
        plt.title("Joint Configuration Over Iterations")
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(
            target_position[0] * np.ones(len(position_data)),
            linestyle="--",
        )
        plt.plot(
            target_position[1] * np.ones(len(position_data)),
            linestyle="--",
        )
        # Reset color cycle
        plt.gca().set_prop_cycle(None)
        plt.plot(position_data[:, :2])
        plt.legend(["xref", "yref", "x", "y"])
        plt.title("End Effector Position (meters)")
        plt.xlabel("Iteration")
        plt.grid()

        # plt.figure()
        # plt.plot(position_data[:, 0], position_data[:, 1], "o-")
        # plt.title("End Effector Position Over Iterations")
        # plt.xlabel("X Position")
        # plt.ylabel("Y Position")
        # plt.grid()

        plt.show()
