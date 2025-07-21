from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

# Optional pinocchio import
try:
    import pinocchio as pin

    HAS_PINOCCHIO = True
except ImportError:
    pin = None
    HAS_PINOCCHIO = False


class PinocchioNotAvailableError(ImportError):
    """Raised when pinocchio is required but not installed."""

    def __init__(
        self,
        message: str = "Pinocchio is required for this functionality. Install with: pip install pin",
    ):
        super().__init__(message)


def require_pinocchio() -> None:
    """Check if pinocchio is available, raise error if not."""
    if not HAS_PINOCCHIO:
        raise PinocchioNotAvailableError


class InverseKinematicsSolver:
    def __init__(
        self,
        pinocchio_model,  # noqa: ANN001
        step_size: float = 0.2,
        max_iter: int = 1000,
        tol: float = 1e-4,
        plot_level: int = 0,
        print_level: int = 0,
    ) -> None:
        # Check pinocchio availability at initialization
        require_pinocchio()

        self.model = pinocchio_model
        self.data = self.model.createData()
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = tol
        self.plot_level = plot_level
        self.print_level = print_level

        # TODO: Pass frame_id as a parameter
        self.FRAME_ID = self.model.getFrameId("fingertip")

        # Determine the maximum reach of the robot

        # Get neutral configuration
        q = np.zeros(self.model.nq)
        # Get the position of the end effector in the neutral configuration
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        self.max_reach = np.linalg.norm(self.data.oMf[self.FRAME_ID].translation[:2])

        print("Maximum reach of the robot:", self.max_reach)

    def __call__(
        self, q: np.ndarray, dq: np.ndarray, target_position: np.ndarray
    ) -> np.ndarray:
        q_data = [q]
        dq_data = [dq]
        position_data = []
        self.target_position = target_position

        norm_target_position = np.linalg.norm(target_position[:2])
        # Check if the target position is reachable
        if norm_target_position > self.max_reach:
            raise ValueError(
                f"Target position {target_position} with norm {norm_target_position} is not reachable. "
                f"Maximum reach of the robot is {self.max_reach}."
            )

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


@dataclass
class PathGeometry:
    """Dataclass representing the geometry parameters of a reference path."""

    type: str = "ellipse"
    origin: tuple = (0.0, 0.1, 0.0)
    orientation: tuple = (0.0, 0.0, 0.0)
    length: float = 0.2
    width: float = 0.12
    direction: int = +1


class ReferencePath:
    """Class for computing points on a lemniscate reference path."""

    def __init__(self, geometry: PathGeometry, max_reach: float = 0.21) -> None:
        """
        Initialize the reference lemniscate with geometry parameters.

        Args:
            geometry: A PathGeometry object containing the path parameters
        """
        self.geometry = geometry
        self.max_reach = max_reach
        self.validate_geometry()

    def __call__(self, theta: float) -> np.ndarray:
        """
        Compute a point on the path for a given theta parameter.

        Args:
            theta: Parameter value between 0 and 1 representing position on the path

        Returns:
            2D position (x, y) on the path
        """
        return self.compute_point(theta)

    def validate_geometry(self) -> None:
        """
        Validate the geometry parameters of the reference path.
        Raises ValueError if the parameters are not valid.
        """
        if self.geometry.type not in ["lemniscate", "ellipse"]:
            raise ValueError("Invalid path type. Must be 'lemniscate' or 'ellipse'.")

        if self.geometry.length <= 0 or self.geometry.width <= 0:
            raise ValueError("Length and width must be positive values.")

        if self.geometry.direction not in [-1, 1]:
            raise ValueError("Direction must be either -1 or +1.")

        path_samples = np.array(
            [self.compute_point(theta) for theta in np.linspace(0, 1, 100)]
        )
        path_samples = path_samples[:, :2]

        if any(np.linalg.norm(path_samples, axis=1) > self.max_reach):
            raise ValueError(
                f"Path exceeds maximum reach of {self.max_reach} meters. "
                "Adjust the geometry parameters."
            )

    def compute_point(self, theta: float) -> np.ndarray:
        """
        Compute a point on the path for a given theta parameter.

        Args:
            theta: Parameter value between 0 and 1 representing position on the path

        Returns:
            2D position (x, y) on the path
        """
        s = self.geometry.direction * 2 * np.pi * (1 + theta)
        pos_ref = np.zeros(3)

        # Lemniscate equations
        if self.geometry.type == "lemniscate":
            pos_ref[0] = (self.geometry.length / 2) * np.cos(s) / (1 + np.sin(s) ** 2)
            pos_ref[1] = (
                (self.geometry.width / 2)
                * np.sqrt(2)
                * np.sin(2 * s)
                / (1 + np.sin(s) ** 2)
            )
        elif self.geometry.type == "ellipse":
            pos_ref[0] = (self.geometry.length / 2) * np.cos(s)
            pos_ref[1] = (self.geometry.width / 2) * np.sin(s)

        # Apply rotation and translation
        pos_ref = pos_ref @ Rotation.from_euler(
            "xyz", self.geometry.orientation
        ).as_matrix() + np.array(self.geometry.origin)

        # Return only x and y coordinates
        return pos_ref[:2]


def get_mjcf_path(file_name: str) -> Path:
    # Get the absolute path of the current file (the file containing this function)
    current_file_path = Path(__file__).resolve()

    # Get the directory containing the current file
    current_dir = current_file_path.parent

    # Return the path to reacher.xml in the same directory
    return (current_dir / file_name).with_suffix(".xml")
