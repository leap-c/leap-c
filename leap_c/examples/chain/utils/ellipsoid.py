import numpy as np
from matplotlib import pyplot as plt


class Ellipsoid:
    def __init__(self, center: np.ndarray, radii: np.ndarray, seed: int = 0):
        self.center = center
        self.radii = radii

        self.surface = self.spherical_to_cartesian(
            phi=np.linspace(0, 2 * np.pi, 100), theta=np.linspace(0, np.pi, 100)
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def plot_surface(self) -> plt.Figure:
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(
            *self.surface.transpose(2, 0, 1),
            rstride=4,
            cstride=4,
            color="b",
            alpha=0.75,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Ellipsoid")
        return fig

    def plot_points(self, points: np.ndarray) -> plt.Figure:
        fig = self.plot_surface()
        ax = fig.get_axes()[0]
        ax.plot(points[..., 0], points[..., 1], points[..., 2], "o", color="r")
        return fig

    def spherical_to_cartesian(
        self, phi: np.ndarray | float, theta: np.ndarray | float
    ) -> np.ndarray:
        x = self.radii[0] * np.outer(np.cos(phi), np.sin(theta)) + self.center[0]
        y = self.radii[1] * np.outer(np.sin(phi), np.sin(theta)) + self.center[1]
        z = self.radii[2] * np.outer(np.ones_like(phi), np.cos(theta)) + self.center[2]

        out = np.stack((x, y, z), axis=-1)

        if type(phi) is float:
            return out.squeeze()

        return out

    def sample_within_range(
        self, phi_range: list[float, float], theta_range: list[float, float], size: int
    ) -> np.ndarray:
        phi = self.rng.uniform(low=phi_range[0], high=phi_range[1], size=size)
        theta = self.rng.uniform(low=theta_range[0], high=theta_range[1], size=size)

        return self.spherical_to_cartesian(phi, theta)
