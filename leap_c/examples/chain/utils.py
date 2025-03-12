from abc import abstractmethod
from typing import Callable

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot
from casadi import vertcat
from casadi.tools import entry, struct_symSX
from matplotlib import animation

latexify_plot()


def sample_from_ellipsoid_surface(w, Z):
    """Draws uniform sample from the surface of an ellipsoid with center w and variability matrix Z

    Parameters
    ----------
    w (numpy.ndarray): Center of the ellipsoid
    Z (numpy.ndarray): Symmetric positive definite variability matrix

    Returns
    -------
    numpy.ndarray: A point on the surface of the ellipsoid

    """
    n = w.shape[0]  # dimension
    lam, v = np.linalg.eig(Z)

    # Sample uniformly from unit sphere surface
    # (for surface sampling, we don't use the radius adjustment with random.rand())
    rng = np.random.default_rng()
    x = rng.normal(size=n)
    x = x / np.linalg.norm(x)  # normalize to unit sphere

    # Project to ellipsoid surface
    y = v @ (np.sqrt(lam) * x) + w
    return y


def sample_from_ellipsoid(w, Z):
    """Draws uniform sample from ellipsoid with center w and variability matrix Z"""
    n = w.shape[0]  # dimension
    lam, v = np.linalg.eig(Z)

    rng = np.random.default_rng()
    # sample in hypersphere
    r = rng.random() ** (1 / n)  # radial position of sample
    x = rng.normal(size=n)
    x = x / np.linalg.norm(x)
    x *= r
    # project to ellipsoid
    y = v @ (np.sqrt(lam) * x) + w

    return y


def plot_timings(results_list, labels, figure_filename=None, t_max=None):
    num_entries = len(labels)
    if num_entries != len(results_list):
        raise ValueError("Number of labels and result files do not match")

    width = 0.8
    fig, ax = plt.subplots(figsize=(7.5, 6))
    bottom = np.zeros(num_entries)
    colors = ["C0", "C1", "C4", "C3", "C6", "C5", "C2", "C7"]

    for i, k in enumerate(results_list[0].keys()):
        vals = [np.mean(res_dict[k]) for res_dict in results_list]
        plt.bar(labels, vals, width, label=k, bottom=bottom, color=colors[i])
        bottom += vals
    if t_max is not None:
        plt.ylim(0, t_max)
        for i in range(bottom.size):
            if bottom[i] > t_max:
                plt.text(i, 0.95 * t_max, f"{bottom[i]:.1f}", ha="center", va="bottom")

    plt.xticks(rotation=10)
    plt.grid(axis="y")
    plt.ylabel("mean computation time [ms]")
    # tight layout
    plt.tight_layout()
    ax.legend()
    if figure_filename is not None:
        plt.savefig(figure_filename)
        print(f"Saved figure to {figure_filename}")
    plt.show()


def plot_steady_state(
    x_ss: np.ndarray, u_ss: np.ndarray, n_mass: int, pos_first_mass: np.ndarray
) -> tuple[plt.Figure, plt.Figure]:
    pos_ss = x_ss[: 3 * (n_mass - 1)]

    # Concatenate xPosFirstMass and pos_ss
    pos_ss = np.concatenate((pos_first_mass, pos_ss))

    vel_ss = x_ss[3 * (n_mass - 1) :]

    # Concatenate vel_ss and u_ss
    vel_first_mass = np.zeros(3)
    vel_last_mass = u_ss
    vel_ss = np.concatenate((vel_first_mass, vel_ss, vel_last_mass))

    pos_fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(pos_ss[0::3], "o-")
    plt.subplot(3, 1, 2)
    plt.plot(pos_ss[1::3], "o-")
    plt.subplot(3, 1, 3)
    plt.plot(pos_ss[2::3], "o-")

    vel_fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(vel_ss[0::3], "o-")
    plt.subplot(3, 1, 2)
    plt.plot(vel_ss[1::3], "o-")
    plt.subplot(3, 1, 3)
    plt.plot(vel_ss[2::3], "o-")

    return pos_fig, vel_fig


def plot_chain_position_traj(simX, yPosWall=None):
    plt.figure()
    nx = simX.shape[1]
    N = simX.shape[0]
    M = int((nx / 3 - 1) / 2)
    # plt.title('Chain position trajectory')

    for i in range(M + 1):
        plt.subplot(M + 1, 3, 3 * i + 1)
        plt.ylabel("x")
        plt.plot(simX[:, 3 * i])
        plt.grid(True)

        plt.subplot(M + 1, 3, 3 * i + 2)
        plt.ylabel("y")
        plt.plot(simX[:, 3 * i + 1])
        if not yPosWall == None:
            plt.plot(yPosWall * np.ones((N,)))
        plt.grid(True)

        plt.subplot(M + 1, 3, 3 * i + 3)
        plt.ylabel("z")
        plt.plot(simX[:, 3 * i + 2])
        plt.grid(True)


def plot_chain_velocity_traj(simX):
    plt.figure()
    nx = simX.shape[1]
    M = int((nx / 3 - 1) / 2)

    simX = simX[:, (M + 1) * 3 :]

    for i in range(M):
        plt.subplot(M, 3, 3 * i + 1)
        plt.plot(simX[:, 3 * i])
        plt.ylabel("vx")
        plt.grid(True)

        plt.subplot(M, 3, 3 * i + 2)
        plt.plot(simX[:, 3 * i + 1])
        plt.ylabel("vy")
        plt.grid(True)

        plt.subplot(M, 3, 3 * i + 3)
        plt.plot(simX[:, 3 * i + 2])
        plt.ylabel("vz")
        plt.grid(True)


def plot_chain_control_traj(simU):
    plt.figure()
    # plt.title('Chain control trajectory, velocities of last mass')
    simU = np.vstack((simU[0, :], simU))

    t = np.array(range(simU.shape[0]))
    plt.subplot(3, 1, 1)
    plt.step(t, simU[:, 0])
    plt.ylabel("vx")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.step(t, simU[:, 1])
    plt.ylabel("vy")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.step(t, simU[:, 2])
    plt.ylabel("vz")
    plt.grid(True)


def plot_chain_position(x, xPosFirstMass):
    if len(x.shape) > 1:
        x = x.flatten()
    if len(xPosFirstMass.shape) > 1:
        xPosFirstMass = xPosFirstMass.flatten()

    nx = x.shape[0]
    M = int((nx / 3 - 1) / 2)

    pos = x[: 3 * (M + 1)]
    pos = np.hstack((xPosFirstMass, pos))  # append fixed mass
    pos_x = pos[::3]
    pos_y = pos[1::3]
    pos_z = pos[2::3]

    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(pos_x)
    plt.title("x position")
    plt.xlabel("mass index ")
    plt.ylabel("mass position ")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(pos_y)
    plt.title("y position")
    plt.xlabel("mass index ")
    plt.ylabel("mass position ")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(pos_z)
    plt.title("z position")
    plt.xlabel("mass index ")
    plt.ylabel("mass position ")
    plt.grid(True)


def plot_chain_position_3D(X, xPosFirstMass, XNames=None):
    """X can be either chain state, or tuple of chain states
    Xnames is a list of strings
    """
    if not isinstance(X, tuple):
        X = (X,)

    if XNames is None:
        XNames = []
        for i in range(len(X)):
            XNames += ["pos" + str(i + 1)]

    if not isinstance(XNames, list):
        XNames = [XNames]

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(xPosFirstMass[0], xPosFirstMass[1], xPosFirstMass[2], "rx")
    for i, x in enumerate(X):
        if len(x.shape) > 1:
            x = x.flatten()
        if len(xPosFirstMass.shape) > 1:
            xPosFirstMass = xPosFirstMass.flatten()

        nx = x.shape[0]
        M = int((nx / 3 - 1) / 2)
        pos = x[: 3 * (M + 1)]
        pos = np.hstack((xPosFirstMass, pos))  # append fixed mass
        pos_x = pos[::3]
        pos_y = pos[1::3]
        pos_z = pos[2::3]

        ax.plot(pos_x, pos_y, pos_z, ".-", label=XNames[i])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend()


def get_plot_lims(a):
    a_min = np.amin(a)
    a_max = np.amax(a)
    # make sure limits are not equal to each other
    eps = 1e-12
    if np.abs(a_min - a_max) < eps:
        a_min -= 1e-3
        a_max += 1e-3

    return (a_min, a_max)


def animate_chain_position(simX, xPosFirstMass, Ts=0.1, yPosWall=None):
    """Creates animation of the chain, where simX contains the state trajectory.
    dt defines the time gap (in seconds) between two succesive entries.
    """
    # chain positions
    Nsim = simX.shape[0]
    nx = simX.shape[1]
    M = int((nx / 3 - 1) / 2)
    pos = simX[:, : 3 * (M + 1)]

    pos_x = np.hstack((xPosFirstMass[0] * np.ones((Nsim, 1)), pos[:, ::3]))
    pos_y = np.hstack((xPosFirstMass[1] * np.ones((Nsim, 1)), pos[:, 1::3]))
    pos_z = np.hstack((xPosFirstMass[2] * np.ones((Nsim, 1)), pos[:, 2::3]))

    # limits in all three dimensions

    # ylim_x = (np.amin( pos_x), np.amax( pos_x))
    # ylim_y = (np.amin( pos_y), np.amax( pos_y))
    # ylim_z = (np.amin( pos_z), np.amax( pos_z))
    # eps = 1e-12
    # if np.abs(ylim_x[0] - ylim_x[1]) < eps:
    #     ylim_x[0] += 1e-3
    #     ylim_x[0] += 1e-3

    ylim_x = get_plot_lims(pos_x)
    ylim_y = get_plot_lims(pos_y)
    if yPosWall is not None:
        ylim_y = (min(ylim_y[0], yPosWall) - 0.1, ylim_y[1])
    ylim_z = get_plot_lims(pos_z)

    fig = plt.figure()
    ax1 = fig.add_subplot(311, autoscale_on=False, xlim=(0, M + 2), ylim=ylim_x)
    plt.grid(True)
    ax2 = fig.add_subplot(312, autoscale_on=False, xlim=(0, M + 2), ylim=ylim_y)
    plt.grid(True)
    ax3 = fig.add_subplot(313, autoscale_on=False, xlim=(0, M + 2), ylim=ylim_z)
    plt.grid(True)

    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    ax3.set_ylabel("z")

    # create empty plot
    (line1,) = ax1.plot([], [], ".-")
    (line2,) = ax2.plot([], [], ".-")
    (line3,) = ax3.plot([], [], ".-")

    lines = [line1, line2, line3]

    if yPosWall is not None:
        ax2.plot(yPosWall * np.ones((Nsim,)))

    def init():
        # placeholder for data
        lines = [line1, line2, line3]
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        lines[0].set_data(list(range(M + 2)), pos_x[i, :])
        lines[1].set_data(list(range(M + 2)), pos_y[i, :])
        lines[2].set_data(list(range(M + 2)), pos_z[i, :])

        return lines

    ani = animation.FuncAnimation(fig, animate, Nsim, interval=Ts * 1000, repeat_delay=500, blit=True, init_func=init)
    plt.show()
    return ani


def animate_chain_position_3D(simX, xPosFirstMass, Ts=0.1):
    """Create 3D animation of the chain, where simX contains the state trajectory.
    dt defines the time gap (in seconds) between two succesive entries.
    """
    # chain positions
    Nsim = simX.shape[0]
    nx = simX.shape[1]
    M = int((nx / 3 - 1) / 2)
    pos = simX[:, : 3 * (M + 1)]
    # import pdb; pdb.set_trace()
    pos_x = np.hstack((xPosFirstMass[0] * np.ones((Nsim, 1)), pos[:, ::3]))
    pos_y = np.hstack((xPosFirstMass[1] * np.ones((Nsim, 1)), pos[:, 1::3]))
    pos_z = np.hstack((xPosFirstMass[2] * np.ones((Nsim, 1)), pos[:, 2::3]))

    xlim = get_plot_lims(pos_x)
    ylim = get_plot_lims(pos_y)
    zlim = get_plot_lims(pos_z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", autoscale_on=False, xlim=xlim, ylim=ylim, zlim=zlim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # ax.set_aspect('equal')
    # ax.axis('off')

    # create empty plot
    # line, = ax.plot([], [], [], '.-')
    (line,) = ax.plot(pos_x[0, :], pos_y[1, :], pos_z[2, :], ".-")

    def animate(i):
        line.set_data(pos_x[i, :], pos_y[i, :])
        # line.set_data(pos_x[i,:], pos_y[i,:], pos_z[i,:])
        line.set_3d_properties(pos_z[i, :])
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        animate,
        Nsim,
        interval=Ts * 1000,
        repeat_delay=500,
        blit=True,
    )
    plt.show()
    return ani


def define_param_struct(n_mass: int) -> struct_symSX:
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


def nominal_params_to_structured_nominal_params(nominal_params: dict[str, np.ndarray]) -> dict:
    n_mass = nominal_params["m"].shape[0] + 1
    structured_nominal_params = {}
    for key in ["D", "L", "C"]:
        structured_nominal_params[key] = [nominal_params[key][3 * i : 3 * (i + 1)] for i in range(n_mass - 1)]

    for key in ["m"]:
        structured_nominal_params[key] = [nominal_params[key][i] for i in range(n_mass - 1)]

    for key in ["w"]:
        structured_nominal_params[key] = [nominal_params[key][3 * i : 3 * (i + 1)] for i in range(n_mass - 2)]

    return structured_nominal_params


def _define_nlp_solver(n_mass: int, f_expl: Callable) -> Callable:
    x = struct_symSX(
        [
            entry("pos", shape=(3, 1), repeat=n_mass - 1),
            entry("vel", shape=(3, 1), repeat=n_mass - 2),
        ]
    )

    xdot = ca.SX.sym("xdot", x.cat.shape)

    u = ca.SX.sym("u", 3, 1)

    p = define_param_struct(n_mass=n_mass)
    # decision variables
    w = vertcat(*[x.cat, xdot, u])

    g = vertcat(
        *[
            xdot - f_expl(x=x, u=u, p={key: vertcat(*p[key]) for key in ["m", "D", "L", "C", "w"]}, x0=p["fix_point"]),
            x["pos", -1] - p["p_last"],
            u,
        ]
    )

    nlp = {"x": w, "f": 0, "g": g, "p": p.cat}

    return ca.nlpsol("solver", "ipopt", nlp), x(0), p(0)


class RestingChainSolver:
    def __init__(self, n_mass: int, fix_point: np.ndarray, f_expl: Callable):
        self.n_mass = n_mass
        self.f_expl = f_expl
        self.nlp_solver, x0, p0 = _define_nlp_solver(n_mass=n_mass, f_expl=f_expl)

        p0["fix_point"] = fix_point  # Anchor point of the chain. See f_expl for more details.
        for i_mass in range(n_mass - 1):
            p0["m", i_mass] = 0.033
            p0["D", i_mass] = np.array([1.0, 1.0, 1.0])
            p0["C", i_mass] = np.array([0.1, 0.1, 0.1])
            p0["L", i_mass] = np.array([0.033, 0.033, 0.033])

        for i_pos in range(len(x0["pos"])):
            x0["pos", i_pos] = x0["pos", 0] + p0["L", i_pos] * (i_pos + 1)

        p0["p_last"] = p0["fix_point"] + np.array([1.0, 0.0, 0.0])

        self.x0 = x0
        self.p0 = p0

    def set(self, field: str, value: np.ndarray) -> None:
        self.p0[field] = value

    def set_mass_param(self, i: int, field: str, value: np.ndarray) -> None:
        self.p0[field, i] = value

    def __call__(self, p_last: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.p0["p_last"] = p_last

        self.w0 = np.concatenate([self.x0.cat.full().flatten(), 0 * self.x0.cat.full().flatten(), np.zeros(3)])
        sol = self.nlp_solver(x0=self.w0, lbg=0, ubg=0, p=self.p0.cat)

        nx = self.x0.cat.shape[0]

        x_ss = sol["x"].full()[:nx].flatten()
        u_ss = sol["x"].full()[-3:].flatten()

        return x_ss, u_ss


class Ellipsoid:
    """docstring for Ellipsoid."""

    def __init__(self, center: np.ndarray, radii: np.ndarray, seed: int = 0):
        self.center = center
        self.covariance = np.diag(1 // radii**2)
        self.ellipsoid = self.spherical_to_cartesian(phi=np.linspace(0, 2 * np.pi, 100), theta=np.linspace(0, np.pi, 100))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def plot_surface(self) -> plt.Figure:
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(*self.ellipsoid.transpose(2, 0, 1), rstride=4, cstride=4, color="b", alpha=0.75)
        return fig

    def plot_points(self, points: np.ndarray) -> plt.Figure:
        fig = self.plot_surface()
        ax = fig.get_axes()[0]
        ax.plot(points[..., 0], points[..., 1], points[..., 2], "o", color="r")
        return fig

    def spherical_to_cartesian(self, phi: np.ndarray | float, theta: np.ndarray | float) -> np.ndarray:
        x = np.outer(np.cos(phi), np.sin(theta))
        y = np.outer(np.sin(phi), np.sin(theta))
        z = np.outer(np.ones_like(phi), np.cos(theta))
        sphere = np.stack((x, y, z), axis=-1)[..., None]

        e, v = np.linalg.eig(self.covariance)
        s = v @ np.diag(np.sqrt(e)) @ v.T

        bias = np.array([10, 0, 0])
        out = (s @ sphere).squeeze(-1) + bias

        if type(phi) is float:
            return out.squeeze()

        return out

    def sample_within_range(self, phi_range: list[float, float], theta_range: list[float, float], size: int) -> np.ndarray:
        phi = self.rng.uniform(low=phi_range[0], high=phi_range[1], size=size)
        theta = self.rng.uniform(low=theta_range[0], high=theta_range[1], size=size)

        return self.spherical_to_cartesian(phi, theta)
