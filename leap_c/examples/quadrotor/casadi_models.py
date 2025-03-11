from typing import Dict
import casadi as cs
import numpy as np

from leap_c.examples.quadrotor.utils import quaternion_multiply_casadi, quaternion_rotate_vector_casadi, read_from_yaml


def get_rhs_quadrotor(params: Dict, model_fidelity: str = "low", scale_disturbances: int = 1, sym_params: bool = True):
    """
    Returns the right-hand side of the quadrotor dynamics.
    We model 4 rotors which are controlled by the motor speeds.
        params: Dict containing the parameters of the quadrotor model.
    """

    # states
    x_pos = cs.SX.sym("x_pos", 3)
    x_quat = cs.SX.sym("x_quat", 4)
    x_vel = cs.SX.sym("x_vel", 3)
    x_rot = cs.SX.sym("x_rot", 3)

    if sym_params:
        p_mass = cs.SX.sym("p_mass", 1)
        p = [p_mass]
    else:
        p_mass = params["mass"]
        p = None

    x = cs.vertcat(x_pos, x_quat, x_vel, x_rot)

    # controls
    u_motor_speeds = cs.SX.sym("u_motor_speeds", 4)

    # controls-2-thrust
    fz_props = u_motor_speeds ** 2 * params["motor_cl"]
    f_prop = cs.vertcat(0, 0, cs.sum1(fz_props))

    # controls-2-torques
    tau_prop = cs.vertcat(0, 0, 0)
    for i in range(4):
        tau = cs.vertcat(0, 0, params["motor_kappa"][i] * params["motor_cl"] * u_motor_speeds[i] ** 2)
        fz_prop = cs.vertcat(0, 0, u_motor_speeds[i] ** 2 * params["motor_cl"])
        tau_prop += tau + cs.cross(params["rotor_loc"][i], fz_prop)

    # residuals
    if model_fidelity == "low":
        f_res = cs.vertcat(0, 0, 0)
    elif model_fidelity == "high":
        mean_sqr_motor_speed = cs.sum1(cs.vertcat(*[u_motor_speeds[i] ** 2 for i in range(4)])) / 4
        v_xy = cs.sqrt(x_vel[0] ** 2 + x_vel[1] ** 2)
        v_x, v_y, v_z = x_vel[0], x_vel[1], x_vel[2]
        fres_x = p_mass * (params["cx1_fres"] * v_x +
                                   params["cx2_fres"] * v_x * cs.fabs(v_x) +
                                   params["cx3_fres"] * v_x * mean_sqr_motor_speed)
        fres_y = p_mass * (params["cy1_fres"] * v_y +
                                   params["cy2_fres"] * v_y * cs.fabs(v_y) +
                                   params["cy3_fres"] * v_y * mean_sqr_motor_speed)
        fres_z = p_mass * (params["cz1_fres"] * v_z +
                                   params["cz2_fres"] * v_z ** 3 +
                                   params["cz3_fres"] * v_xy +
                                   params["cz4_fres"] * v_xy ** 2 +
                                   params["cz5_fres"] * v_xy * mean_sqr_motor_speed +
                                   params["cz6_fres"] * v_xy * mean_sqr_motor_speed * v_z +
                                   params["cz7_fres"] * mean_sqr_motor_speed)
        f_res = cs.vertcat(fres_x, fres_y, fres_z) * scale_disturbances  # scale_disturbances0.003 #todo remove
    else:
        raise NotImplementedError(f"Model fidelity {model_fidelity} not implemented")

    # derivatives
    dx_pos = x_vel
    dx_quat = quaternion_multiply_casadi(0.5 * x_quat, cs.vertcat(0, x_rot))

    acc_gravity = cs.vertcat(0, 0, -params["gravity"])
    acc_thrust = 1 / p_mass * quaternion_rotate_vector_casadi(x_quat, f_prop + f_res)
    dx_vel = acc_gravity + acc_thrust

    intertia = np.array(params["inertia"])
    drot = np.linalg.inv(intertia) @ (tau_prop - cs.cross(x_rot, intertia @ x_rot))

    rhs = cs.vertcat(dx_pos, dx_quat, dx_vel, drot)

    u = u_motor_speeds

    if sym_params:
        rhs_func = cs.Function("f_ode", [x, u, *p], [rhs])
    else:
        rhs_func = cs.Function("f_ode", [x, u], [rhs])

    return x, u, p, rhs, rhs_func


def integrate_one_step(rhs_func, x0, u, dt, method="RK4"):
    """
    Integrates one time step of a differential equation using Euler or RK4.

    Parameters:
    - rhs: casadi.SX or casadi.MX, the right-hand side function f(x, u)
    - x0: casadi.SX or casadi.MX, initial state x0
    - u: casadi.SX or casadi.MX, control input u
    - dt: float, time step size
    - method: str, integration method ("Euler" or "RK4")

    Returns:
    - x_next: casadi.SX or casadi.MX, state at the next time step
    """

    if method == "Euler":
        x_next = x0 + dt * rhs_func(x0, u)
    elif method == "RK4":
        k1 = rhs_func(x0, u)
        k2 = rhs_func(x0 + (dt / 2) * k1, u)
        k3 = rhs_func(x0 + (dt / 2) * k2, u)
        k4 = rhs_func(x0 + dt * k3, u)
        x_next = x0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        raise ValueError("Method must be 'Euler' or 'RK4'")

    return x_next


def simulate_trajectory(rhs_func, x0, u_seq, dt, method="RK4"):
    """
    Simulates a trajectory given an initial state and control sequence.

    Parameters:
    - rhs_func: casadi.Function, right-hand side function f(x, u) -> dx/dt
    - x0: np.array, initial state (n,)
    - u_seq: np.array, sequence of control inputs (N, m)
    - dt: float, time step size
    - method: str, integration method ("Euler" or "RK4")

    Returns:
    - X: np.array, trajectory of states (N+1, n)
    """
    N = u_seq.shape[0]  # Number of control steps
    n = x0.shape[0]  # State dimension
    X = np.zeros((N + 1, n))  # State trajectory storage
    X[0, :] = x0  # Store initial state

    # Simulate trajectory
    for i in range(N):
        X[i + 1, :] = integrate_one_step(rhs_func, X[i, :], u_seq[i, :], dt).full().flatten()

    return X


# execute as main to test
if __name__ == "__main__":
    model_params = read_from_yaml("model_params.yaml")

    # Define initial state and control sequence
    x0 = np.array([0] * 13)
    x0[3] = 1
    N = 200
    u_seq = np.ones((N, 4)) * 1000
    u_seq[:, 0] += 0.1
    u_seq[:, 3] += 0.2
    u_seq[:, 1] -= 0.1
    td = 0.0025

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))  # 3 rows, 1 column
    for model_fidelity in ["low", "high"]:
        x, u, rhs, rhs_func = get_rhs_quadrotor(model_params, model_fidelity=model_fidelity)
        X = simulate_trajectory(rhs_func, x0, u_seq, td, method="RK4")
        t = np.arange(0, (N) * td + 1e-6, td)

        # create 3x1 sub
        axes[0].plot(t, X[:, 0], label="x-" + model_fidelity)
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("position (m)")
        axes[1].plot(t, X[:, 1], label="y-" + model_fidelity)
        axes[0].set_xlabel("time (s)")
        axes[2].plot(t, X[:, 2], label="z-" + model_fidelity)
        axes[0].set_xlabel("time (s)")
    plt.legend()
    plt.show()
