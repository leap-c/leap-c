import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

from leap_c.examples.quadrotor.casadi_models import get_rhs_quadrotor, integrate_one_step
from leap_c.examples.quadrotor.utils import read_from_yaml


class QuadrotorStop(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self,
            render_mode: str | None = None,
    ):
        self.weight_velocity = 1
        self.weight_constraint_violation = 1e5
        self.fig, self.axes = None, None

        self.model_params = read_from_yaml("./examples/quadrotor/model_params.yaml")

        self.sim_params = {
            "dt": 0.005,
            "t_sim": 2.0
        }
        x, u, p, rhs, self.rhs_func = get_rhs_quadrotor(self.model_params,
                                                        model_fidelity="high",
                                                        scale_disturbances=0.001,
                                                        sym_params=False)

        x_high = np.array(
            [
                100.0, 100.0, 100.0,  # position
                1.5, 1.5, 1.5, 1.5,  # quaternion
                50, 50, 50,  # velocity
                50, 50, 50,  # angular velocity
            ],
            dtype=np.float32,
        )
        x_low = np.array(
            [
                -100.0, -100.0, -100.0,  # position
                -1, -1, -1, -1,  # quaternion
                -50, -50, -50,  # velocity
                -50, -50, -50,  # angular velocity
            ],
            dtype=np.float32,
        )
        self.x_low, self.x_high = x_low, x_high

        u_high = np.array([self.model_params["motor_omega_max"]] * 4, dtype=np.float32)
        u_low = np.array([0.0] * 4, dtype=np.float32)

        self.action_space = spaces.Box(u_low, u_high, dtype=np.float32)
        self.observation_space = spaces.Box(x_low, x_high, dtype=np.float32)

        self.reset_needed = True
        self.trajectory, self.time_steps, self.action_trajectory = None, None, None
        self.t = 0
        self.x = None

        # For rendering
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute the dynamics of the pendulum on cart."""
        if self.reset_needed:
            raise Exception("Call reset before using the step method.")
        dt = self.sim_params["dt"]

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.x = integrate_one_step(self.rhs_func, self.x, action, dt).full().flatten()
        self.x[3:3 + 4] = self.x[3:3 + 4] / np.linalg.norm(self.x[3:3 + 4])
        self.t += dt
        self.trajectory = [self.x] if self.trajectory is None else self.trajectory + [self.x]
        self.time_steps = [self.t] if self.time_steps is None else self.time_steps + [self.t]
        self.action_trajectory = [action] if self.action_trajectory is None else self.action_trajectory + [action]

        term = False
        trunc = False
        # (all(self.x < self.x_high) and all(self.x > self.x_low)) and
        if not bool(np.isnan(self.x).sum()) and (self.x[7:10].sum() <= 1000) and (self.x[7:10].sum() >= -1000):
            if np.isnan(self.x).sum() >= 1:
                print("Bigger 1, should not be")
            r = - dt * (self.weight_velocity * np.linalg.norm(self.x[7:10]) +
                        self.weight_constraint_violation * max(self.x[2] - self.model_params["bound_z"], 0))

        else:
            print(f"Truncation at time {self.t} with state {self.x}")
            r = -1e5
            trunc = True
            term = True

        if self.t >= self.sim_params["t_sim"]:
            term = True
        print(r)
        self.reset_needed = trunc or term

        return self.x, r, term, trunc, {}

    def reset(
            self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        if self._np_random is None:
            raise RuntimeError("The first reset needs to be called with a seed.")
        self.t = 0

        vx = np.random.uniform(0, 3)
        vy = np.random.uniform(0, 3)
        self.x = np.array([0, 0, 0,
                           1, 0, 0, 0,
                           vx, vy, 0,
                           0, 0, 0], dtype=np.float32)
        self.reset_needed = False

        self.trajectory, self.time_steps, self.action_trajectory = [self.x], [self.t], None
        return self.x, {}

    def render(self, interactive: bool = True):

        if self.fig is None:
            if interactive:
                plt.ion()
            self.fig, self.axes = plt.subplots(3, 3, figsize=(12, 9), sharey='row')
        fig, axes = self.fig, self.axes

        if int(self.time_steps[-1]/self.sim_params["dt"]) % 10 != 0 and interactive:
            return
        for ax in axes.flatten():
            ax.clear()
        trajectory = np.array(self.trajectory)
        axes[0, 0].plot(self.time_steps, trajectory[:, 0])
        axes[0, 0].set_title(r"position $p_x$")
        axes[0, 0].set_xlabel("time (s)")
        axes[0, 0].set_ylabel("position (m)")

        axes[0, 1].plot(self.time_steps, trajectory[:, 1])
        axes[0, 1].set_title(r"position $p_y$")
        axes[0, 1].set_xlabel("time (s)")

        axes[0, 2].plot(self.time_steps, trajectory[:, 2])
        axes[0, 2].hlines(self.model_params["bound_z"], 0, self.sim_params["t_sim"], colors='tab:red',
                          linestyles='dashed')
        axes[0, 2].set_title(r"position $p_z$")
        axes[0, 2].set_xlabel("time (s)")

        axes[1, 0].plot(self.time_steps, trajectory[:, 7])
        axes[1, 0].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
        axes[1, 0].set_xlabel("time (s)")
        axes[1, 0].set_ylabel(r"velocity ($\frac{m}{s}$)")
        axes[1, 0].set_title(r"velocity $v_x$")

        axes[1, 1].plot(self.time_steps, trajectory[:, 8])
        axes[1, 1].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
        axes[1, 1].set_xlabel("time (s)")
        axes[1, 1].set_title(r"velocity $v_y$")

        axes[1, 2].plot(self.time_steps, trajectory[:, 9])
        axes[1, 2].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
        axes[1, 2].set_xlabel("time (s)")
        axes[1, 2].set_title(r"velocity $v_z$")

        axes[2, 0].plot(self.time_steps, trajectory[:, 10])
        axes[2, 0].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
        axes[2, 0].set_xlabel("time (s)")
        axes[2, 0].set_ylabel(r"angular velocity ($\frac{m}{s}$)")
        axes[2, 0].set_title(r"angular velocity $\omega_x$")

        axes[2, 1].plot(self.time_steps, trajectory[:, 11])
        axes[2, 1].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
        axes[2, 1].set_xlabel("time (s)")
        axes[2, 1].set_title(r"angular velocity $\omega_y$")

        axes[2, 2].plot(self.time_steps, trajectory[:, 12])
        axes[2, 2].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
        axes[2, 2].set_xlabel("time (s)")
        axes[2, 2].set_title(r"angular velocity $\omega_z$")
        plt.tight_layout()
        if interactive:
            plt.pause(0.001)
        else:
            plt.show()

# execute as main to test
if __name__ == "__main__":
    env = QuadrotorStop()
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action = [970.437] * 4
        obs, reward, done, _, _ = env.step(action)
        print(f"reward:{reward}")
    env.render()
