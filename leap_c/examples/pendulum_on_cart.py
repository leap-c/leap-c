from typing import Any

import casadi as ca
import numpy as np
import pygame
from acados_template import AcadosModel, AcadosOcp
from casadi.tools import struct_symSX
from gymnasium.spaces import Box
from leap_c.examples.render_utils import draw_arrow
from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.mpc import MPC
from leap_c.ocp_env import OCPEnv
from pygame import gfxdraw


class PendulumOnCartMPC(MPC):
    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 20,
        T_horizon: float = 1.0,
        Fmax: float = 80.0,
        exact_hess_dyn: bool = True,
        discount_factor: float = 0.99,
        n_batch: int = 1,
        
    ):
        if params is None:
            params = {
                "M": np.array([1.0]),  # mass of the cart [kg]
                "m": np.array([0.1]),  # mass of the ball [kg]
                "g": np.array([9.81]),  # gravity constant [m/s^2]
                "l": np.array([0.8]),  # length of the rod [m]
                "Q": np.diag([2e3, 2e3, 1e-2, 1e-2]),  # state cost
                "R": np.diag([2e-1]),  # control cost
            }

        ocp = export_parametric_ocp(
            nominal_param=params,
            learnable_param=learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
            Fmax=Fmax,
        )
        configure_ocp_solver(ocp, exact_hess_dyn)

        self.given_default_param_dict = params

        super().__init__(ocp=ocp, discount_factor=discount_factor, n_batch=n_batch)


class PendulumOnCartOcpEnv(OCPEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    mpc: PendulumOnCartMPC

    def __init__(
        self,
        mpc: PendulumOnCartMPC,
        dt: float = 0.1,
        max_time: float = 10.0,
        noise_magnitude: float = 0.1,
        render_mode: str | None = None,
    ):
        super().__init__(
            mpc,
            dt=dt,
            max_time=max_time,
        )
        low = self.state_space.low
        high = self.state_space.high
        low[1] = -2 * np.pi
        high[1] = 2 * np.pi
        self.state_space = Box(low=low, high=high, dtype=self.state_space.dtype)  # type:ignore
        self.noise_mag = noise_magnitude

        # For rendering
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self.pos_trajectory = None
        self.pole_end_trajectory = None
        self.screen_width = 600
        self.screen_height = 400
        self.window = None
        self.clock = None
        self.action_to_take = None

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        """Execute the dynamics of the pendulum on cart and add random noise to the resulting cart velocity."""
        self.action_to_take = action
        frame = None
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            frame = self.render()
        o, r, term, trunc, info = super().step(
            action
        )  # o is the next state as np.ndarray, next parameters as MPCParameter
        info["frame"] = frame
        state = o[0].copy()
        state[2] += self.current_noise
        theta = state[1]
        if theta > 2 * np.pi:
            theta -= 2 * np.pi
        elif theta < -2 * np.pi:
            theta += 2 * np.pi
        state[1] = theta
        self.x = state
        self.current_noise = self.next_noise()
        o = (state, o[1])

        info["mpc_cost"] = -r
        r = abs(np.pi - (abs(theta))) / (10 * np.pi)  # Reward for swingup; Max: 0.1
        if state not in self.state_space:
            term = True  # Just terminating should be enough punishment when reward is positive

        return o, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:  # type: ignore
        res = super().reset(seed=seed, options=options)
        self.current_noise = self.next_noise()
        self.pos_trajectory = None
        self.pole_end_trajectory = None
        return res

    def init_state(self):
        return np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float32)

    def next_noise(self) -> float:
        """Return the next noise to be added to the state."""
        if self._np_random is None:
            raise ValueError("First, reset needs to be called with a seed.")
        return self._np_random.uniform(-1 * self.noise_mag, 0)

    def include_this_state_trajectory_to_rendering(self, state_trajectory: np.ndarray):
        """Meant for setting a state trajectory for rendering.
        If a state trajectory is not set before the next call of render,
        the rendering will not render a state trajectory.
        NOTE: The record_video wrapper of gymnasium will call render() AFTER every step.
        This means if you use the wrapper,
        make a step,
        calculate action and state trajectory from the observations,
        and input the state trajectory with this function before taking the next step,
        the picture being rendered after this next step will be showing the trajectory planned BEFORE DOING the step.
        """
        self.pos_trajectory = []
        self.pole_end_trajectory = []
        length = self.mpc.given_default_param_dict["l"]
        for x in state_trajectory:
            self.pos_trajectory.append(x[0])  # Only take coordinate
            self.pole_end_trajectory.append(self.calc_pole_end(x[0], x[1], length))

    def calc_pole_end(
        self, x_coord: float, theta: float, length: float
    ) -> tuple[float, float]:
        # NOTE: The minus is necessary because theta is seen as counterclockwise
        pole_x = x_coord - length * np.sin(theta)
        pole_y = length * np.cos(theta)
        return pole_x, pole_y

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        params = self.mpc.given_default_param_dict
        world_width = self.mpc.ocp.constraints.ubx[0] - self.mpc.ocp.constraints.lbx[0]
        center = (int(self.screen_width / 2), int(self.screen_height / 2))
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (params["l"])
        cartwidth = 50.0
        cartheight = 30.0
        axleoffset = cartheight / 4.0
        ground_height = 180

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((255, 255, 255))

        # ground
        gfxdraw.hline(canvas, 0, self.screen_width, ground_height, (0, 0, 0))

        # cart
        left, right, top, bot = (
            -cartwidth / 2,
            cartwidth / 2,
            cartheight / 2,
            -cartheight / 2,
        )

        pos = self.x[0]  # type:ignore
        theta = self.x[1]  # type:ignore
        cartx = pos * scale + center[0]
        cart_coords = [(left, bot), (left, top), (right, top), (right, bot)]
        cart_coords = [(c[0] + cartx, c[1] + ground_height) for c in cart_coords]
        gfxdraw.aapolygon(canvas, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(canvas, cart_coords, (0, 0, 0))

        # Draw the action and noise arrow
        Fmax = self.mpc.ocp.constraints.ubu.item()
        if self.action_to_take is None:
            raise ValueError(
                "action_to_take is None, but it should be set before rendering."
            )
        action_length = abs(int(self.action_to_take.item() / Fmax * scale))

        if self.action_to_take.item() > 0:  # Draw on the right side
            action_origin = (int(cartx + right), ground_height)
            action_rotate_deg = 270
            if self.current_noise > 0:
                noise_origin = (action_origin[0] + action_length, action_origin[1])
                noise_rotate_deg = action_rotate_deg
            else:
                noise_origin = (int(cartx + left), ground_height)
                noise_rotate_deg = 90
        else:  # Draw on the left side
            action_origin = (int(cartx + left), ground_height)
            action_rotate_deg = 90
            if self.current_noise < 0:
                noise_origin = (action_origin[0] - action_length, action_origin[1])
                noise_rotate_deg = action_rotate_deg
            else:
                noise_origin = (int(cartx + right), ground_height)
                noise_rotate_deg = 270
        head_size = 8
        draw_arrow(
            canvas,
            action_origin,
            action_length,
            head_size,
            head_size,
            action_rotate_deg,
            color=(0, 0, 255),
            width_line=3,
        )
        noise_length = abs(int(self.current_noise / Fmax * scale))
        draw_arrow(
            canvas,
            noise_origin,
            noise_length,
            head_size,
            head_size,
            noise_rotate_deg,
            color=(255, 0, 0),
            width_line=3,
        )

        # pole
        left, right, top, bot = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(left, bot), (left, top), (right, top), (right, bot)]:
            coord = pygame.math.Vector2(coord).rotate_rad(theta)
            coord = (coord[0] + cartx, coord[1] + ground_height + axleoffset)
            pole_coords.append(coord)
        pole_color = (202, 152, 101)
        gfxdraw.aapolygon(canvas, pole_coords, pole_color)
        gfxdraw.filled_polygon(canvas, pole_coords, pole_color)

        # Axle of pole
        gfxdraw.aacircle(
            canvas,
            int(cartx),
            int(ground_height + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            canvas,
            int(cartx),
            int(ground_height + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        # Draw the planned trajectory if it exists
        if self.pos_trajectory is not None:
            if self.pole_end_trajectory is None:
                raise AttributeError(
                    "Why is pole_end_trajectory None, but pos_trajectory isn't?"
                )
            planxs = [int(x * scale + center[0]) for x in self.pos_trajectory]
            plan_pole_end = [
                (
                    int(x * scale + center[0]),
                    int(ground_height + axleoffset + y * scale - polewidth / 2),
                )
                for x, y in self.pole_end_trajectory
            ]

            # Draw the positions offset in the y direction for better visibility
            for i, planx in enumerate(planxs):
                if abs(planx) > self.screen_width:
                    # Dont render out of bounds
                    continue
                gfxdraw.pixel(canvas, int(planx), int(ground_height + i), (255, 5, 5))
            for i, plan_pole_end in enumerate(plan_pole_end):
                if abs(plan_pole_end[0]) > self.screen_width:
                    # Dont render out of bounds
                    continue
                gfxdraw.pixel(
                    canvas, int(plan_pole_end[0]), int(plan_pole_end[1]), (5, 255, 5)
                )

        canvas = pygame.transform.flip(canvas, False, True)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))  # type:ignore
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])  # type:ignore

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def configure_ocp_solver(
    ocp: AcadosOcp,
    exact_hess_dyn: bool
):
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"  # "GAUSS_NEWTON"
    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = True


def f_expl_expr(model: AcadosModel) -> ca.SX:
    p = find_param_in_p_or_p_global(["M", "m", "g", "l"], model)

    M = p["M"]
    m = p["m"]
    g = p["g"]
    l = p["l"]

    theta = model.x[1]
    v1 = model.x[2]
    dtheta = model.x[3]

    F = model.u[0]

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m * cos_theta * cos_theta
    f_expl = ca.vertcat(
        v1,
        dtheta,
        (-m * l * sin_theta * dtheta * dtheta + m * g * cos_theta * sin_theta + F)
        / denominator,
        (
            -m * l * cos_theta * sin_theta * dtheta * dtheta
            + F * cos_theta
            + (M + m) * g * sin_theta
        )
        / (l * denominator),
    )

    return f_expl


def disc_dyn_expr(model: AcadosModel, dt: float) -> ca.SX:
    f_expl = f_expl_expr(model)

    x = model.x
    u = model.u

    # discrete dynamics via RK4
    p = ca.vertcat(
        *list(find_param_in_p_or_p_global(["M", "m", "g", "l"], model).values())
    )

    ode = ca.Function("ode", [x, u, p], [f_expl])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)
    k3 = ode(x + dt / 2 * k2, u, p)
    k4 = ode(x + dt * k3, u, p)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def cost_expr_ext_cost(model: AcadosModel) -> ca.SX:
    x = model.x
    u = model.u

    p = find_param_in_p_or_p_global(["Q", "R"], model)
    Q = p["Q"]
    R = p["R"]

    return 0.5 * (
        ca.mtimes([ca.transpose(x), Q, x]) + ca.mtimes([ca.transpose(u), R, u])
    )


def cost_expr_ext_cost_0(model: AcadosModel) -> ca.SX:
    return cost_expr_ext_cost(model)


def cost_expr_ext_cost_e(model: AcadosModel) -> ca.SX:
    x = model.x
    Q = find_param_in_p_or_p_global(["Q"], model)["Q"]

    return 0.5 * (ca.mtimes([ca.transpose(x), Q, x]))


def export_parametric_ocp(
    nominal_param: dict[str, np.ndarray],
    cost_type: str = "EXTERNAL",
    name: str = "pendulum_on_cart",
    learnable_param: list[str] | None = None,
    Fmax: float = 80.0,
    N_horizon: int = 50,
    tf: float = 2.0,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = tf
    dt = ocp.solver_options.tf / ocp.solver_options.N_horizon

    ######## Model ########
    ocp.model.name = name

    ocp.dims.nx = 4
    ocp.dims.nu = 1

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_param, learnable_param=learnable_param, ocp=ocp
    )

    ocp.model.disc_dyn_expr = disc_dyn_expr(model=ocp.model, dt=dt)

    ######## Cost ########
    if cost_type == "EXTERNAL":
        ocp.cost.cost_type_0 = cost_type
        ocp.model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0(ocp.model)

        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(ocp.model)

        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(ocp.model)
    else:
        raise ValueError(f"Cost type {cost_type} not supported.")
        # TODO: Implement NONLINEAR_LS with y_expr = sqrt(Q) * x and sqrt(R) * u

    ######## Constraints ########
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([-2.5])
    ocp.constraints.ubx = -ocp.constraints.lbx
    ocp.constraints.idxbx = np.array([0])

    # #############################
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp
