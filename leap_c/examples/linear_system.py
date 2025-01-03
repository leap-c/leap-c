"""
linear system
"""

from typing import Any

import casadi as cs
import numpy as np
import pygame
from acados_template import AcadosModel, AcadosOcp
from casadi.tools import struct_symSX
from leap_c.examples.render_utils import draw_arrow, draw_ellipse_from_eigen
from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.mpc import MPC
from leap_c.ocp_env import OCPEnv
from pygame import draw, gfxdraw
from scipy.linalg import solve_discrete_are


class LinearSystemMPC(MPC):
    """MPC for a system with linear dynamics and quadratic cost functions.
    The equations are given by:
        Dynamics: x_next = Ax + Bu + b
        Initial_Cost: V0
        Stage_cost: J = 0.5 * (x'Qx + u'Ru + f'cat(x,u))
        Terminal Cost: x'SOLUTION_ARE(Q,R)x
        Hard constraints:
            lbx = 0, -1
            ubx = 1, 1
            lbu = -1
            ubu = 1
        Slack:
            Both entries of x are slacked linearly, i.e., the punishment in the cost is slack_weights^T*violation.
            The slack weights are [1e2, 1e2].
    """

    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 20,
        discount_factor: float = 0.99,
        n_batch: int = 1,
    ):
        if params is None:
            params = {
                "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
                "B": np.array([[0.03125], [0.25]]),
                "Q": np.identity(2),
                "R": np.identity(1),
                "b": np.array([[0.0], [0.0]]),
                "f": np.array([[0.0], [0.0], [0.0]]),
                "V_0": np.array([1e-3]),
            }

        Q = params["Q"]
        R = params["R"]
        W = np.block([[Q, np.zeros((2, 1))], [np.zeros((1, 2)), R]])
        vals, vecs = np.linalg.eig(W)
        if not np.all(vals > 0):
            raise ValueError("Q and R should be positive definite.")

        learnable_params = learnable_params if learnable_params is not None else []
        ocp = export_parametric_ocp(
            param=params, learnable_params=learnable_params, N_horizon=N_horizon
        )
        configure_ocp_solver(ocp)

        self.given_default_param_dict = params

        super().__init__(ocp=ocp, discount_factor=discount_factor, n_batch=n_batch)


class LinearSystemOcpEnv(OCPEnv):
    """The idea is that the linear system describes a point mass that is pushed by a hidden force (noise)
    and the agent is required to learn to control the point mass in such a way that this force does not push
    the point mass over its boundaries (the constraints) while still minimizing the distance to the origin and
    minimizing control effort.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    mpc: LinearSystemMPC

    def __init__(
        self,
        mpc: LinearSystemMPC,
        dt: float = 0.1,
        max_time: float = 10.0,
        render_mode: str | None = None,
    ):
        super().__init__(
            mpc,
            dt=dt,
            max_time=max_time,
        )

        # Will be added after doing a step.
        self.current_noise = None

        # For rendering
        self.window_size = 512  # The size of the PyGame window
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.state_trajectory = None
        self.action_to_take = None

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        """Execute the dynamics of the linear system and push the resulting state with a random noise.
        If rendering is turned on, it will render the state BEFORE taking the step."""
        self.action_to_take = action
        frame = None
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            frame = self.render()
        o, r, term, trunc, info = super().step(
            action
        )  # o is the next state as np.ndarray, next parameters as MPCParameter
        info["frame"] = frame

        state = o[0].copy()
        state[0] += self.current_noise
        self.x = state
        self.current_noise = self.next_noise()
        o = (state, o[1])

        if state not in self.state_space:
            r -= 1e2
            term = True

        return o, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:  # type: ignore
        res = super().reset(seed=seed, options=options)
        self.state_trajectory = None
        self.action_to_take = None
        self.current_noise = self.next_noise()
        return res

    def next_noise(self) -> float:
        """Return the next noise to be added to the state."""
        if self._np_random is None:
            raise ValueError("First, reset needs to be called with a seed.")
        return self._np_random.uniform(-0.1, 0)

    def init_state(self):
        return self.mpc.ocp.constraints.x0.astype(dtype=np.float32)

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

        Parameters:
            planned_trajectory: The state trajectory to render, of shape (N+1, xdim).
        """
        if state_trajectory.shape != (self.mpc.N + 1, self.mpc.ocp.dims.nx):
            raise ValueError(
                f"Length of state trajectory should be {(self.mpc.N + 1, self.mpc.ocp.dims.nx)}, but is {state_trajectory.shape}."
            )
        if not np.allclose(state_trajectory[0], self.x):  # type:ignore
            raise ValueError(
                f"Initial state of state trajectory should be {self.x}, but is {state_trajectory[0]}. Are you sure this is the correct one?"
            )
        self.state_trajectory = state_trajectory

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        center = (int(self.window_size / 2), int(self.window_size / 2))
        scale = self.window_size / 2  # -1 to 1 has to span the whole window

        params = self.mpc.given_default_param_dict

        Q = params["Q"]
        P = np.array([[0, 1], [1, 0]])
        permuted_Q = (
            P @ Q @ P
        )  # Q needs to be permuted to correspond to how the state is drawn.
        # Draw the optimum as black circle and ellipsoids representing the curvature of the cost function.
        # NOTE: The linear part "f" is ignored
        gfxdraw.aacircle(canvas, center[0], center[1], 8, (0, 0, 0))
        gfxdraw.filled_circle(canvas, center[0], center[1], 8, (0, 0, 0))
        draw_ellipse_from_eigen(canvas, center, 64, permuted_Q)
        draw_ellipse_from_eigen(canvas, center, 128, permuted_Q)
        draw_ellipse_from_eigen(canvas, center, 192, permuted_Q)
        draw_ellipse_from_eigen(canvas, center, 256, permuted_Q)

        state_in_window = self.x * scale + center  # type:ignore
        # Make the ball stay within the halfplane on top of the screen.
        # This means making [-1, 1] the constraints on the x axis (left-right)
        # and [0, 1] the constraints on the y axis (down-up).
        # High y coordinate is low on the screen.
        state_y = self.window_size - int(state_in_window[0])
        state_x = int(state_in_window[1])
        # Draw agent as circle
        gfxdraw.aacircle(canvas, state_x, state_y, 6, (0, 0, 200))
        gfxdraw.filled_circle(canvas, state_x, state_y, 6, (0, 0, 200))

        # Draw the state trajectory if it exists
        if self.state_trajectory is not None:
            planxs = self.state_trajectory * scale + center

            bound = int(self.window_size / 2)
            for i, planx in enumerate(planxs):
                if np.any(np.abs(planx) > bound):
                    # State out of window, dont render it
                    continue
                gfxdraw.pixel(canvas, int(planx[0]), int(planx[1]), (0, 0, 255))
            self.state_trajectory = None  # Don't render the same trajectory again, except it is set explicitly

        # Draw constraints
        gfxdraw.hline(canvas, 0, self.window_size, center[1], (255, 0, 0))
        gfxdraw.hline(canvas, 0, self.window_size, center[1] + 1, (255, 0, 0))

        # Double line-thickness for better look
        gfxdraw.hline(canvas, 0, self.window_size, 0, (255, 0, 0))
        gfxdraw.hline(canvas, 0, self.window_size, 1, (255, 0, 0))

        gfxdraw.vline(canvas, 0, 0, self.window_size, (255, 0, 0))
        gfxdraw.vline(canvas, 1, 0, self.window_size, (255, 0, 0))

        gfxdraw.vline(canvas, self.window_size - 2, 0, self.window_size, (255, 0, 0))
        gfxdraw.vline(canvas, self.window_size - 1, 0, self.window_size, (255, 0, 0))

        A = params["A"]
        B = params["B"]
        b = params["b"]

        # Draw a green line to the transformed state
        lin_transformed_state = A @ self.x
        destination = (
            int(scale * lin_transformed_state[1] + center[0]),
            self.window_size - int(scale * lin_transformed_state[0] + center[1]),
        )
        draw.aaline(canvas, (0, 255, 0), (state_x, state_y), destination, 2)

        # Draw the action the agent takes as blue (as agent) line#
        if self.action_to_take is None:
            raise ValueError("action_to_be_taken should be set before calling render.")
        destination_old = destination
        displacement_action = B @ self.action_to_take
        destination = (
            destination_old[0] + int(scale * displacement_action[1]),
            destination_old[1] - int(scale * displacement_action[0]),
        )
        draw.aaline(canvas, (0, 0, 255), destination_old, destination, 2)

        # Draw the displacement by "b" as green line
        destination_old = destination
        destination = (
            destination_old[0] + int(scale * b[1].item()),
            destination_old[1] - int(scale * b[0].item()),
        )
        draw.aaline(canvas, (0, 255, 0), destination_old, destination, 2)

        # Draw the noise as red arrow
        arrow_size = 8
        destination_old = destination
        arrow_length = int(abs(scale * self.current_noise))  # type:ignore
        destination = (
            destination_old[0],
            destination_old[1] - arrow_length,  # type:ignore
        )
        color = (255, 0, 0)
        draw_arrow(
            canvas,
            destination_old,
            arrow_length,
            arrow_size,
            arrow_size,
            0,
            color,  # type:ignore
            2,  # type:ignore
        )

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


def disc_dyn_expr(model: AcadosModel):
    """
    Define the discrete dynamics function expression.
    """
    x = model.x
    u = model.u

    param = find_param_in_p_or_p_global(["A", "B", "b"], model)

    return param["A"] @ x + param["B"] @ u + param["b"]


def cost_expr_ext_cost(model: AcadosModel):
    """
    Define the external cost function expression.
    """
    x = model.x
    u = model.u
    param = find_param_in_p_or_p_global(["Q", "R", "f"], model)

    return 0.5 * (
        cs.transpose(x) @ param["Q"] @ x
        + cs.transpose(u) @ param["R"] @ u
        + cs.transpose(param["f"]) @ cs.vertcat(x, u)
    )


def cost_expr_ext_cost_0(model: AcadosModel):
    """
    Define the external cost function expression at stage 0.
    """
    param = find_param_in_p_or_p_global(["V_0"], model)

    return param["V_0"] + cost_expr_ext_cost(model)


def cost_expr_ext_cost_e(model: AcadosModel, param: dict[str, np.ndarray]):
    """
    Define the external cost function expression at the terminal stage as the solution of the discrete-time algebraic Riccati
    equation.
    """

    x = model.x

    return 0.5 * cs.mtimes(
        [
            cs.transpose(x),
            solve_discrete_are(param["A"], param["B"], param["Q"], param["R"]),
            x,
        ]
    )


def configure_ocp_solver(
    ocp: AcadosOcp,
):
    ocp.solver_options.tf = ocp.solver_options.N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1

    # Set nominal parameters. Could be done at AcadosOcpSolver initialization?


def export_parametric_ocp(
    param: dict[str, np.ndarray],
    cost_type="EXTERNAL",
    name: str = "lti",
    learnable_params: list[str] | None = None,
    N_horizon=20,
) -> AcadosOcp:
    """
    Export a parametric optimal control problem (OCP) for a discrete-time linear time-invariant (LTI) system.

    Parameters:
        param:
            Dictionary containing the parameters of the system. Keys should include "A", "B", "b", "V_0", and "f".
        cost_type:
            Type of cost function to use. Options are "LINEAR_LS" or "EXTERNAL".
        name:
            Name of the model.
        learnable_params:
            List of parameters that should be learnable.
        N_horizon:
            The length of the prediction horizon.

    Returns:
        AcadosOcp
            An instance of the AcadosOcp class representing the optimal control problem.
    """
    if learnable_params is None:
        learnable_params = []
    ocp = AcadosOcp()

    ocp.model.name = name

    ocp.dims.nx = 2
    ocp.dims.nu = 1

    ocp.model.x = cs.SX.sym("x", ocp.dims.nx)  # type:ignore
    ocp.model.u = cs.SX.sym("u", ocp.dims.nu)  # type:ignore

    ocp.solver_options.N_horizon = N_horizon

    ocp = translate_learnable_param_to_p_global(
        nominal_param=param,
        learnable_param=learnable_params,
        ocp=ocp,
    )

    ocp.model.disc_dyn_expr = disc_dyn_expr(ocp.model)

    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0(ocp.model)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = cost_expr_ext_cost(ocp.model)

    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = cost_expr_ext_cost_e(ocp.model, param)

    ocp.constraints.idxbx_0 = np.array([0, 1])
    ocp.constraints.lbx_0 = np.array([-1.0, -1.0])
    ocp.constraints.ubx_0 = np.array([1.0, 1.0])

    ocp.constraints.idxbx = np.array([0, 1])
    ocp.constraints.lbx = np.array([-0.0, -1.0])
    ocp.constraints.ubx = np.array([+1.0, +1.0])
    ocp.constraints.x0 = np.array([0.5, 0.0])  # Default constraints.x0?

    # Only slack first dimension
    # ocp.constraints.idxsbx = np.array([0])
    # ocp.cost.zl = np.array([1e2])
    # ocp.cost.zu = np.array([1e2])
    # ocp.cost.Zl = np.diag([0])
    # ocp.cost.Zu = np.diag([0])
    # Slack both dimensions
    ocp.constraints.idxsbx = np.array([0, 1])
    ocp.cost.zl = np.ones((2,)) * 1e2
    ocp.cost.zu = np.ones((2,)) * 1e2
    ocp.cost.Zl = np.diag([0, 0])
    ocp.cost.Zu = np.diag([0, 0])

    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([-1.0])
    ocp.constraints.ubu = np.array([+1.0])

    # TODO: Make a PR to acados to allow struct_symSX | struct_symMX in acados_template and then concatenate there
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []
    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp
