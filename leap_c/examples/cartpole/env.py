import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gymnasium.envs.classic_control import utils as gym_utils
from typing import Optional


class CartPoleEnv(gym.Env):
    """
    An environment of a pendulum on a cart meant for swinging
    the pole into an upright position and holding it there.

    Observation Space:
    ------------------

    The observation is a `ndarray` with shape `(4,)` and dtype `np.float32`
    representing the state of the system.

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Pole Angle (theta)    | -2pi                | 2pi               |
    | 2   | Cart Velocity         | -Inf                | Inf               |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    NOTE: Like in the original CartPole environment, the range above for the cart position denotes
    the possible range of the cart's center of mass in the observation space,
    but the episode terminates if it leaves the interval (-2.4, 2.4) already.
    NOTE: The pole angle is actually bounded between -2pi and 2pi by always adding/subtracting
    (in the negative / in the positive case) the highest multiple of 2pi
    until the pole angle is within the bounds again.
    NOTE: Contrary to the original CartPoleEnv, the state space here is arranged like
    [x, theta, dx, dtheta] instead of [x, dx, theta, dtheta].
    NOTE: A positive angle theta is interpreted as counterclockwise rotation.


    Action Space:
    -------------

    The action is a `ndarray` with shape `(1,)` which can take values in the range (-Fmax, Fmax) indicating the direction
    of the fixed force the cart is pushed with (action > 0 -> push right).


    Reward:
    -------
    Since this is an environment for the swingup task, the agent achieves maximum reward when the pole
    is upright (theta = 0) and minimum reward when the pole is hanging down (theta = pi or theta = -pi).
    More precisely, the reward in a step is bounded between 0 and 0.1, given by
    r = abs(pi - (abs(theta))) / (10 * pi)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
    ):
        self.gravity = 9.81
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.8
        self.Fmax = 80.0
        self.dt = 0.05
        self.max_time = 10.0
        self.x_threshold = 2.4

        def f_explicit(
            x,
            u,
            g=self.gravity,
            M=self.masscart,
            m=self.masspole,
            l=self.length,  # noqa E741
        ):
            _, theta, dx, dtheta = x
            F = u.item()
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            denominator = M + m - m * cos_theta * cos_theta
            return np.array(
                [
                    dx,
                    dtheta,
                    (
                        -m * l * sin_theta * dtheta * dtheta
                        + m * g * cos_theta * sin_theta
                        + F
                    )
                    / denominator,
                    (
                        -m * l * cos_theta * sin_theta * dtheta * dtheta
                        + F * cos_theta
                        + (M + m) * g * sin_theta
                    )
                    / (l * denominator),
                ]
            )

        def rk4_step(f, x, u, h):
            k1 = f(x, u)
            k2 = f(x + 0.5 * h * k1, u)
            k3 = f(x + 0.5 * h * k2, u)
            k4 = f(x + h * k3, u)
            return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        self.integrator = lambda x, u, t: rk4_step(f_explicit, x, u, t)

        high = np.array(
            [
                self.x_threshold * 2,
                2 * np.pi,
                10,
                10,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(-self.Fmax, self.Fmax, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.reset_needed = True
        self.t = 0
        self.x = None

        # For rendering
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self.pos_trajectory = None
        self.pole_end_trajectory = None
        self.x_trajectory = None
        self.screen_width = 600
        self.screen_height = 400
        self.window = None
        self.clock = None

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute the dynamics of the pendulum on cart."""
        if self.reset_needed:
            raise Exception("Call reset before using the step method.")
        self.x = self.integrator(self.x, action, self.dt)
        self.x_trajectory.append(self.x)  # type: ignore
        self.t += self.dt
        theta = self.x[1]
        if theta > 2 * np.pi:
            theta = theta % 2 * np.pi
        elif theta < -2 * np.pi:
            theta = -(-theta % 2 * np.pi)  # "Symmetric" modulo
        self.x[1] = theta

        r = abs(np.pi - (abs(theta))) / (10 * np.pi)  # Reward for swingup; Max: 0.1

        term = False
        trunc = False
        info = {}
        if self.x[0] > self.x_threshold or self.x[0] < -self.x_threshold:
            term = True  # Just terminating should be enough punishment when reward is positive
            info = {"task": {"violation": True, "success": False}}
        if self.t > self.max_time:
            # check if the pole is upright in the last 10 steps
            if len(self.x_trajectory) >= 10:
                success = all(
                    np.abs(self.x_trajectory[i][1]) < 0.1 for i in range(-10, 0)
                )
            else:
                success = False  # Not enough data to determine success

            info = {"task": {"violation": False, "success": success}}
            trunc = True
        self.reset_needed = trunc or term

        return self.x, r, term, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        self.t = 0
        self.x = self.init_state(options)
        self.reset_needed = False

        self.x_trajectory = []
        self.pos_trajectory = None
        self.pole_end_trajectory = None
        return self.x, {}

    def init_state(self, options: Optional[dict] = None) -> np.ndarray:
        return np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float32)

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
        for x in state_trajectory:
            self.pos_trajectory.append(x[0])
            self.pole_end_trajectory.append(self.calc_pole_end(x[0], x[1], self.length))

    def calc_pole_end(
        self, x_coord: float, theta: float, length: float
    ) -> tuple[float, float]:
        # NOTE: The minus is necessary because theta is seen as counterclockwise
        pole_x = x_coord - length * np.sin(theta)
        pole_y = length * np.cos(theta)
        return pole_x, pole_y

    def render(self):
        import pygame
        from pygame import gfxdraw

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.x is None:
            return None

        world_width = 2 * self.x_threshold
        center = (int(self.screen_width / 2), int(self.screen_height / 2))
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * self.length
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
            import pygame

            pygame.display.quit()
            pygame.quit()


class CartPoleBalanceEnv(CartPoleEnv):
    def init_state(self, options: Optional[dict] = None) -> np.ndarray:
        low, high = gym_utils.maybe_parse_reset_bounds(
            options,
            -0.05,
            0.05,  # default low
        )  # default high
        return self.np_random.uniform(low=low, high=high, size=(4,))
