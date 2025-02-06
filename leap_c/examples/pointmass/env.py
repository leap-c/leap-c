from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PointMassEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        dt: float = 0.01,
        max_time: float = 10.0,
        render_mode: str | None = None,
    ):
        super().__init__()

        # Will be added after doing a step.
        self.current_noise = None

        # For rendering
        if render_mode is not None:
            raise NotImplementedError("Rendering is not implemented yet.")

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        self.action_to_take = action
        frame = None
        # if self.render_mode == "human" or self.render_mode == "rgb_array":
        #     frame = self.render()
        o, r, term, trunc, info = super().step(
            action
        )  # o is the next state as np.ndarray, next parameters as MPCParameter
        info["frame"] = frame

        state = o[0].copy()
        state[-2:] += self.current_noise
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
        return self._np_random.uniform(-0.1, 0, size=2)

    def init_state(self):
        return self.mpc.ocp.constraints.x0.astype(dtype=np.float32)
