import abc
from typing import Any, Generic, TypeVar

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from leap_c.controller import CtxType
from leap_c.utils.latexify import latex_plot_context

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class MatplotlibRenderEnv(abc.ABC, gym.Env[ObsType, ActType], Generic[ObsType, ActType, CtxType]):
    """A class for Gymnasium environments to handle Matplotlib rendering.

    This mixin provides the boilerplate for `render()` and `close()` methods. To use it, an
    environment class must inherit from this mixin and implement two abstract methods:
    - `_render_setup()`: For one-time plot initialization.
    - `_render_frame()`: For updating the plot on each render call.

    The environment must also have a `render_mode` attribute.

    Attributes:
        render_mode: The mode for rendering the environment.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str | None = None, **kwargs: Any) -> None:
        """Initializes the environment with a specified render mode.

        Args:
            render_mode: The mode in which to render the environment.
                Can be `"human"` for interactive display or `"rgb_array"` for image output.
            **kwargs: Additional keyword arguments for the environment.
        """
        super().__init__(**kwargs)

        self.render_mode = render_mode

        self._fig: Figure | None = None
        self._ax: Axes | np.ndarray | None = None
        self._render_initialized: bool = False

    def render(self) -> np.ndarray | list[np.ndarray] | None:
        """Renders the environment.

        Handles initialization on the first call and delegates to the environment-specific drawing
        methods.
        """
        with latex_plot_context():
            if self.render_mode is None:
                gym.logger.warn(
                    "Cannot render because render_mode is not set. Set `render_mode` at "
                    "initialization, e.g., `gym.make(env_id, render_mode='human')`."
                )
                return None

            if not self._render_initialized:
                self._render_setup()
                self._render_initialized = True

            self._render_frame()

            if self.render_mode == "human":
                self._fig.canvas.draw()  # type: ignore
                self._fig.canvas.flush_events()  # type: ignore
            elif self.render_mode == "rgb_array":
                canvas = FigureCanvas(self._fig)
                canvas.draw()  # type: ignore

                image = np.frombuffer(
                    self._fig.canvas.buffer_rgba(),  # type:ignore
                    dtype=np.uint8,
                )
                width, height = self._fig.canvas.get_width_height()  # type:ignore
                return image.reshape(height, width, 4)[:, :, :3]
            else:
                raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def set_ctx(self, ctx: CtxType) -> None:
        """Sets the context for rendering.

        This method can be used to pass additional information to the rendering methods.

        Args:
            ctx: The context to set.
        """
        self.ctx = ctx

    def close(self) -> None:
        """Closes the rendering window."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        self._render_initialized = False

    @abc.abstractmethod
    def _render_setup(self) -> None:
        """One-time setup for the rendering.

        This method should create the figure and axes (e.g., `self.fig, self.ax = plt.subplots()`)
        and draw any static elements of the plot.
        """

    @abc.abstractmethod
    def _render_frame(self) -> None:
        """Update the plot with the current environment state.

        This method is called on every `render()` call and should update
        dynamic elements like agent position, trajectories, etc.
        """
