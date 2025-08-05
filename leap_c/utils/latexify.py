import shutil
from contextlib import contextmanager
from typing import Any, Iterator

from matplotlib import pyplot as plt


@contextmanager
def latex_plot_context(**kw: Any) -> Iterator[None]:
    """
    A context manager that temporarily applies LaTeX-style settings to Matplotlib plots.

    Parameters:
        **kw: Any additional Matplotlib rcParams to override the LaTeX defaults.

    Automatically disables 'text.usetex' if LaTeX is not available in PATH.

    Usage:
        with latex_plot_context():
            # LaTeX-styled plot

        with latex_plot_context(font_size=14, axes_titlesize=16):
            # override some defaults
    """
    text_usetex = shutil.which("latex") is not None

    base_config = {
        "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "text.usetex": text_usetex,
        "font.family": "serif",
    }

    base_config.update(kw)

    with plt.rc_context(base_config):
        yield


def latex_plot_decorator(**kw: Any):
    """
    A decorator that applies LaTeX-style settings to Matplotlib plots within the decorated function.

    Usage:
        @latex_plot_decorator()
        def plot():
            # LaTeX-styled plot

        @latex_plot_decorator(font_size=14)
        def plot_custom():
            # override some defaults
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with latex_plot_context(**kw):
                return func(*args, **kwargs)

        return wrapper

    return decorator
