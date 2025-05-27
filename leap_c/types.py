from typing import TypeVar


# This TypeVar is used to define a generic type for calculation contexts
# used by differentiable functions, controllers, planners etc. that are
# wrapped in autograd functions.

CalcCtx = TypeVar("CalcCtx")
