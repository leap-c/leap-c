from abc import ABC, abstractmethod

import numpy as np


class DiffFunction(ABC):
    """Abstract base class for differentiable functions.

    Subclasses must implement `forward` and `backward` methods.

    - The `forward` method computes outputs from inputs and returns a
      tuple containing a context object followed by one or more output
      arrays.

    - The `backward` method receives the context and the gradients of
      the outputs, and returns a tuple of gradients with respect to the
      inputs.

    Example:
        class MyFunction:
            def forward(
                self, *inputs: np.ndarray, ctx: Optional[dict] = None
            ):
                if ctx is None:
                    ctx = {}
                # Store intermediate values for backward
                ctx["saved"] = inputs[0]

                # Example computation
                out1 = inputs[0] + inputs[1]
                out2 = inputs[0] * 2

                return ctx, out1, out2

            def backward(
                self, ctx: dict, *grad_outputs: np.ndarray  # type: ignore
            ):
                # Retrieve saved values
                x = ctx["saved"]

                # Example gradient computation
                grad_out1, grad_out2 = grad_outputs
                grad_x = grad_out1 + 2 * grad_out2
                grad_y = grad_out1

                return grad_x, grad_y
    """

    @abstractmethod
    def forward(self, ctx=None, *inputs: np.ndarray | None):
        """Computes the output of the function given inputs.

        Args:
            inputs: Input arrays to the function.
            ctx: Optional context object that can be used to store intermediate
                values for the backward pass.

        Returns:
            A tuple where the first element is a context object, followed by
            one or more output arrays.
        """
        ...

    @abstractmethod
    def backward(self, ctx, *output_grads: np.ndarray | None):
        """Computes the gradient of the function with respect to its inputs.

        Args:
            ctx: The context object returned from the forward pass.
            output_grads: Gradients with respect to each output.

        Returns:
            A tuple of gradients with respect to each input.
        """
        ...
