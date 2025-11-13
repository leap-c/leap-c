from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from typing import Callable, Iterable, Literal, get_args

import torch
import torch.nn as nn

from leap_c.controller import ParameterizedController
from leap_c.torch.nn.bounded_distributions import (
    BoundedDistribution,
    BoundedTransform,
    SquashedGaussian,
)

Activation = Literal["relu", "tanh", "sigmoid", "leaky_relu"]
WeightInit = Literal["orthogonal"]


def string_to_activation(activation: Activation) -> nn.Module:
    """Convert a string to an activation.

    Args:
        activation ({"relu", "tanh", "sigmoid", "leaky_relu"}): The activation function to convert.

    Raises:
        ValueError: If the activation function is not recognized.

    Returns:
        nn.Module: The activation function as a torch module.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(
        f"Activation function `{activation}` not recognized; available options are: "
        f"{', '.join(get_args(Activation))}."
    )


def orthogonal_init(module: nn.Module) -> None:
    """Initialize the weights of a module using orthogonal initialization.

    Args:
        module (nn.Module): The module to initialize.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        module.bias.data.fill_(0.0)


def string_to_weight_init(weight_init: WeightInit) -> Callable[[nn.Module], None]:
    """Convert a string to an initializiation method.

    Args:
        weight_init ({"orthogonal"}): The weight initialization method. For now, only "orthogonal"
            is supported.

    Raises:
        ValueError: If the weight initialization method is not recognized.

    Returns:
        Callable[[nn.Module], None]: The weight initialization function that takes in a module and
            initializes its weights, returning nothing.
    """
    if weight_init == "orthogonal":
        return orthogonal_init
    raise ValueError(
        f"Weight initialization `{weight_init}` not recognized; available options are: "
        f"{', '.join(get_args(WeightInit))}."
    )


@dataclass(kw_only=True)
class MlpConfig:
    """Configuration for a multi-layer perceptron (MLP).

    Attributes:
        hidden_dims: A sequence of integers representing the sizes of the hidden layers. If `None`,
            no hidden layers will be used, and the MLP will be replaced with a parameter tensor with
            the given output size.
        activation: The activation function to use in the hidden layers.
        weight_init: The weight initialization method to use for the hidden layers. If `None`, no
            initialization will be applied.
    """

    hidden_dims: Sequence[int] | None = (256, 256, 256)
    activation: Activation = "relu"
    weight_init: WeightInit | None = "orthogonal"  # If None, no init will be used


class Mlp(nn.Module):
    """A base class for a multi-layer perceptron (MLP).

    The MLP includes a configurable number of layers and activation functions.

    Attributes:
        mlp: The MLP model. Is `None` if no hidden layers were set in the config (see
            `MlpConfig.hidden_dims`), in which case the parameter tensor `param` is set instead.
        param: A parameter tensor with the given output size. Is `None` if hidden layers were set in
            the config, in which case the MLP model `mlp` is set instead.
    """

    mlp: nn.Module | None
    param: nn.Parameter | None

    def __init__(
        self,
        input_sizes: int | Iterable[int],
        output_sizes: int | Iterable[int],
        mlp_cfg: MlpConfig,
    ) -> None:
        """Initializes the MLP.

        Args:
            input_sizes: The sizes of the input tensors. Inputs will be concatenated.
            output_sizes: The sizes of the output tensors.
                Outputs will be split according to these sizes.
            mlp_cfg: The configuration for the MLP.
        """
        super().__init__()

        comb_input_dim = input_sizes if isinstance(input_sizes, Integral) else sum(input_sizes)
        if isinstance(output_sizes, Integral):
            self._output_dims = [output_sizes]
            comb_output_dim = output_sizes
        else:
            self._output_dims = list(output_sizes)
            comb_output_dim = sum(self._output_dims)

        if mlp_cfg.hidden_dims is None or len(mlp_cfg.hidden_dims) == 0:
            self.mlp = None
            self.param = nn.Parameter(torch.zeros(comb_output_dim))
            return

        layers: list[nn.Module] = []
        activation = string_to_activation(mlp_cfg.activation)
        for in_sz, out_sz in pairwise((comb_input_dim, *mlp_cfg.hidden_dims, comb_output_dim)):
            layers.extend((nn.Linear(in_sz, out_sz), activation))

        self.mlp = nn.Sequential(*layers[:-1])
        self.param = None

        if mlp_cfg.weight_init is not None:
            self.mlp.apply(string_to_weight_init(mlp_cfg.weight_init))

    def forward(self, *x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.param is not None:
            batch_size = x[0].shape[0]
            y = self.param.unsqueeze(0).expand(batch_size, -1)
        else:
            if isinstance(x, tuple):
                x = torch.cat(x, dim=-1)  # type: ignore
            y = self.mlp(x)  # type: ignore

        if len(self._output_dims) == 1:
            return y

        return torch.split(y, self._output_dims, dim=-1)


def init_mlp_params_with_inverse_default(
    mlp: Mlp,
    bounded_fun: BoundedDistribution | BoundedTransform,
    controller: ParameterizedController,
) -> None:
    """Initialize the parameters of the MLP to produce the default parameters.

    The MLP is initialized such that the mean of the gaussian transformed by the squashing
    of the SquashedGaussian corresponds to the default parameters defined by the controller.

    This function assumes
        1. that the MLP is used to predict Mean and Std of a
        SquashedGaussian over the parameters of a ParameterizedController.
        2. that the MLP uses only a nn.Parameter for the mean.
        3. that the default parameters can be obtained by controller.default_param(None)

    Args:
        mlp: The MLP used to predict the parameters.
        bounded_fun: The bounded distribution or the bounded transform
            that are applied such that the mlp output corresponds to values within
            the parameter space.
        controller: The parameterized controller needed to obtain the default parameters.
    """
    if mlp.param is not None:
        if not isinstance(bounded_fun, SquashedGaussian) and not isinstance(
            bounded_fun, BoundedTransform
        ):
            raise ValueError(
                "Initializing the parameters with the inverse default "
                "only works for SquashedGaussian or BoundedTransform, "
                f"but got {type(bounded_fun)}."
            )
        try:
            # Hope you fail when the default param is dependent on the observation
            # and else everything is fine
            params = controller.default_param(obs=None)
            params = torch.tensor(params, dtype=mlp.param.dtype, device=mlp.param.device)
        except Exception as e:
            raise ValueError(
                "Initializing the parameters with the inverse default only makes sense if the "
                "default parameters of the controller do not depend on the "
                "observation. Could it be that this is not the case here?"
            ) from e
        param_dim = params.shape[0]
        with torch.no_grad():
            params_untransformed = bounded_fun.inverse(x=params[None, :], padding=0)
            mlp.param[:param_dim] = params_untransformed.flatten()
    # NOTE Do nothing if the mlp uses hidden layers
