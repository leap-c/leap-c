from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal

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
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Activation function {activation} not recognized.")


def orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        module.bias.data.fill_(0.0)


def string_to_weight_init(weight_init: WeightInit) -> Callable[[nn.Module], None]:
    if weight_init == "orthogonal":
        return orthogonal_init
    else:
        raise ValueError(f"Weight initialization {weight_init} not recognized.")


@dataclass(kw_only=True)
class MlpConfig:
    """Configuration for a multi-layer perceptron (MLP).

    Attributes:
        hidden_dims: A sequence of integers representing the sizes of the hidden
            layers. If None, no hidden layers will be used, and the MLP will be
            replaced with a parameter tensor of the output size.
        activation: The activation function to use in the hidden layers.
        weight_init: The weight initialization method to use for the hidden layers.
            If None, no initialization will be applied.
    """

    hidden_dims: Sequence[int] | None = (256, 256, 256)
    activation: Activation = "relu"
    weight_init: WeightInit | None = "orthogonal"  # If None, no init will be used


class Mlp(nn.Module):
    """A base class for a multi-layer perceptron (MLP) with a configurable number of
    layers and activation functions.

    Attributes:
        activation: The activation function to use in the hidden layers.
        mlp: The multi-layer perceptron model. Is None if no hidden layers were set in the config,
            and a parameter tensor is used instead.
        param: A parameter tensor of the output size. Is None if hidden layers were set
            in the config.
    """

    activation: nn.Module
    mlp: nn.Module | None
    param: nn.Parameter | None

    def __init__(
        self,
        input_sizes: int | list[int],
        output_sizes: int | list[int],
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

        self.activation = string_to_activation(mlp_cfg.activation)

        if isinstance(input_sizes, int):
            input_sizes = [input_sizes]
        self._comb_input_dim = sum(input_sizes)
        self._input_dims = input_sizes

        if isinstance(output_sizes, int):
            output_sizes = [output_sizes]
        self._comb_output_dim = sum(output_sizes)
        self._output_dims = output_sizes

        if mlp_cfg.hidden_dims is None or len(mlp_cfg.hidden_dims) == 0:
            self.mlp = None
            self.param = nn.Parameter(torch.zeros(self._comb_output_dim))
            return

        # mlp
        layers = []
        prev_d = self._comb_input_dim
        for d in [*mlp_cfg.hidden_dims, self._comb_output_dim]:
            layers.extend([nn.Linear(prev_d, d), self.activation])
            prev_d = d

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
    """
    Sets the parameters of the MLP such that the mean of the gaussian transformed by the squashing
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
