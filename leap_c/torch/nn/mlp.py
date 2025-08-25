from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn


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
    hidden_dims: Sequence[int] | None = (256, 256, 256)
    activation: Activation = "relu"
    weight_init: WeightInit | None = "orthogonal"  # If None, no init will be used


class MLP(nn.Module):
    """A base class for a multi-layer perceptron (MLP) with a configurable number of
    layers and activation functions.

    Attributes:
        activation: The activation function to use in the hidden layers.
        mlp: The multi-layer perceptron model.
    """

    def __init__(
        self,
        input_sizes: int | list[int],
        output_sizes: int | list[int],
        mlp_cfg: MlpConfig,
    ) -> None:
        """Initializes the MLP.

        Args:
            input_sizes: The sizes of the input tensors.
            output_sizes: The sizes of the output tensors
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

        if mlp_cfg.hidden_dims is not None:
            # mlp
            layers = []
            prev_d = self._comb_input_dim
            for d in [*mlp_cfg.hidden_dims, self._comb_output_dim]:
                layers.extend([nn.Linear(prev_d, d), self.activation])
                prev_d = d

            self.mlp = nn.Sequential(*layers[:-1])

            if mlp_cfg.weight_init is not None:
                self.mlp.apply(string_to_weight_init(mlp_cfg.weight_init))

            self.param = None
        else:
            self.mlp = nn.Parameter(torch.zeros(self._comb_output_dim))

    def forward(self, *x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(x, tuple):
            x = torch.cat(x, dim=-1)  # type: ignore
        y = self.mlp(x)

        if len(self._output_dims) == 1:
            return y
        y = torch.split(y, self._output_dims, dim=-1)
        return y
