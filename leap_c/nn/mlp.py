from collections.abc import Sequence
from dataclasses import dataclass
import torch
import torch.nn as nn


def string_to_activation(activation: str) -> nn.Module:
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


@dataclass(kw_only=True)
class MLPConfig:
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"


class MLP(nn.Module):
    """A base class for a multi-layer perceptron (MLP) with a configurable number of
    layers and activation functions.

    Attributes:
        activation: The activation function to use in the hidden layers.
        mlp: The multi-layer perceptron model.
    """
    def __init__(
        self,
        input_dims: int | list[int],
        output_dims: int | list[int],
        mlp_cfg: MLPConfig,
    ) -> None:
        """Initializes the MLP.

        Args:
            input_dims: The dimensionality of the input tensor.
            output_dims: The dimensionality of the output tensor.
            mlpcfg: The configuration for the MLP.
        """
        super().__init__()

        self.activation = string_to_activation(mlp_cfg.activation)

        if isinstance(input_dims, int):
            input_dims = [input_dims]
        self._comb_input_dim = sum(input_dims)
        self._input_dims = input_dims

        if isinstance(output_dims, int):
            output_dims = [output_dims]
        self._comb_output_dim = sum(output_dims)
        self._output_dims = output_dims

        layers = []
        prev_d = self._comb_input_dim
        for d in mlp_cfg.hidden_dims:
            layers.extend([nn.Linear(prev_d, d), self.activation])
            prev_d = d

        self.mlp = nn.Sequential(*layers[:-1])

    def forward(self, *x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:

        if isinstance(x, tuple):
            x = torch.cat(x, dim=-1)  # type: ignore
        y = self.mlp(x)
        y = torch.split(y, self._output_dims, dim=-1)

        return y
