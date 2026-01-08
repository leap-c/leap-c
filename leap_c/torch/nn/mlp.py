from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from typing import Callable, Iterable, Literal, get_args

import torch
import torch.nn as nn

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


class BetaParameterNetwork(Mlp):
    """Neural network that outputs mode and concentration parameters for Beta distribution."""

    def __init__(
        self,
        input_sizes: int | Iterable[int],
        output_sizes: int | Iterable[int],
        mlp_cfg: MlpConfig,
    ) -> None:
        # Initialize parent with input_dim, two outputs of output_dim each (mode and concentration)
        super().__init__(
            input_sizes=input_sizes,
            output_sizes=output_sizes,
            mlp_cfg=mlp_cfg,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward call.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            mode: Mode parameter in [0, 1], shape (..., output_dim)
            concentration: Concentration parameter > 2, shape (..., output_dim)
        """
        # Get raw outputs from parent MLP
        mode_raw, concentration_raw = super().forward(x)

        # Mode must be in [0, 1]
        mode = torch.sigmoid(input=mode_raw)

        # Clamp mode to avoid numerical issues (NaN or boundary values)
        # Use small epsilon to keep values strictly in (eps, 1-eps)
        eps = 1e-6
        mode = torch.clamp(mode, min=eps, max=1.0 - eps)

        # Replace any NaN values with 0.5 (middle of valid range)
        mode = torch.where(
            torch.isnan(mode), torch.tensor(0.5, device=mode.device, dtype=mode.dtype), mode
        )

        # Concentration must be > 2 for valid parameterization
        concentration = torch.nn.functional.softplus(input=concentration_raw) + 2.0

        # Clamp concentration to reasonable range [2 + eps, 100]
        concentration = torch.clamp(concentration, min=2.0 + eps, max=100.0)

        # Replace any NaN values with 3.0 (safe default)
        concentration = torch.where(
            torch.isnan(concentration),
            torch.tensor(3.0, device=concentration.device, dtype=concentration.dtype),
            concentration,
        )

        return mode, concentration
