import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F


class TanhGaussian(nn.Module):
    """A Gaussian transformed by a Gaussian.

    The output is sampled from this distribution and then squashed with a tanh function.
    # TODO (Jasper): Why are we not using the transformed distr class from torch.
    """

    def __init__(
        self,
        minimal_std: float = 1e-3,
        log_tensors: bool = False,
    ):
        """Initializes the TanhNormal module.

        Args:
            minimal_std: The minimal standard deviation of the distribution.
                Will always be added to the softplus of the std to form the actual std.
            log_tensors: Whether to log EVERY INDEX ONE FOR ONE (but still meaned over batch dimensions) of the mean and std tensors in the stats dict.
        """
        super().__init__()
        self.minimal_std = minimal_std
        self.log_tensors = log_tensors

    def forward(
        self, mean: torch.Tensor, std: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
            deterministic: If True, the output will just be tanh(mean), no sampling is taking place.

        Returns:
            an output sampled from the TanhNormal, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        stats = dict()
        std = F.softplus(std) + self.minimal_std

        if deterministic:
            action = mean
            log_prob = 1
        else:
            dist = Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action)

        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)

        return real_action, real_log_prob, stats


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


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: str,
    use_activation_on_output: bool,
) -> nn.Sequential:
    """Returns a simple Multi-Layer Perceptron with orthogonal weight initialization.

    Parameters:
        input_dim: The dimension of the input.
        output_dim: The dimension of the output.
        hidden_dims: The dimensions of the hidden layers.
        activation: The type of activation function used in the hidden layers.
        use_activation_on_output: Whether to use an activation function on the output layer.
    """

    dims = [input_dim] + list(hidden_dims) + [output_dim]

    fcs = []
    for prev_d, d in zip(dims[:-1], dims[1:]):
        fcs.extend([nn.Linear(prev_d, d), string_to_activation(activation)])
    if use_activation_on_output:
        mlp = nn.Sequential(*fcs)
    else:
        mlp = nn.Sequential(*fcs[:-1])

    # TODO(Leonard): Do we need orthogonal initialization?
    # mlp.apply(weight_init)
    return mlp
