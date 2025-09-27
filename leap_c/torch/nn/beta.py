import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions.beta import Beta


class ScaledBeta(nn.Module):
    """A concave (alpha, beta > 1 is enforced) scaled Beta distribution.
    Samples the output from a Beta distribution specified by the input,
    and then scales and shifts the result to match the space.

    Can for example be used to enforce certain action bounds of a stochastic policy.

    Attributes:
        scale: The scale of the space-fitting transform.
        loc: The location of the space-fitting transform (for shifting).
    """

    scale: torch.Tensor
    loc: torch.Tensor

    def __init__(
        self,
        space: spaces.Box,
    ):
        """Initializes the ScaledBeta module.

        Args:
            space: Space the output should fit to.
        """
        super().__init__()

        loc = space.low
        scale = space.high - space.low

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(
        self, log_alpha: torch.Tensor, log_beta: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Note that alpha and beta are enforced to be > 1 to ensure concavity.
        Args:
            log_alpha: The logarithm of the alpha parameter of the Beta distribution.
            log_beta: The logarithm of the beta parameter of the Beta distribution.
            deterministic: If True, the output will just be spacefitting(mode),
                no sampling is taking place.

        Returns:
            An output sampled from the ScaledBeta distribution, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        # Add 1 to ensure concavity
        alpha = torch.exp(log_alpha) + 1.0
        beta = torch.exp(log_beta) + 1.0

        dist = Beta(alpha, beta)

        if deterministic:
            y = dist.mode
        else:
            # reparameterization trick
            y = dist.rsample()
        log_prob = dist.log_prob(y)

        y_scaled = y * self.scale[None, :] + self.loc[None, :]
        log_prob -= torch.log(self.scale[None, :])
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # We could return the mean of alpha and beta as stats,
        # but I think they should at least be investigated for each action
        # dimension independently
        return y_scaled, log_prob, {}
