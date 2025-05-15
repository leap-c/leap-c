import torch
from torch.distributions import TransformedDistribution, Normal, TanhTransform, AffineTransform


class ClampedTanhTransform(TanhTransform):
    def __init__(self, epsilon: float = 1e-6, cache_size: int = 0):
        super().__init__(cache_size=cache_size)
        self.epsilon = epsilon

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        y = super()._call(x)
        return y.clamp(-1 + self.epsilon, 1 - self.epsilon)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = y.clamp(-1 + self.epsilon, 1 - self.epsilon)
        return super()._inverse(y)



class SquashedGaussianButBetter(TransformedDistribution):
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        epsilon: float = 1e-6
    ):
        base = Normal(mean, std)
        transforms = [
            ClampedTanhTransform(epsilon=epsilon),
            AffineTransform(loc=loc, scale=scale)
        ]
        super().__init__(base, transforms)
        self.scale = scale

    def entropy(self) -> torch.Tensor:
        base_entropy = self.base_dist.entropy()
        affine_log_scale = torch.log(self.scale.abs())
        return base_entropy + affine_log_scale

    @property
    def mode(self) -> torch.Tensor:
        y = self.base_dist.mode
        for transform in self.transforms:
            y = transform(y)
        return y
