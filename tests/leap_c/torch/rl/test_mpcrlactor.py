import torch

from leap_c.controller import ParameterizedController


class DummyController(ParameterizedController):
    def __init__(self, param_dim: int) -> None:
        self._param_dim = param_dim

    def forward(self, obs: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        return param

    @property
    def parameter_dim(self) -> int:
        return self._param_dim

    def default_param(self, obs: None = None) -> torch.Tensor:
        return torch.arange(self._param_dim)


# def test_default_param_initialization():
#     cfg = SacZopTrainerConfig()
#     cfg.init_param_with_default = True
#     cfg.actor_mlp.hidden_dims = None  # No hidden layers, just a parameter tensor
#     cfg.distribution_name = "squashed_gaussian"
#     controller = DummyController(param_dim=4)

#     # actor = MpcSacActor()
