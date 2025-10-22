from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from leap_c.controller import ParameterizedController
from leap_c.torch.nn.extractor import IdentityExtractor
from leap_c.torch.rl.sac_fop import FoaActor, FopActor, SacFopTrainerConfig
from leap_c.torch.rl.sac_zop import MpcSacActor, SacZopTrainerConfig


@dataclass
class DummyCtx:
    log = {}
    status = torch.zeros(1)


class DummyController(ParameterizedController):
    def __init__(self, param_dim: int) -> None:
        super().__init__()
        self._param_dim = param_dim

    def forward(
        self, obs: torch.Tensor, param: torch.Tensor, ctx=None
    ) -> tuple[DummyCtx, torch.Tensor]:
        return DummyCtx(), param

    @property
    def parameter_dim(self) -> int:
        return self._param_dim

    def default_param(self, obs: None = None) -> torch.Tensor:
        return torch.arange(self._param_dim, dtype=torch.float32)

    @property
    def param_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=np.array([-10.0] * self._param_dim), high=np.array([20.0] * self._param_dim)
        )


def test_default_param_initialization_zop():
    cfg = SacZopTrainerConfig()
    cfg.init_param_with_default = True
    cfg.actor_mlp.hidden_dims = None  # No hidden layers, just a parameter tensor
    cfg.distribution_name = "squashed_gaussian"
    param_dim = 4
    controller = DummyController(param_dim=param_dim)
    extractor = IdentityExtractor
    dummy_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))

    actor = MpcSacActor(
        extractor_cls=extractor,
        observation_space=dummy_obs_space,
        controller=controller,
        distribution_name=cfg.distribution_name,
        mlp_cfg=cfg.actor_mlp,
        init_param_with_default=cfg.init_param_with_default,
    )

    output = actor(torch.zeros((2, 3)), deterministic=True)
    assert output.param.shape == (2, param_dim)
    for sample in output.param:
        assert torch.allclose(sample, controller.default_param())


def test_default_param_initialization_fop():
    cfg = SacFopTrainerConfig()
    cfg.init_param_with_default = True
    cfg.actor_mlp.hidden_dims = None  # No hidden layers, just a parameter tensor
    cfg.distribution_name = "squashed_gaussian"
    param_dim = 4
    controller = DummyController(param_dim=param_dim)
    dummy_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
    extractor = IdentityExtractor(dummy_obs_space)

    actor = FopActor(
        extractor=extractor,
        mlp_cfg=cfg.actor_mlp,
        controller=controller,
        distribution_name=cfg.distribution_name,
        correction=cfg.entropy_correction,
        init_param_with_default=cfg.init_param_with_default,
    )

    output = actor(torch.zeros((2, 3)), deterministic=True)
    assert output.param.shape == (2, param_dim)
    for sample in output.param:
        assert torch.allclose(sample, controller.default_param())


def test_default_param_initialization_foa():
    cfg = SacFopTrainerConfig()
    cfg.init_param_with_default = True
    cfg.actor_mlp.hidden_dims = None  # No hidden layers, just a parameter tensor
    cfg.distribution_name = "squashed_gaussian"
    param_dim = 4
    controller = DummyController(param_dim=param_dim)
    dummy_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
    extractor = IdentityExtractor(dummy_obs_space)
    dummy_action_space = gym.spaces.Box(low=np.array([-100.0]), high=np.array([300.0]), shape=(1,))

    actor = FoaActor(
        action_space=dummy_action_space,
        extractor=extractor,
        mlp_cfg=cfg.actor_mlp,
        controller=controller,
        init_param_with_default=cfg.init_param_with_default,
    )

    output = actor(torch.zeros((2, 3)), deterministic=True)
    assert output.param.shape == (2, param_dim)
    for sample in output.param:
        assert torch.allclose(sample, controller.default_param())
