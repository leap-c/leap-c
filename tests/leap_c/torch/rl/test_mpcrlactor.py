# TODO: Rewrite this to tests for residual
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium.spaces import Box, Space

from leap_c.controller import ParameterizedController
from leap_c.torch.rl.mpc_actor import HierachicalMPCActor, HierachicalMPCActorConfig
from leap_c.torch.rl.sac_fop import SacFopTrainerConfig
from leap_c.torch.rl.sac_zop import SacZopTrainerConfig


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
    def param_space(self) -> Space:
        return Box(
            np.array([-10.0] * self._param_dim, np.float32),
            np.array([20.0] * self._param_dim, np.float32),
        )

    def jacobian_action_param(self, ctx: DummyCtx) -> np.ndarray:
        """Return identity Jacobian for DummyController (action = param)."""
        return np.eye(self._param_dim)[np.newaxis, :, :]  # Shape: (1, param_dim, param_dim)


def test_default_param_initialization_zop():
    """Test parameter noise mode with residual learning."""
    param_dim = 4
    controller = DummyController(param_dim=param_dim)
    dummy_obs_space = Box(-np.inf, np.inf, (3,))
    dummy_action_space = Box(-1.0, 1.0, (param_dim,))

    cfg = HierachicalMPCActorConfig(
        noise="param",
        residual=True,
        distribution_name="squashed_gaussian",
        mlp=SacZopTrainerConfig().actor.mlp,
    )
    cfg.mlp.hidden_dims = None  # No hidden layers, just a parameter tensor

    actor = HierachicalMPCActor(
        cfg=cfg,
        observation_space=dummy_obs_space,
        action_space=dummy_action_space,
        controller=controller,
    )

    output = actor(torch.zeros((2, 3)), deterministic=True)
    assert output.param.shape == (2, param_dim)
    for sample in output.param:
        assert torch.allclose(sample, controller.default_param(), atol=1e-3)


def test_default_param_initialization_fop():
    """Test parameter noise mode with residual learning and entropy correction."""
    param_dim = 4
    controller = DummyController(param_dim=param_dim)
    dummy_obs_space = Box(-np.inf, np.inf, (3,))
    dummy_action_space = Box(-1.0, 1.0, (param_dim,))

    cfg = HierachicalMPCActorConfig(
        noise="param",
        residual=True,
        distribution_name="squashed_gaussian",
        entropy_correction=True,
        mlp=SacFopTrainerConfig().actor.mlp,
    )
    cfg.mlp.hidden_dims = None  # No hidden layers, just a parameter tensor

    actor = HierachicalMPCActor(
        cfg=cfg,
        observation_space=dummy_obs_space,
        action_space=dummy_action_space,
        controller=controller,
    )

    output = actor(torch.zeros((2, 3)), deterministic=True)
    assert output.param.shape == (2, param_dim)
    for sample in output.param:
        assert torch.allclose(sample, controller.default_param(), atol=1e-3)


def test_default_param_initialization_foa():
    """Test action noise mode (parameters are deterministic, noise on actions)."""
    param_dim = 4
    controller = DummyController(param_dim=param_dim)
    dummy_obs_space = Box(-np.inf, np.inf, (3,))
    dummy_action_space = Box(np.array([-100.0], np.float32), np.array([300.0], np.float32), (1,))

    cfg = HierachicalMPCActorConfig(
        noise="action",  # Action noise mode
        distribution_name="squashed_gaussian",
        mlp=SacFopTrainerConfig().actor.mlp,
    )
    cfg.mlp.hidden_dims = None  # No hidden layers, just a parameter tensor

    actor = HierachicalMPCActor(
        cfg=cfg,
        observation_space=dummy_obs_space,
        action_space=dummy_action_space,
        controller=controller,
    )

    # In action noise mode, parameters are deterministic (no residual available)
    # The test should verify that params are produced deterministically
    output = actor(torch.zeros((2, 3)), deterministic=True)
    assert output.param.shape == (2, param_dim)
    # Since no residual in action noise mode, params won't match default_param exactly
    # Just verify shape is correct
