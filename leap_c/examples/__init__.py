from functools import partial
from pathlib import Path
from typing import Any, Literal

from gymnasium import Env, make

from ..controller import ParameterizedController
from .cartpole.controller import CartPoleController, CartPoleControllerConfig
from .cartpole.env import CartPoleEnv
from .chain.controller import ChainController, ChainControllerConfig
from .chain.env import ChainEnv
from .hvac.controller import HvacController
from .hvac.env import StochasticThreeStateRcEnv
from .pointmass.controller import PointMassController, PointMassControllerConfig
from .pointmass.env import PointMassEnv

ExampleEnvName = Literal["cartpole", "chain", "pointmass", "hvac"]
ENV_REGISTRY = {
    "cartpole": CartPoleEnv,
    "chain": ChainEnv,
    "pointmass": PointMassEnv,
    "hvac": StochasticThreeStateRcEnv,
}

ExampleControllerName = Literal[
    "cartpole",
    "cartpole_stagewise",
    "chain",
    "chain_stagewise",
    "pointmass",
    "pointmass_stagewise",
    "hvac",
    "hvac_stagewise",
]
CONTROLLER_REGISTRY = {
    "cartpole": (CartPoleController, CartPoleControllerConfig, dict()),
    "cartpole_stagewise": (
        CartPoleController,
        CartPoleControllerConfig,
        {"param_interface": "stagewise"},
    ),
    "chain": (ChainController, ChainControllerConfig, dict()),
    "chain_stagewise": (
        ChainController,
        ChainControllerConfig,
        {"param_interface": "stagewise"},
    ),
    "pointmass": (PointMassController, PointMassControllerConfig, dict()),
    "pointmass_stagewise": (
        PointMassController,
        PointMassControllerConfig,
        {"param_interface": "stagewise"},
    ),
    "hvac": HvacController,
    "hvac_stagewise": partial(HvacController, stagewise=True),
}


def create_env(env_name: ExampleEnvName | str, **kw: Any) -> Env:
    """Create an environment based on the given name.

    The methods first attempts to find the environment in the `leap-c` registry. If not found, it
    tries to create it using `gymnasium.make`. If both attempts fail, an exception is raised.

    Args:
        env_name: Name of the environment.
        **kw: Additional keyword arguments passed to the environment constructor.

    Returns:
        An instance of the requested environment.

    Raises:
        Error: If the environment is neither registered in `leap-c` or `gymnasium`.
    """
    if env_name in ENV_REGISTRY:
        return ENV_REGISTRY[env_name](**kw)
    kw["disable_env_checker"] = True
    return make(env_name, **kw)


def create_controller(
    controller_name: ExampleControllerName,
    reuse_code_base_dir: Path | None = None,
    **kw: Any,
) -> ParameterizedController:
    """Create a controller.

    Args:
        controller_name: Name of the controller.
        reuse_code_base_dir: Directory to reuse code base from, e.g., generated code.
        **kw: Additional keyword arguments passed to the controller config constructor.

    Returns:
        An instance of the requested controller.
    """
    if controller_name not in CONTROLLER_REGISTRY:
        raise ValueError(f"Controller '{controller_name}' is not registered or does not exist.")

    if controller_name == "hvac" or controller_name == "hvac_stagewise":
        controller_class = CONTROLLER_REGISTRY[controller_name]
    else:
        controller_class, config_class, default_cfg_kwargs = CONTROLLER_REGISTRY[controller_name]
        cfg = config_class(**{**default_cfg_kwargs, **kw})
        kw = {"cfg": cfg}

    if reuse_code_base_dir is not None:
        export_directory = reuse_code_base_dir / controller_name
        kw.pop("export_directory", None)  # remove if present
        try:
            return controller_class(**kw, export_directory=export_directory)
        except TypeError:
            pass

    return controller_class(**kw)
