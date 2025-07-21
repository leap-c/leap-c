from functools import partial
from .cartpole.env import CartPoleEnv
from .cartpole.controller import CartPoleController
from .chain.env import ChainEnv
from .chain.controller import ChainController
from .pointmass.env import PointMassEnv
from .pointmass.controller import PointMassController


ENV_REGISTRY = {
    "cartpole": CartPoleEnv,
    "chain": ChainEnv,
    "pointmass": PointMassEnv,
}


CONTROLLER_REGISTRY = {
    "cartpole": CartPoleController,
    "cartpole_stagewise": partial(CartPoleController, stagewise=True),
    "chain": ChainController,
    "chain_stagewise": partial(ChainController, stagewise=True),
    "pointmass": PointMassController,
    "pointmass_stagewise": partial(PointMassController, stagewise=True),
}


def create_env(env_name: str, **kw):
    """Create an environment based on the given name."""
    if env_name in ENV_REGISTRY:
        return ENV_REGISTRY[env_name](**kw)

    raise ValueError(f"Environment '{env_name}' is not registered.")


def create_controller(env_name: str):
    """Create a controller based on the given environment name."""
    if env_name in CONTROLLER_REGISTRY:
        controller_class = CONTROLLER_REGISTRY[env_name]
        if controller_class is not None:
            return controller_class()

    raise ValueError(f"Controller for environment '{env_name}' is not registered or does not exist.")
