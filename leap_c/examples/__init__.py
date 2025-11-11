"""In this module, we manage all the example environments, controllers and planners."""

from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeAlias

from gymnasium import Env

from leap_c.controller import CtxType, ParameterizedController
from leap_c.examples.cartpole.env import CartPoleEnv
from leap_c.examples.cartpole.planner import CartPolePlanner, CartPolePlannerConfig
from leap_c.examples.chain.env import ChainEnv
from leap_c.examples.chain.planner import ChainControllerConfig, ChainPlanner
from leap_c.examples.hvac.controller import HvacController
from leap_c.examples.hvac.env import StochasticThreeStateRcEnv
from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.examples.pointmass.planner import PointMassControllerConfig, PointMassPlanner
from leap_c.planner import ControllerFromPlanner, ParameterizedPlanner

ExampleEnvName = Literal["cartpole", "chain", "pointmass", "hvac"]
ENV_REGISTRY = {
    "cartpole": CartPoleEnv,
    "chain": ChainEnv,
    "pointmass": PointMassEnv,
    "hvac": StochasticThreeStateRcEnv,
}


def create_env(env_name: ExampleEnvName, **kw: Any) -> Env:
    """Create an environment based on the given name.

    Args:
        env_name: Name of the environment.
        **kw: Additional keyword arguments passed to the environment constructor.

    Returns:
        An instance of the requested environment.
    """
    if env_name in ENV_REGISTRY:
        return ENV_REGISTRY[env_name](**kw)

    raise ValueError(f"Environment '{env_name}' is not registered.")


PLANNER_REGISTRY = {
    "cartpole": (CartPolePlanner, CartPolePlannerConfig, dict()),
    "cartpole_stagewise": (
        CartPolePlanner,
        CartPolePlannerConfig,
        {"param_interface": "stagewise"},
    ),
    "chain": (ChainPlanner, ChainControllerConfig, dict()),
    "chain_stagewise": (
        ChainPlanner,
        ChainControllerConfig,
        {"param_interface": "stagewise"},
    ),
    "pointmass": (PointMassPlanner, PointMassControllerConfig, dict()),
    "pointmass_stagewise": (
        PointMassPlanner,
        PointMassControllerConfig,
        {"param_interface": "stagewise"},
    ),
    "hvac": HvacController,
    "hvac_stagewise": partial(HvacController, stagewise=True),
}
ExamplePlannerName = Literal[
    "cartpole",
    "cartpole_stagewise",
    "chain",
    "chain_stagewise",
    "pointmass",
    "pointmass_stagewise",
    "hvac",
    "hvac_stagewise",
]


def create_planner(
    planner_name: ExamplePlannerName, reuse_code_base_dir: Path | None = None, **kw: Any
) -> ParameterizedPlanner[CtxType]:
    """Create a planner.

    Args:
        planner_name: Name of the planner.
        reuse_code_base_dir: Directory to reuse code base from, e.g., generated code.
        **kw: Additional keyword arguments passed to the planner config constructor.

    Returns:
        An instance of the requested planner.
    """
    if planner_name not in PLANNER_REGISTRY:
        raise ValueError(f"Planner '{planner_name}' is not registered or does not exist.")

    # TODO: Remove this distinction when hvac provides the same interface as the other examples
    if planner_name == "hvac" or planner_name == "hvac_stagewise":
        planner_class = PLANNER_REGISTRY[planner_name]
    else:
        planner_class, config_class, default_cfg_kwargs = PLANNER_REGISTRY[planner_name]
        cfg = config_class(**{**default_cfg_kwargs, **kw})
        kw = {"cfg": cfg}

    if reuse_code_base_dir is not None:
        export_directory = reuse_code_base_dir / planner_name
        kw.pop("export_directory", None)  # remove if present
        try:
            return planner_class(**kw, export_directory=export_directory)
        except TypeError:
            pass

    return planner_class(**kw)


CONTROLLER_REGISTRY = {}
# controllers are a superset of planners
ExampleControllerName: TypeAlias = ExamplePlannerName


def create_controller(
    controller_name: ExampleControllerName, reuse_code_base_dir: Path | None = None, **kw: Any
) -> ParameterizedController[CtxType]:
    """Create a controller or create a planner and wrap it as a controller.

    Args:
        controller_name: Name of the controller.
        reuse_code_base_dir: Directory to reuse code base from, e.g., generated code.
        **kw: Additional keyword arguments passed to the planner config constructor.

    Returns:
        An instance of the requested controller.
    """
    if controller_name in PLANNER_REGISTRY:
        planner = create_planner(controller_name, reuse_code_base_dir=reuse_code_base_dir, **kw)
        return ControllerFromPlanner(planner)

    if controller_name not in CONTROLLER_REGISTRY:
        raise ValueError(f"Controller '{controller_name}' is not registered or does not exist.")

    controller_class, config_class, default_cfg_kwargs = CONTROLLER_REGISTRY[controller_name]
    cfg = config_class(**{**default_cfg_kwargs, **kw})
    kw = {"cfg": cfg}

    if reuse_code_base_dir is not None:
        export_directory = reuse_code_base_dir / controller_name
        kw.pop("export_directory", None)

        try:
            return controller_class(**kw, export_directory=export_directory)
        except TypeError:
            pass

    return controller_class(**kw)
