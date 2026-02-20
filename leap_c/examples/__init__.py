"""In this module, we manage all the example environments, controllers and planners."""

from pathlib import Path
from typing import Any, Literal, TypeAlias
from warnings import warn

from gymnasium import Env

from leap_c.controller import CtxType, ParameterizedController
from leap_c.examples.cartpole.env import CartPoleEnv
from leap_c.examples.cartpole.planner import CartPolePlanner, CartPolePlannerConfig
from leap_c.examples.chain.env import ChainEnv
from leap_c.examples.chain.planner import ChainControllerConfig, ChainPlanner
from leap_c.examples.hvac.dataset import DataConfig, HvacDataset
from leap_c.examples.hvac.env import StochasticThreeStateRcEnv
from leap_c.examples.hvac.planner import HvacPlanner, HvacPlannerConfig
from leap_c.examples.mass_spring_damper.env import MassSpringDamperEnv
from leap_c.examples.mass_spring_damper.planner import (
    MassSpringDamperPlanner,
    MassSpringDamperPlannerConfig,
)
from leap_c.examples.pointmass.env import PointMassEnv
from leap_c.examples.pointmass.planner import PointMassControllerConfig, PointMassPlanner
from leap_c.planner import ControllerFromPlanner, ParameterizedPlanner

ExampleEnvName = Literal[
    "cartpole", "chain", "mass_spring_damper", "pointmass", "hvac", "hvac_continual"
]
ENV_REGISTRY = {
    "cartpole": CartPoleEnv,
    "chain": ChainEnv,
    "mass_spring_damper": MassSpringDamperEnv,
    "pointmass": PointMassEnv,
    "hvac": StochasticThreeStateRcEnv,
    "hvac_continual": lambda **kw: StochasticThreeStateRcEnv(
        dataset=HvacDataset(cfg=DataConfig(mode="continual")), **kw
    ),
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
    "mass_spring_damper": (MassSpringDamperPlanner, MassSpringDamperPlannerConfig, dict()),
    "pointmass": (PointMassPlanner, PointMassControllerConfig, dict()),
    "pointmass_stagewise": (
        PointMassPlanner,
        PointMassControllerConfig,
        {"param_interface": "stagewise"},
    ),
    "hvac": (HvacPlanner, HvacPlannerConfig, dict()),
    "hvac_stagewise": (
        HvacPlanner,
        HvacPlannerConfig,
        {"param_interface": "reference", "param_granularity": "stagewise"},
    ),
}
ExamplePlannerName = Literal[
    "cartpole",
    "cartpole_stagewise",
    "chain",
    "chain_stagewise",
    "mass_spring_damper",
    "pointmass",
    "pointmass_stagewise",
    "hvac",
    "hvac_stagewise",
]


CONTROLLER_REGISTRY = {}
# controllers are a superset of planners
ExampleControllerName: TypeAlias = ExamplePlannerName


def _create_from_registry(
    kind: Literal["planner", "controller"],
    name: str,
    reuse_code_base_dir: Path | None,
    **kwargs: Any,
) -> ParameterizedPlanner[CtxType] | ParameterizedController[CtxType]:
    """Helper to create a planner or controller from the corresponding registry."""
    registry = PLANNER_REGISTRY if kind == "planner" else CONTROLLER_REGISTRY
    if name not in registry:
        raise ValueError(f"{kind.capitalize()} '{name}' is not registered or does not exist.")

    cls, cfg_cls, default_cfg_kwargs = registry[name]
    cfg = cfg_cls(**default_cfg_kwargs, **kwargs)

    kwargs_cls = {"cfg": cfg}
    if reuse_code_base_dir is not None:
        export_directory = reuse_code_base_dir / name
        try:
            return cls(**kwargs_cls, export_directory=export_directory)
        except TypeError as e:
            if "export_directory" not in str(e):
                raise  # unrelated TypeError, re-raise it
            warn(
                f"{cls.__name__} does not support 'export_directory' argument; ignoring "
                f"'reuse_code_base_dir' for this {kind}.",
                RuntimeWarning,
                2,
            )
    return cls(**kwargs_cls)


def create_planner(
    planner_name: ExamplePlannerName, reuse_code_base_dir: Path | None = None, **kwargs: Any
) -> ParameterizedPlanner[CtxType]:
    """Create a planner.

    Args:
        planner_name: Name of the planner.
        reuse_code_base_dir: Directory to reuse code base from, e.g., generated code.
        **kwargs: Additional keyword arguments passed to the planner's config constructor.

    Returns:
        An instance of the requested planner.
    """
    return _create_from_registry("planner", planner_name, reuse_code_base_dir, **kwargs)


def create_controller(
    controller_name: ExampleControllerName, reuse_code_base_dir: Path | None = None, **kwargs: Any
) -> ParameterizedController[CtxType]:
    """Create a controller or create a planner and wrap it as a controller.

    Args:
        controller_name: Name of the controller.
        reuse_code_base_dir: Directory to reuse code base from, e.g., generated code.
        **kwargs: Additional keyword arguments passed to the controller's config constructor.

    Returns:
        An instance of the requested controller.
    """
    if controller_name in PLANNER_REGISTRY:
        planner = create_planner(controller_name, reuse_code_base_dir, **kwargs)
        return ControllerFromPlanner(planner)
    return _create_from_registry("controller", controller_name, reuse_code_base_dir, **kwargs)
