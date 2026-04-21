"""In this module, we manage all the example environments, controllers and planners."""

from importlib import import_module
from pathlib import Path
from typing import Any, Literal, TypeAlias
from warnings import warn

from gymnasium import Env

from leap_c.controller import CtxType, ParameterizedController
from leap_c.planner import ControllerFromPlanner, ParameterizedPlanner

ExampleEnvName = Literal[
    "cartpole",
    "chain",
    "mass_spring_damper",
    "pointmass",
    "hvac",
    "hvac_continual",
    "race_car",
]
ENV_REGISTRY: dict[str, tuple[str, str]] = {
    "cartpole": ("leap_c.examples.cartpole.env", "CartPoleEnv"),
    "chain": ("leap_c.examples.chain.env", "ChainEnv"),
    "mass_spring_damper": ("leap_c.examples.mass_spring_damper.env", "MassSpringDamperEnv"),
    "pointmass": ("leap_c.examples.pointmass.env", "PointMassEnv"),
    "hvac": ("leap_c.examples.hvac.env", "StochasticThreeStateRcEnv"),
    "hvac_continual": ("leap_c.examples.hvac.env", "ContinualStochasticThreeStateRcEnv"),
    "race_car": ("leap_c.examples.race_car.env", "RaceCarEnv"),
}


def create_env(env_name: ExampleEnvName, **kw: Any) -> Env:
    """Create an environment based on the given name.

    Args:
        env_name: Name of the environment.
        **kw: Additional keyword arguments passed to the environment constructor.

    Returns:
        An instance of the requested environment.
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Environment '{env_name}' is not registered.")

    module_path, cls_name = ENV_REGISTRY[env_name]
    module = import_module(module_path)
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise ValueError(f"Class '{cls_name}' not found in module '{module_path}'.")
    return cls(**kw)


PLANNER_REGISTRY: dict[str, tuple[str, str, str, dict[str, Any]]] = {
    "cartpole": (
        "leap_c.examples.cartpole.planner",
        "CartPolePlanner",
        "CartPolePlannerConfig",
        {},
    ),
    "cartpole_stagewise": (
        "leap_c.examples.cartpole.planner",
        "CartPolePlanner",
        "CartPolePlannerConfig",
        {"param_interface": "stagewise"},
    ),
    "chain": ("leap_c.examples.chain.planner", "ChainPlanner", "ChainControllerConfig", {}),
    "chain_stagewise": (
        "leap_c.examples.chain.planner",
        "ChainPlanner",
        "ChainControllerConfig",
        {"param_interface": "stagewise"},
    ),
    "mass_spring_damper": (
        "leap_c.examples.mass_spring_damper.planner",
        "MassSpringDamperPlanner",
        "MassSpringDamperPlannerConfig",
        {},
    ),
    "pointmass": (
        "leap_c.examples.pointmass.planner",
        "PointMassPlanner",
        "PointMassControllerConfig",
        {},
    ),
    "pointmass_stagewise": (
        "leap_c.examples.pointmass.planner",
        "PointMassPlanner",
        "PointMassControllerConfig",
        {"param_interface": "stagewise"},
    ),
    "hvac": ("leap_c.examples.hvac.planner", "HvacPlanner", "HvacPlannerConfig", {}),
    "hvac_stagewise": (
        "leap_c.examples.hvac.planner",
        "HvacPlanner",
        "HvacPlannerConfig",
        {"param_interface": "reference", "param_granularity": "stagewise"},
    ),
    "race_car": (
        "leap_c.examples.race_car.planner",
        "RaceCarPlanner",
        "RaceCarPlannerConfig",
        {},
    ),
    "race_car_stagewise": (
        "leap_c.examples.race_car.planner",
        "RaceCarPlanner",
        "RaceCarPlannerConfig",
        {"param_interface": "stagewise"},
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
    "race_car",
    "race_car_stagewise",
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

    module_path, cls_name, cfg_cls_name, default_cfg_kwargs = registry[name]
    module = import_module(module_path)
    cls = getattr(module, cls_name, None)
    cfg_cls = getattr(module, cfg_cls_name, None)
    if cls is None or cfg_cls is None:
        raise ValueError(
            f"Planner class '{cls_name}' or config class '{cfg_cls_name}' not found in module "
            f"'{module_path}'."
        )

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
