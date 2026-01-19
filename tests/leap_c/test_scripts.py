import os
import shutil
import tempfile
import types
from pathlib import Path

import pytest

import leap_c
from leap_c.examples import CONTROLLER_REGISTRY, ENV_REGISTRY, PLANNER_REGISTRY

if getattr(leap_c, "__file__", None):
    LEAP_C_ROOT = Path(leap_c.__file__).resolve().parent.parent
else:
    LEAP_C_ROOT = Path(next(iter(leap_c.__path__))).resolve().parent
LEAP_C_ROOT_SCRIPTS = LEAP_C_ROOT / "scripts"
CTRL_REGISTRY = CONTROLLER_REGISTRY | PLANNER_REGISTRY


def find_all_scripts():
    scripts = {}
    for script_path in LEAP_C_ROOT_SCRIPTS.glob("run_*"):
        if script_path.is_file():
            script_name = script_path.stem[4:]  # Remove 'run_' prefix
            scripts[script_name] = script_path
    return scripts


def _create_script_module(script_name: str):
    script_path = LEAP_C_ROOT_SCRIPTS / f"run_{script_name}.py"

    if not script_path.exists():
        raise ValueError(f"Script {script_name} does not exist at {script_path}")

    script_module = types.ModuleType(f"leap_c_test_scripts.{script_name}")
    with open(script_path, "r") as f:
        exec(f.read(), script_module.__dict__)

    return script_module


def create_cfg(
    script_name: str, env: str = "cartpole", seed: int = 13, controller: str | None = None
):
    """Returns the according script config dataclass."""
    script_module = _create_script_module(script_name)
    create_cfg_fn = getattr(script_module, "create_cfg", None)

    if create_cfg_fn is None:
        raise ValueError(f"Script {script_name} does not have a create_cfg function")

    # call the create_cfg function to get the config
    if controller is None:
        cfg = create_cfg_fn(env=env, seed=seed)
    else:
        cfg = create_cfg_fn(env=env, seed=seed, controller=controller)

    return cfg


def run_script(script_name: str, cfg, **kw):
    """Runs the script with the given name and arguments.

    We assume that the method is called `run_{script_name}`.
    """
    script_module = _create_script_module(script_name)
    run_fn = getattr(script_module, f"run_{script_name}", None)

    if run_fn is None:
        raise ValueError(f"Script {script_name} does not have a run_{script_name} function")

    # Call the run function with the config
    return run_fn(cfg, **kw)


def _filter_scripts(scripts):
    """Filter scripts based on TEST_SCRIPT environment variable."""
    test_script_env = os.environ.get("TEST_SCRIPT", "")
    test_script_filter = test_script_env.split(",") if test_script_env else None
    if test_script_filter:
        return [s for s in scripts if s in test_script_filter]
    return scripts


@pytest.fixture(params=_filter_scripts(["sac", "sac_zop", "sac_fop", "baseline"]))
def script(request):
    return request.param


@pytest.fixture(params=_filter_scripts(["sac"]))
def script_without_ctrl(request):
    return request.param


@pytest.fixture(params=_filter_scripts(["sac_zop", "sac_fop", "baseline"]))
def script_with_ctrl(request):
    return request.param


@pytest.fixture(scope="module")
def reuse_code_dir():
    path = Path(tempfile.mkdtemp(prefix="leapc_session_tmp_"))

    yield path  # This is the temp dir you can use

    # Cleanup after the session
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def scripts_dict():
    return find_all_scripts()


def _get_envs():
    """Get environments, optionally filtered by TEST_ENV environment variable."""
    test_env = os.environ.get("TEST_ENV", "")
    test_env_filter = test_env.split(",") if test_env else None

    envs = list(ENV_REGISTRY.keys())
    if test_env_filter:
        envs = [e for e in envs if e in test_env_filter]

    return envs


@pytest.fixture(params=_get_envs())
def env(request):
    return request.param


def _get_env_controller_pairs():
    """Generate env-controller pairs, optionally filtered by environment variables."""
    # Get filter from environment variables (comma-separated for multiple values)
    test_env = os.environ.get("TEST_ENV", "")
    test_env_filter = test_env.split(",") if test_env else None
    test_ctrl = os.environ.get("TEST_CONTROLLER", "")
    test_controller_filter = test_ctrl.split(",") if test_ctrl else None

    pairs = []
    for env in ENV_REGISTRY.keys():
        for controller in CTRL_REGISTRY.keys():
            if controller.startswith(env):
                # Apply filters if set
                if test_env_filter and env not in test_env_filter:
                    continue
                if test_controller_filter and controller not in test_controller_filter:
                    continue
                pairs.append((env, controller))

    return pairs


@pytest.fixture(
    params=_get_env_controller_pairs(),
    ids=[f"{env}-{controller}" for env, controller in _get_env_controller_pairs()],
)
def env_controller_pair(request):
    """Yields (env, controller) pairs where controller matches the env prefix."""
    return request.param


def test_find_all_scripts(script, scripts_dict):
    assert script in scripts_dict, f"{script} script not found"


def _run_script(script, tmp_path, env, reuse_code_dir, controller=None):
    if (
        controller
        in [
            # "cartpole",
            # "cartpole_stagewise",
            # "chain",
            # "chain_stagewise",
            # "pointmass",
            # "pointmass_stagewise",
            # "hvac",
            # "hvac_stagewise",
        ]
    ):
        pytest.skip(f"{controller} controller")

    cfg = create_cfg(script, env=env, controller=controller)
    cfg.log = False
    cfg.trainer.train_steps = 5
    cfg.trainer.train_start = 0
    cfg.trainer.update_freq = 1
    cfg.trainer.batch_size = 4
    cfg.trainer.val_num_rollouts = 1
    cfg.trainer.val_num_render_rollouts = 0

    kw = {
        "output_path": tmp_path,
        "device": "cpu",  # Use CPU for testing
    }

    if controller is not None:
        kw["reuse_code_dir"] = reuse_code_dir
        cfg.controller = controller

    run_script(script, cfg, **kw)


def test_run_script_with_ctrl(tmp_path, script_with_ctrl, env_controller_pair, reuse_code_dir):
    env, controller = env_controller_pair
    _run_script(script_with_ctrl, tmp_path, env, reuse_code_dir, controller=controller)


def test_run_script(tmp_path, script_without_ctrl, env, reuse_code_dir):
    _run_script(script_without_ctrl, tmp_path, env, reuse_code_dir)
