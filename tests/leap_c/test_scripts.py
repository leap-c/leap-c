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

    We assume that the method is called `run_{script_name}`."""
    script_module = _create_script_module(script_name)
    run_fn = getattr(script_module, f"run_{script_name}", None)

    if run_fn is None:
        raise ValueError(f"Script {script_name} does not have a run_{script_name} function")

    # Call the run function with the config
    return run_fn(cfg, **kw)


@pytest.fixture(params=["sac", "sac_zop", "sac_fop", "controller"])
def script(request):
    return request.param


@pytest.fixture(params=["sac"])
def script_without_ctrl(request):
    return request.param


@pytest.fixture(params=["sac_zop", "sac_fop", "controller"])
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


@pytest.fixture(params=ENV_REGISTRY.keys())
def env(request):
    return request.param


@pytest.fixture(
    params=[
        (env, controller)
        for env in ENV_REGISTRY.keys()
        for controller in CTRL_REGISTRY.keys()
        if controller.startswith(env)
    ],
    ids=[
        f"{env}-{controller}"
        for env in ENV_REGISTRY.keys()
        for controller in CTRL_REGISTRY.keys()
        if controller.startswith(env)
    ],
)
def env_controller_pair(request):
    """Yields (env, controller) pairs where controller matches the env prefix."""
    return request.param


def test_find_all_scripts(script, scripts_dict):
    assert script in scripts_dict, f"{script} script not found"


def _run_script(script, tmp_path, env, reuse_code_dir, controller=None):
    if controller == "hvac_stagewise":
        # TODO (Dirk): Fix HVAC stagewise for testing
        pytest.skip("HVAC stagewise is not supported in this test")

    cfg = create_cfg(script, env=env, controller=controller)
    cfg.log = False
    cfg.trainer.train_steps = 10
    cfg.trainer.train_start = 0
    cfg.trainer.update_freq = 1
    cfg.trainer.batch_size = 8
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
