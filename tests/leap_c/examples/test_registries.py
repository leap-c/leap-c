"""Collections of tests for env, planner, and controller registries.

Ensure all declared classes are present in registries and are found for instantiation. Does not test
instantiation of the classes themselves, just that the registries are correctly set up and used.
"""

from typing import get_args
from unittest.mock import MagicMock, Mock, patch

import pytest

from leap_c.examples import (
    CONTROLLER_REGISTRY,
    ENV_REGISTRY,
    PLANNER_REGISTRY,
    ExampleEnvName,
    ExamplePlannerName,
    create_controller,
    create_env,
    create_planner,
)


class TestEnvs:
    def test_env_registry(self) -> None:
        """Tests the entries of `ExampleEnvName` are as registered."""
        assert set(get_args(ExampleEnvName)) == set(ENV_REGISTRY), (
            "Env name mismatch between `ExampleEnvName` and `ENV_REGISTRY`."
        )

    def test_create_env__raises__with_invalid_name(self) -> None:
        """Tests that `create_env` raises a `ValueError` when given an invalid env name."""
        wrong_name = "invalid_env"
        with pytest.raises(ValueError, match=f"Environment '{wrong_name}' is not registered."):
            create_env(wrong_name)

    @pytest.mark.parametrize("env_name", ENV_REGISTRY)
    def test_create_env__calls_constructor_correctly(self, env_name: str) -> None:
        """Tests that `create_env` calls the correct constructor for each env."""
        module_path, cls_name = ENV_REGISTRY[env_name]
        kwargs = {"foo": object(), "bar": object()}
        with patch(f"{module_path}.{cls_name}", return_value=Mock()) as mock_cls:
            env: Mock = create_env(env_name, **kwargs)

            mock_env = mock_cls.return_value

            mock_cls.assert_called_once_with(**kwargs)
            assert env is mock_env


class TestPlanners:
    def test_planner_registry(self) -> None:
        """Tests that the entries of `ExamplePlannerName` are as registered."""
        assert set(get_args(ExamplePlannerName)) == set(PLANNER_REGISTRY), (
            "Planner name mismatch between `ExampleEnvName` and `PLANNER_REGISTRY`."
        )

    def test_create_planner__raises__with_invalid_name(self) -> None:
        """Tests that `create_planner` raises a `ValueError` when given an invalid planner name."""
        wrong_name = "invalid_env"
        match = f"Planner '{wrong_name}' is not registered or does not exist."
        with pytest.raises(ValueError, match=match):
            create_planner(wrong_name)

    @pytest.mark.parametrize("planner_name", PLANNER_REGISTRY)
    def test_create_planner__calls_constructor_correctly(self, planner_name: str) -> None:
        """Tests that `create_planner` calls the correct constructor for each planner and cfg."""
        module_path, cls_name, cfg_cls_name, default_kwargs = PLANNER_REGISTRY[planner_name]
        reuse_code_base_dir = MagicMock()
        kwargs = {"foo": object(), "bar": object()}
        with (
            patch(f"{module_path}.{cls_name}", return_value=Mock()) as mock_cls,
            patch(f"{module_path}.{cfg_cls_name}", return_value=Mock()) as mock_cfg_cls,
        ):
            planner: Mock = create_planner(planner_name, reuse_code_base_dir, **kwargs)

            mock_planner = mock_cls.return_value
            mock_cfg = mock_cfg_cls.return_value
            mock_export_directory = reuse_code_base_dir.__truediv__()

            mock_cfg_cls.assert_called_once_with(**default_kwargs, **kwargs)
            mock_cls.assert_called_once_with(cfg=mock_cfg, export_directory=mock_export_directory)
            assert planner is mock_planner


class TestControllers:
    def test_controller_registry(self) -> None:
        """Tests that the entries of `ExampleControllerName` are as registered."""
        registered = set(CONTROLLER_REGISTRY).union(PLANNER_REGISTRY)
        assert set(get_args(ExamplePlannerName)) == registered, (
            "Controller name mismatch between `ExampleControllerName` and "
            "`CONTROLLER_REGISTRY + PLANNER_REGISTRY`."
        )

    def test_create_controller__raises__with_invalid_name(self) -> None:
        """Tests that `create_planner` raises a `ValueError` when given an invalid ctrl name."""
        wrong_name = "invalid_env"
        match = f"Controller '{wrong_name}' is not registered or does not exist."
        with pytest.raises(ValueError, match=match):
            create_controller(wrong_name)

    @pytest.mark.parametrize("planner_name", PLANNER_REGISTRY)
    def test_create_controller__delegates_correctly_to_create_planner__with_planner_name(
        self, planner_name: str
    ) -> None:
        """Tests `create_controller` delegates to `create_planner` when given a planner name."""
        reuse_code_base_dir = object()
        kwargs = {"foo": object(), "bar": object()}
        with (
            patch("leap_c.examples.create_planner", return_value=Mock()) as mock_create_planner,
            patch("leap_c.examples.ControllerFromPlanner", return_value=Mock()) as mock_CtrlFromPln,
        ):
            ctrl: Mock = create_controller(planner_name, reuse_code_base_dir, **kwargs)

            mock_ctrl = mock_CtrlFromPln.return_value
            mock_planner = mock_create_planner.return_value
            mock_create_planner.assert_called_once_with(planner_name, reuse_code_base_dir, **kwargs)
            mock_CtrlFromPln.assert_called_once_with(mock_planner)
            assert ctrl is mock_ctrl

    @pytest.mark.parametrize("controller_name", CONTROLLER_REGISTRY)
    def test_create_controller__calls_constructor_correctly(self, controller_name: str) -> None:
        """Tests that `create_controller` calls the correct constructor for each ctrl and cfg."""
        module_path, cls_name, cfg_cls_name, default_kwargs = CONTROLLER_REGISTRY[controller_name]
        reuse_code_base_dir = MagicMock()
        kwargs = {"foo": object(), "bar": object()}
        with (
            patch(f"{module_path}.{cls_name}", return_value=Mock()) as mock_cls,
            patch(f"{module_path}.{cfg_cls_name}", return_value=Mock()) as mock_cfg_cls,
        ):
            planner: Mock = create_controller(controller_name, reuse_code_base_dir, **kwargs)

            mock_planner = mock_cls.return_value
            mock_cfg = mock_cfg_cls.return_value
            mock_export_directory = reuse_code_base_dir.__truediv__()

            mock_cfg_cls.assert_called_once_with(**default_kwargs, **kwargs)
            mock_cls.assert_called_once_with(cfg=mock_cfg, export_directory=mock_export_directory)
            assert planner is mock_planner
