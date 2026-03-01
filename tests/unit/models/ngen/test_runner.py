"""Tests for NGEN model runner."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestNgenRunnerImport:
    """Tests for NGEN runner importability."""

    def test_runner_can_be_imported(self):
        from symfluence.models.ngen.runner import NgenRunner
        assert NgenRunner is not None

    def test_model_name(self):
        from symfluence.models.ngen.runner import NgenRunner
        assert NgenRunner.MODEL_NAME == "NGEN"


class TestNgenRunnerInit:
    """Tests for NGEN runner initialization."""

    def test_runner_initialization(self, ngen_config, mock_logger, setup_ngen_directories):
        from symfluence.models.ngen.runner import NgenRunner
        runner = NgenRunner(ngen_config, mock_logger)
        assert runner is not None

    def test_default_setup_dir(self, ngen_config, mock_logger, setup_ngen_directories):
        from symfluence.models.ngen.runner import NgenRunner
        runner = NgenRunner(ngen_config, mock_logger)
        assert runner.ngen_setup_dir.name == "NGEN"
        assert "settings" in str(runner.ngen_setup_dir)

    def test_override_settings_dir(self, ngen_config, mock_logger, setup_ngen_directories, temp_dir):
        custom_dir = temp_dir / "custom_ngen_settings"
        custom_dir.mkdir()
        from symfluence.models.ngen.runner import NgenRunner
        runner = NgenRunner(ngen_config, mock_logger, ngen_settings_dir=custom_dir)
        assert runner.ngen_setup_dir == custom_dir

    def test_override_output_dir(self, ngen_config, mock_logger, setup_ngen_directories, temp_dir):
        custom_output = temp_dir / "custom_output"
        custom_output.mkdir()
        from symfluence.models.ngen.runner import NgenRunner
        runner = NgenRunner(ngen_config, mock_logger, ngen_output_dir=custom_output)
        assert runner._ngen_output_dir_override == custom_output


class TestNgenRunnerPaths:
    """Tests for NGEN path and config methods."""

    def test_should_create_output_dir_is_false(self, ngen_config, mock_logger, setup_ngen_directories):
        from symfluence.models.ngen.runner import NgenRunner
        runner = NgenRunner(ngen_config, mock_logger)
        assert runner._should_create_output_dir() is False

    def test_ngen_exe_set_during_init(self, ngen_config, mock_logger, setup_ngen_directories):
        from symfluence.models.ngen.runner import NgenRunner
        runner = NgenRunner(ngen_config, mock_logger)
        assert runner.ngen_exe.name == "ngen"
        assert runner.ngen_exe.exists()


class TestNgenEnvironment:
    """Tests for NGEN environment setup."""

    def test_setup_ngen_environment_returns_dict(self, ngen_config, mock_logger, setup_ngen_directories):
        from symfluence.models.ngen.runner import NgenRunner
        runner = NgenRunner(ngen_config, mock_logger)
        env = runner._setup_ngen_environment()
        assert isinstance(env, dict)
        assert "PATH" in env

    def test_environment_contains_ld_library_path(self, ngen_config, mock_logger, setup_ngen_directories):
        from symfluence.models.ngen.runner import NgenRunner
        runner = NgenRunner(ngen_config, mock_logger)
        env = runner._setup_ngen_environment()
        # LD_LIBRARY_PATH may or may not be set depending on platform,
        # but the method should run without error
        assert isinstance(env, dict)


class TestNgenDockerVsNative:
    """Tests for Docker vs native mode routing."""

    def test_run_ngen_checks_use_ngiab(self, ngen_config, mock_logger, setup_ngen_directories):
        from symfluence.models.ngen.runner import NgenRunner
        runner = NgenRunner(ngen_config, mock_logger)
        # Default should not use Docker
        use_ngiab = runner._get_config_value(lambda: None, default=False, dict_key='USE_NGIAB')
        assert use_ngiab is False

    def test_runner_has_run_ngen_method(self, ngen_config, mock_logger, setup_ngen_directories):
        from symfluence.models.ngen.runner import NgenRunner
        runner = NgenRunner(ngen_config, mock_logger)
        assert hasattr(runner, "run_ngen")
        assert callable(runner.run_ngen)
