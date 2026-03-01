"""Tests for HYPE model runner."""

from unittest.mock import patch

import pytest


class TestHYPERunnerImport:
    """Tests for HYPE runner importability."""

    def test_runner_can_be_imported(self):
        from symfluence.models.hype.runner import HYPERunner
        assert HYPERunner is not None

    def test_model_name(self):
        from symfluence.models.hype.runner import HYPERunner
        assert HYPERunner.MODEL_NAME == "HYPE"


class TestHYPERunnerInit:
    """Tests for HYPE runner initialization."""

    def test_runner_initialization(self, hype_config, mock_logger, setup_hype_directories):
        from symfluence.models.hype.runner import HYPERunner
        runner = HYPERunner(hype_config, mock_logger)
        assert runner is not None

    def test_setup_dir_is_hype_settings(self, hype_config, mock_logger, setup_hype_directories):
        from symfluence.models.hype.runner import HYPERunner
        runner = HYPERunner(hype_config, mock_logger)
        assert runner.setup_dir.name == "HYPE"
        assert "settings" in str(runner.setup_dir)


class TestHYPERunnerPaths:
    """Tests for HYPE runner path methods."""

    def test_build_run_command_returns_list(self, hype_config, mock_logger, setup_hype_directories):
        from symfluence.models.hype.runner import HYPERunner
        runner = HYPERunner(hype_config, mock_logger)
        cmd = runner._build_run_command()
        assert isinstance(cmd, list)
        assert len(cmd) == 2

    def test_build_run_command_ends_with_slash(self, hype_config, mock_logger, setup_hype_directories):
        from symfluence.models.hype.runner import HYPERunner
        runner = HYPERunner(hype_config, mock_logger)
        cmd = runner._build_run_command()
        assert cmd[1].endswith("/")

    def test_get_output_dir_contains_hype(self, hype_config, mock_logger, setup_hype_directories):
        from symfluence.models.hype.runner import HYPERunner
        runner = HYPERunner(hype_config, mock_logger)
        output_dir = runner._get_output_dir()
        assert "HYPE" in str(output_dir)

    def test_get_expected_outputs_is_empty(self, hype_config, mock_logger, setup_hype_directories):
        from symfluence.models.hype.runner import HYPERunner
        runner = HYPERunner(hype_config, mock_logger)
        assert runner._get_expected_outputs() == []
