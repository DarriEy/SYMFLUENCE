# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for LISFLOOD model runner."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from symfluence.models.lisflood.runner import LisfloodRunner, _find_pcraster_site_packages


class TestFindPcrasterSitePackages:
    """Tests for the _find_pcraster_site_packages helper."""

    def test_returns_string_or_none(self):
        """The function should return a string path or None."""
        result = _find_pcraster_site_packages()
        assert result is None or isinstance(result, str)

    @patch("symfluence.models.lisflood.runner._conda_env_site_packages")
    def test_returns_path_when_conda_env_found(self, mock_conda_site):
        fake_path = "/opt/conda/envs/pcraster312/lib/python3.12/site-packages"
        mock_conda_site.return_value = fake_path
        result = _find_pcraster_site_packages()
        # pcraster is not importable in test env, so it should use the conda path
        assert result == fake_path

    @patch("symfluence.models.lisflood.runner._conda_env_site_packages", return_value=None)
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_handles_missing_conda(self, mock_run, mock_conda_site):
        """Should not raise if conda is not installed."""
        result = _find_pcraster_site_packages()
        assert result is None or isinstance(result, str)


class TestLisfloodRunnerImport:
    """Tests for LisfloodRunner importability."""

    def test_runner_can_be_imported(self):
        assert LisfloodRunner is not None

    def test_model_name(self):
        assert LisfloodRunner.MODEL_NAME == "LISFLOOD"


class TestLisfloodRunnerInit:
    """Tests for LisfloodRunner initialization."""

    def test_runner_initialization(self, lisflood_config, mock_logger, setup_lisflood_directories):
        runner = LisfloodRunner(lisflood_config, mock_logger)
        assert runner is not None

    def test_settings_dir_is_lisflood(self, lisflood_config, mock_logger, setup_lisflood_directories):
        runner = LisfloodRunner(lisflood_config, mock_logger)
        assert runner.settings_dir.name == "LISFLOOD"
        assert "settings" in str(runner.settings_dir)


class TestLisfloodRunnerEnvironment:
    """Tests for runner environment setup."""

    def test_sets_conda_prefix_when_pcraster_found(self, lisflood_config, mock_logger, setup_lisflood_directories):
        """When pcraster site-packages is found, CONDA_PREFIX should be set."""
        runner = LisfloodRunner(lisflood_config, mock_logger)

        # Create a fake settings file
        settings_path = runner.settings_dir / "settings.xml"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("<lfsettings/>")

        fake_site = "/opt/conda/envs/pcraster312/lib/python3.12/site-packages"
        with patch("symfluence.models.lisflood.runner._find_pcraster_site_packages", return_value=fake_site):
            with patch.object(runner, "execute_subprocess") as mock_exec:
                mock_exec.return_value = Mock(success=True, returncode=0)
                runner.run()

                # Check the env passed to execute_subprocess
                call_kwargs = mock_exec.call_args
                env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env", {})
                assert env.get("CONDA_PREFIX") == "/opt/conda/envs/pcraster312"

    def test_removes_pcraster_threads_when_num_threads_1(
        self, lisflood_config, mock_logger, setup_lisflood_directories
    ):
        """When num_threads=1 (default), PCRASTER_NR_WORKER_THREADS should be removed."""
        runner = LisfloodRunner(lisflood_config, mock_logger)

        settings_path = runner.settings_dir / "settings.xml"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("<lfsettings/>")

        # Set the env var before the runner clears it
        with patch.dict(os.environ, {"PCRASTER_NR_WORKER_THREADS": "4"}):
            with patch("symfluence.models.lisflood.runner._find_pcraster_site_packages", return_value=None):
                with patch.object(runner, "execute_subprocess") as mock_exec:
                    mock_exec.return_value = Mock(success=True, returncode=0)
                    runner.run()

                    call_kwargs = mock_exec.call_args
                    env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env", {})
                    assert "PCRASTER_NR_WORKER_THREADS" not in env


class TestLisfloodRunnerCommand:
    """Tests for runner command construction."""

    def test_constructs_python_c_command(self, lisflood_config, mock_logger, setup_lisflood_directories):
        runner = LisfloodRunner(lisflood_config, mock_logger)

        settings_path = runner.settings_dir / "settings.xml"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("<lfsettings/>")

        with patch("symfluence.models.lisflood.runner._find_pcraster_site_packages", return_value=None):
            with patch.object(runner, "execute_subprocess") as mock_exec:
                mock_exec.return_value = Mock(success=True, returncode=0)
                runner.run()

                call_args = mock_exec.call_args
                cmd = call_args.kwargs.get("command") or call_args[0][0]
                assert cmd[0] == sys.executable
                assert cmd[1] == "-c"
                assert "lisflood.main" in cmd[2]
                assert "main(" in cmd[2]


class TestLisfloodRunnerErrors:
    """Tests for runner error handling."""

    def test_raises_file_not_found_for_missing_settings(self, lisflood_config, mock_logger, setup_lisflood_directories):
        runner = LisfloodRunner(lisflood_config, mock_logger)
        # Do NOT create settings.xml
        with pytest.raises(FileNotFoundError, match="settings not found"):
            runner.run()
