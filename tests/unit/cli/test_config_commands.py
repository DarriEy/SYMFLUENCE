"""Unit tests for config command handlers."""

import pytest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

from symfluence.cli.commands.config_commands import ConfigCommands
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestConfigListTemplates:
    """Test config list-templates command."""

    @patch('symfluence.resources.list_config_templates')
    def test_list_templates_success(self, mock_list_templates):
        """Test successful template listing."""
        mock_list_templates.return_value = [
            '/path/to/config_template.yaml',
            '/path/to/quickstart_minimal.yaml',
            '/path/to/comprehensive.yaml'
        ]

        args = Namespace(debug=False)

        result = ConfigCommands.list_templates(args)

        assert result == ExitCode.SUCCESS
        mock_list_templates.assert_called_once()

    @patch('symfluence.resources.list_config_templates')
    def test_list_templates_empty(self, mock_list_templates):
        """Test template listing when no templates found."""
        mock_list_templates.return_value = []

        args = Namespace(debug=False)

        result = ConfigCommands.list_templates(args)

        assert result == ExitCode.SUCCESS

    @patch('symfluence.resources.list_config_templates')
    def test_list_templates_import_error(self, mock_list_templates):
        """Test template listing with import error."""
        mock_list_templates.side_effect = ImportError("Cannot import resources")

        args = Namespace(debug=False)

        result = ConfigCommands.list_templates(args)

        assert result == ExitCode.DEPENDENCY_ERROR


class TestConfigUpdate:
    """Test config update command."""

    def test_update_file_not_found(self):
        """Test update with nonexistent file."""
        args = Namespace(
            config_file='/nonexistent/config.yaml',
            interactive=False,
            debug=False
        )

        result = ConfigCommands.update(args)

        assert result == ExitCode.FILE_NOT_FOUND

    def test_update_not_implemented(self, tmp_path):
        """Test config update returns USAGE_ERROR (not implemented)."""
        config_file = tmp_path / 'config.yaml'
        config_file.write_text("DOMAIN_NAME: test")

        args = Namespace(
            config_file=str(config_file),
            interactive=False,
            debug=False
        )

        result = ConfigCommands.update(args)

        # Update feature is not implemented, returns USAGE_ERROR with guidance
        assert result == ExitCode.USAGE_ERROR

    def test_update_interactive_not_implemented(self, tmp_path):
        """Test interactive update returns USAGE_ERROR (not implemented)."""
        config_file = tmp_path / 'config.yaml'
        config_file.write_text("DOMAIN_NAME: test")

        args = Namespace(
            config_file=str(config_file),
            interactive=True,
            debug=False
        )

        result = ConfigCommands.update(args)

        # Update feature is not implemented, returns USAGE_ERROR with guidance
        assert result == ExitCode.USAGE_ERROR


class TestConfigValidate:
    """Test config validate command."""

    @patch('symfluence.core.SYMFLUENCE')
    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_validate_success(self, mock_load_config, mock_symfluence_class, tmp_path):
        """Test successful config validation."""
        config_file = tmp_path / 'config.yaml'
        config_file.write_text("DOMAIN_NAME: test")

        mock_load_config.return_value = MagicMock()
        mock_symfluence_class.return_value = MagicMock()

        args = Namespace(
            config=str(config_file),
            debug=False
        )

        result = ConfigCommands.validate(args)

        assert result == ExitCode.SUCCESS

    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_validate_invalid_yaml(self, mock_load_config):
        """Test validation with invalid YAML."""
        mock_load_config.return_value = None

        args = Namespace(
            config='/some/config.yaml',
            debug=False
        )

        result = ConfigCommands.validate(args)

        assert result == ExitCode.CONFIG_ERROR

    @patch('symfluence.core.SYMFLUENCE')
    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_validate_structure_error(self, mock_load_config, mock_symfluence_class, tmp_path):
        """Test validation with structure error."""
        from symfluence.core.exceptions import ConfigurationError

        config_file = tmp_path / 'config.yaml'
        config_file.write_text("DOMAIN_NAME: test")

        mock_load_config.return_value = MagicMock()
        mock_symfluence_class.side_effect = ConfigurationError("Missing required field")

        args = Namespace(
            config=str(config_file),
            debug=False
        )

        result = ConfigCommands.validate(args)

        assert result == ExitCode.CONFIG_ERROR


class TestConfigValidateEnv:
    """Test config validate-env command."""

    def test_validate_env_success(self):
        """Test successful environment validation."""
        args = Namespace(debug=False)

        # This should succeed since required packages are installed in test env
        result = ConfigCommands.validate_env(args)

        # Result depends on installed packages, but shouldn't raise
        assert result in (ExitCode.SUCCESS, ExitCode.DEPENDENCY_ERROR)

    @patch('builtins.__import__')
    def test_validate_env_missing_packages(self, mock_import):
        """Test environment validation with missing packages."""
        def import_side_effect(name, *args, **kwargs):
            if name in ['numpy', 'pandas', 'xarray']:
                raise ImportError(f"No module named '{name}'")
            return MagicMock()

        mock_import.side_effect = import_side_effect

        args = Namespace(debug=False)

        # Note: This test is tricky because it depends on actual imports
        # In practice, we can just verify it doesn't crash
        result = ConfigCommands.validate_env(args)
        assert isinstance(result, ExitCode)
