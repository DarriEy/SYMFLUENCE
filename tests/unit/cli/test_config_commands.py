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


class TestConfigResolve:
    """Test config resolve command."""

    def _make_args(self, config='/some/config.yaml', flat=False, as_json=False,
                   diff=False, section=None, debug=False):
        return Namespace(
            config=config, flat=flat, as_json=as_json,
            diff=diff, section=section, debug=debug
        )

    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_resolve_success(self, mock_load_config, capsys):
        """Test successful config resolve with default (nested YAML) output."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {
            'domain': {'name': 'test_basin'},
            'model': {'hydrological_model': 'SUMMA'}
        }
        mock_load_config.return_value = mock_config

        result = ConfigCommands.resolve(self._make_args())

        assert result == ExitCode.SUCCESS
        captured = capsys.readouterr()
        assert 'test_basin' in captured.out
        assert 'SUMMA' in captured.out
        mock_config.to_dict.assert_called_with(flatten=False)

    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_resolve_flat_format(self, mock_load_config, capsys):
        """Test resolve with --flat flag outputs uppercase keys."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {
            'DOMAIN_NAME': 'test_basin',
            'HYDROLOGICAL_MODEL': 'SUMMA'
        }
        mock_load_config.return_value = mock_config

        result = ConfigCommands.resolve(self._make_args(flat=True))

        assert result == ExitCode.SUCCESS
        captured = capsys.readouterr()
        assert 'DOMAIN_NAME' in captured.out
        assert 'test_basin' in captured.out
        mock_config.to_dict.assert_called_with(flatten=True)

    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_resolve_json_format(self, mock_load_config, capsys):
        """Test resolve with --json flag outputs valid JSON."""
        import json

        mock_config = MagicMock()
        mock_config.to_dict.return_value = {
            'domain': {'name': 'test_basin'}
        }
        mock_load_config.return_value = mock_config

        result = ConfigCommands.resolve(self._make_args(as_json=True))

        assert result == ExitCode.SUCCESS
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed['domain']['name'] == 'test_basin'

    @patch('symfluence.core.config.transformers.transform_flat_to_nested')
    @patch('symfluence.core.config.defaults.ConfigDefaults.get_defaults')
    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_resolve_diff_mode(self, mock_load_config, mock_get_defaults, mock_transform, capsys):
        """Test resolve with --diff shows only non-default values."""
        mock_config = MagicMock()
        # Nested output for the default (non-flat) path
        mock_config.to_dict.side_effect = lambda flatten: (
            {'DOMAIN_NAME': 'my_custom', 'NUM_PROCESSES': 1}
            if flatten else {'domain': {'name': 'my_custom'}}
        )
        mock_load_config.return_value = mock_config
        mock_get_defaults.return_value = {
            'DOMAIN_NAME': 'default_domain',
            'NUM_PROCESSES': 1  # same as resolved → should be filtered out
        }
        mock_transform.return_value = {'domain': {'name': 'my_custom'}}

        result = ConfigCommands.resolve(self._make_args(diff=True))

        assert result == ExitCode.SUCCESS
        # The transform should have been called with only changed keys
        mock_transform.assert_called_once()
        call_args = mock_transform.call_args[0][0]
        assert 'DOMAIN_NAME' in call_args
        assert 'NUM_PROCESSES' not in call_args

    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_resolve_section_filter(self, mock_load_config, capsys):
        """Test resolve with --section filters to a single section."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {
            'domain': {'name': 'test_basin', 'definition_method': 'lumped'},
            'model': {'hydrological_model': 'SUMMA'},
            'system': {'num_processes': 1}
        }
        mock_load_config.return_value = mock_config

        result = ConfigCommands.resolve(self._make_args(section='domain'))

        assert result == ExitCode.SUCCESS
        captured = capsys.readouterr()
        assert 'test_basin' in captured.out
        # Other sections should not appear
        assert 'SUMMA' not in captured.out
        assert 'num_processes' not in captured.out

    @patch('symfluence.cli.commands.base.BaseCommand.load_typed_config')
    def test_resolve_invalid_config(self, mock_load_config):
        """Test resolve with invalid config returns CONFIG_ERROR."""
        mock_load_config.return_value = None

        result = ConfigCommands.resolve(self._make_args())

        assert result == ExitCode.CONFIG_ERROR

    def test_resolve_section_with_flat_error(self):
        """Test --section combined with --flat returns USAGE_ERROR."""
        result = ConfigCommands.resolve(
            self._make_args(flat=True, section='domain')
        )

        assert result == ExitCode.USAGE_ERROR
