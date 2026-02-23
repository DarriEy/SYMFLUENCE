"""Unit tests for base command class."""

import os
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.cli.commands.base import DEFAULT_CONFIG_PATH, BaseCommand
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestDefaultConfigPath:
    """Test default config path handling."""

    def test_default_config_path_exists(self):
        """Test that DEFAULT_CONFIG_PATH is defined."""
        assert DEFAULT_CONFIG_PATH is not None
        assert isinstance(DEFAULT_CONFIG_PATH, str)

    def test_default_config_path_value(self):
        """Test default config path value when env var not set."""
        # If SYMFLUENCE_DEFAULT_CONFIG is not set, should use default
        if 'SYMFLUENCE_DEFAULT_CONFIG' not in os.environ:
            assert DEFAULT_CONFIG_PATH == './0_config_files/config_template.yaml'


class TestGetConfigPath:
    """Test get_config_path method."""

    def test_get_config_path_from_args(self):
        """Test getting config path from args."""
        args = Namespace(config='/custom/path/config.yaml')

        result = BaseCommand.get_config_path(args)

        assert result == '/custom/path/config.yaml'

    def test_get_config_path_fallback_to_default(self):
        """Test fallback to default config path."""
        args = Namespace(config=None)

        result = BaseCommand.get_config_path(args)

        assert result == DEFAULT_CONFIG_PATH

    def test_get_config_path_no_config_attr(self):
        """Test fallback when config attribute missing."""
        args = Namespace()  # No config attribute

        result = BaseCommand.get_config_path(args)

        assert result == DEFAULT_CONFIG_PATH


class TestValidateConfig:
    """Test validate_config method."""

    def test_validate_config_exists(self, tmp_path):
        """Test validation of existing config file."""
        config_file = tmp_path / 'config.yaml'
        config_file.write_text("DOMAIN_NAME: test")

        result = BaseCommand.validate_config(str(config_file), required=True)

        assert result is True

    def test_validate_config_not_exists_required(self):
        """Test validation of missing required config."""
        result = BaseCommand.validate_config('/nonexistent/config.yaml', required=True)

        assert result is False

    def test_validate_config_not_exists_not_required(self):
        """Test validation of missing non-required config."""
        result = BaseCommand.validate_config('/nonexistent/config.yaml', required=False)

        assert result is True


class TestLoadTypedConfig:
    """Test load_typed_config method."""

    @patch('symfluence.core.config.models.SymfluenceConfig')
    def test_load_typed_config_success(self, mock_config_class, tmp_path):
        """Test successful config loading."""
        config_file = tmp_path / 'config.yaml'
        config_file.write_text("DOMAIN_NAME: test")

        mock_config = MagicMock()
        mock_config_class.from_file.return_value = mock_config

        result = BaseCommand.load_typed_config(str(config_file), required=True)

        assert result == mock_config
        mock_config_class.from_file.assert_called_once()

    def test_load_typed_config_file_not_found(self):
        """Test loading config from nonexistent file."""
        result = BaseCommand.load_typed_config('/nonexistent/config.yaml', required=True)

        assert result is None

    def test_load_typed_config_not_required_and_not_found(self):
        """Test loading non-required config from nonexistent file."""
        result = BaseCommand.load_typed_config('/nonexistent/config.yaml', required=False)

        assert result is None

    @patch('symfluence.core.config.models.SymfluenceConfig')
    def test_load_typed_config_configuration_error(self, mock_config_class, tmp_path):
        """Test loading config with configuration error."""
        from symfluence.core.exceptions import ConfigurationError

        config_file = tmp_path / 'config.yaml'
        config_file.write_text("DOMAIN_NAME: test")

        mock_config_class.from_file.side_effect = ConfigurationError("Invalid config")

        result = BaseCommand.load_typed_config(str(config_file), required=True)

        assert result is None


class TestSetConsole:
    """Test set_console method."""

    def test_set_console(self):
        """Test setting a custom console."""
        mock_console = MagicMock()
        original_console = BaseCommand._console

        try:
            BaseCommand.set_console(mock_console)
            assert BaseCommand._console == mock_console
        finally:
            # Restore original console
            BaseCommand._console = original_console


class TestDeprecatedMethods:
    """Test deprecated backward compatibility methods."""

    def test_print_error(self):
        """Test deprecated print_error method."""
        with patch.object(BaseCommand._console, 'error') as mock_error:
            BaseCommand.print_error("Test error")
            mock_error.assert_called_once_with("Test error")

    def test_print_success(self):
        """Test deprecated print_success method."""
        with patch.object(BaseCommand._console, 'success') as mock_success:
            BaseCommand.print_success("Test success")
            mock_success.assert_called_once_with("Test success")

    def test_print_info(self):
        """Test deprecated print_info method."""
        with patch.object(BaseCommand._console, 'info') as mock_info:
            BaseCommand.print_info("Test info")
            mock_info.assert_called_once_with("Test info")
