"""Unit tests for project command handlers."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from symfluence.cli.commands.project_commands import ProjectCommands
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestProjectInit:
    """Test project init command."""

    @patch('symfluence.cli.services.InitializationManager')
    def test_init_success(self, mock_init_manager_class, tmp_path):
        """Test successful project initialization."""
        mock_manager = MagicMock()
        mock_manager.generate_config.return_value = {'DOMAIN_NAME': 'test_domain'}
        mock_manager.write_config.return_value = tmp_path / 'config_test_domain.yaml'
        mock_init_manager_class.return_value = mock_manager

        args = Namespace(
            preset=None,
            domain='test_domain',
            model='SUMMA',
            start_date='2020-01-01',
            end_date='2020-12-31',
            forcing='ERA5',
            discretization='lumped',
            definition_method='subset',
            output_dir=str(tmp_path),
            scaffold=False,
            minimal=False,
            comprehensive=True,
            interactive=False,
            debug=False
        )

        result = ProjectCommands.init(args)

        assert result == ExitCode.SUCCESS
        mock_manager.generate_config.assert_called_once()
        mock_manager.write_config.assert_called_once()

    @patch('symfluence.cli.services.InitializationManager')
    def test_init_with_scaffold(self, mock_init_manager_class, tmp_path):
        """Test project initialization with scaffold creation."""
        mock_manager = MagicMock()
        mock_manager.generate_config.return_value = {'DOMAIN_NAME': 'test_domain'}
        mock_manager.write_config.return_value = tmp_path / 'config_test_domain.yaml'
        mock_manager.create_scaffold.return_value = tmp_path / 'test_domain'
        mock_init_manager_class.return_value = mock_manager

        args = Namespace(
            preset=None,
            domain='test_domain',
            model='SUMMA',
            start_date='2020-01-01',
            end_date='2020-12-31',
            forcing='ERA5',
            discretization='lumped',
            definition_method='subset',
            output_dir=str(tmp_path),
            scaffold=True,
            minimal=False,
            comprehensive=True,
            interactive=False,
            debug=False
        )

        result = ProjectCommands.init(args)

        assert result == ExitCode.SUCCESS
        mock_manager.create_scaffold.assert_called_once()

    @patch('symfluence.cli.services.InitializationManager')
    def test_init_value_error(self, mock_init_manager_class):
        """Test init with invalid value."""
        mock_manager = MagicMock()
        mock_manager.generate_config.side_effect = ValueError("Invalid domain name")
        mock_init_manager_class.return_value = mock_manager

        args = Namespace(
            preset=None,
            domain='',  # Invalid
            model='SUMMA',
            start_date='2020-01-01',
            end_date='2020-12-31',
            forcing='ERA5',
            discretization='lumped',
            definition_method='subset',
            output_dir='./',
            scaffold=False,
            minimal=False,
            comprehensive=True,
            interactive=False,
            debug=False
        )

        result = ProjectCommands.init(args)

        # ValueError is caught by the decorator and returns VALIDATION_ERROR
        assert result == ExitCode.VALIDATION_ERROR

    @patch('symfluence.cli.wizard.ProjectWizard')
    def test_init_interactive_mode(self, mock_wizard_class, tmp_path):
        """Test project initialization in interactive mode."""
        mock_wizard = MagicMock()
        mock_wizard.run.return_value = ExitCode.SUCCESS
        mock_wizard_class.return_value = mock_wizard

        args = Namespace(
            interactive=True,
            output_dir=str(tmp_path),
            scaffold=False,
            debug=False
        )

        result = ProjectCommands.init(args)

        assert result == ExitCode.SUCCESS
        mock_wizard.run.assert_called_once()


class TestProjectPourPoint:
    """Test project pour-point command."""

    @patch('symfluence.project.pour_point_workflow.setup_pour_point_workflow')
    def test_pour_point_success(self, mock_setup, tmp_path):
        """Test successful pour point setup."""
        mock_result = MagicMock()
        mock_result.config_file = tmp_path / 'config.yaml'
        mock_result.used_auto_bounding_box = False
        mock_setup.return_value = mock_result

        args = Namespace(
            coordinates='51.1722/-115.5717',
            domain_def='subset',
            domain_name='test_watershed',
            bounding_box_coords=None,
            output_dir=str(tmp_path),
            debug=False
        )

        result = ProjectCommands.pour_point(args)

        assert result == ExitCode.SUCCESS
        mock_setup.assert_called_once()

    def test_pour_point_invalid_coordinates(self, tmp_path):
        """Test pour point with invalid coordinates."""
        args = Namespace(
            coordinates='91/0',  # Latitude out of range
            domain_def='subset',
            domain_name='test_watershed',
            bounding_box_coords=None,
            output_dir=str(tmp_path),
            debug=False
        )

        result = ProjectCommands.pour_point(args)

        assert result == ExitCode.VALIDATION_ERROR

    def test_pour_point_invalid_bounding_box(self, tmp_path):
        """Test pour point with invalid bounding box."""
        args = Namespace(
            coordinates='51.1722/-115.5717',
            domain_def='subset',
            domain_name='test_watershed',
            bounding_box_coords='40/10/50/20',  # lat_min > lat_max
            output_dir=str(tmp_path),
            debug=False
        )

        result = ProjectCommands.pour_point(args)

        assert result == ExitCode.VALIDATION_ERROR

    @patch('symfluence.project.pour_point_workflow.setup_pour_point_workflow')
    def test_pour_point_file_not_found(self, mock_setup, tmp_path):
        """Test pour point with file not found error."""
        mock_setup.side_effect = FileNotFoundError("Shapefile not found")

        args = Namespace(
            coordinates='51.1722/-115.5717',
            domain_def='subset',
            domain_name='test_watershed',
            bounding_box_coords=None,
            output_dir=str(tmp_path),
            debug=False
        )

        result = ProjectCommands.pour_point(args)

        assert result == ExitCode.FILE_NOT_FOUND


class TestProjectListPresets:
    """Test project list-presets command."""

    @patch('symfluence.cli.services.InitializationManager')
    def test_list_presets_success(self, mock_init_manager_class):
        """Test successful preset listing."""
        mock_manager = MagicMock()
        mock_init_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = ProjectCommands.list_presets(args)

        assert result == ExitCode.SUCCESS
        mock_manager.list_presets.assert_called_once()

    @patch('symfluence.cli.services.InitializationManager')
    def test_list_presets_file_not_found(self, mock_init_manager_class):
        """Test preset listing when preset files not found."""
        mock_manager = MagicMock()
        mock_manager.list_presets.side_effect = FileNotFoundError("Presets dir not found")
        mock_init_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = ProjectCommands.list_presets(args)

        assert result == ExitCode.FILE_NOT_FOUND


class TestProjectShowPreset:
    """Test project show-preset command."""

    @patch('symfluence.cli.services.InitializationManager')
    def test_show_preset_success(self, mock_init_manager_class):
        """Test successful preset display."""
        mock_manager = MagicMock()
        mock_manager.show_preset.return_value = {'DOMAIN_NAME': 'example'}
        mock_init_manager_class.return_value = mock_manager

        args = Namespace(
            preset_name='quickstart',
            debug=False
        )

        result = ProjectCommands.show_preset(args)

        assert result == ExitCode.SUCCESS
        mock_manager.show_preset.assert_called_once_with('quickstart')

    @patch('symfluence.cli.services.InitializationManager')
    def test_show_preset_not_found(self, mock_init_manager_class):
        """Test preset display when preset not found."""
        mock_manager = MagicMock()
        mock_manager.show_preset.return_value = None
        mock_init_manager_class.return_value = mock_manager

        args = Namespace(
            preset_name='nonexistent',
            debug=False
        )

        result = ProjectCommands.show_preset(args)

        assert result == ExitCode.FILE_NOT_FOUND
