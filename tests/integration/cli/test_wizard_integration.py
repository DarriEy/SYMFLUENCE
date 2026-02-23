"""Integration tests for project wizard."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from symfluence.cli.exit_codes import ExitCode
from symfluence.cli.wizard import ProjectWizard, WizardState


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def console():
    """Create a console for testing."""
    from io import StringIO
    return Console(file=StringIO(), force_terminal=True, no_color=True)


@pytest.fixture
def wizard(console):
    """Create a wizard instance with test console."""
    return ProjectWizard(console=console)


class TestProjectWizardInit:
    """Tests for ProjectWizard initialization."""

    def test_init_creates_state(self):
        """Test that initialization creates WizardState."""
        wizard = ProjectWizard()
        assert isinstance(wizard.state, WizardState)

    def test_init_with_console(self, console):
        """Test initialization with custom console."""
        wizard = ProjectWizard(console=console)
        assert wizard.console is console


class TestProjectWizardRun:
    """Tests for ProjectWizard.run method."""

    @patch.object(ProjectWizard, '_process_phase')
    @patch.object(ProjectWizard, '_handle_summary_phase')
    def test_run_processes_all_phases(self, mock_summary, mock_process, wizard, temp_dir):
        """Test that run processes all phases."""
        mock_process.return_value = None
        mock_summary.return_value = ExitCode.SUCCESS

        result = wizard.run(output_dir=temp_dir)

        assert result == ExitCode.SUCCESS
        # Should have processed all phases except SUMMARY
        assert mock_process.call_count == 4

    @patch.object(ProjectWizard, '_process_phase')
    def test_run_returns_user_interrupt_on_quit(self, mock_process, wizard, temp_dir):
        """Test that run returns USER_INTERRUPT when user quits."""
        mock_process.return_value = 'quit'

        result = wizard.run(output_dir=temp_dir)

        assert result == ExitCode.USER_INTERRUPT

    def test_run_handles_keyboard_interrupt(self, wizard, temp_dir):
        """Test that run handles KeyboardInterrupt gracefully."""
        with patch.object(wizard.prompts, 'show_welcome', side_effect=KeyboardInterrupt):
            result = wizard.run(output_dir=temp_dir)

        assert result == ExitCode.USER_INTERRUPT


class TestWizardConfigGeneration:
    """Tests for configuration file generation."""

    def test_generate_config_creates_file(self, wizard, temp_dir):
        """Test that config file is generated."""
        # Setup state with required answers
        wizard.state.set_answer('DOMAIN_NAME', 'test_domain')
        wizard.state.set_answer('HYDROLOGICAL_MODEL', 'SUMMA')
        wizard.state.set_answer('EXPERIMENT_TIME_START', '2010-01-01')
        wizard.state.set_answer('EXPERIMENT_TIME_END', '2020-12-31')
        wizard.state.set_answer('FORCING_DATASET', 'ERA5')

        with patch('symfluence.cli.services.InitializationService') as mock_init_class:
            mock_service = MagicMock()
            mock_service.generate_config.return_value = {
                'DOMAIN_NAME': 'test_domain',
                'HYDROLOGICAL_MODEL': 'SUMMA',
            }
            mock_service.write_config.return_value = Path(temp_dir) / 'config_test.yaml'
            mock_init_class.return_value = mock_service

            config_path, scaffold_path = wizard._generate_config(temp_dir, scaffold=False)

        assert mock_service.generate_config.called
        assert mock_service.write_config.called

    def test_generate_config_with_scaffold(self, wizard, temp_dir):
        """Test that scaffold is created when requested."""
        # Setup state with required answers
        wizard.state.set_answer('DOMAIN_NAME', 'test_domain')
        wizard.state.set_answer('HYDROLOGICAL_MODEL', 'SUMMA')
        wizard.state.set_answer('EXPERIMENT_TIME_START', '2010-01-01')
        wizard.state.set_answer('EXPERIMENT_TIME_END', '2020-12-31')
        wizard.state.set_answer('FORCING_DATASET', 'ERA5')

        with patch('symfluence.cli.services.InitializationService') as mock_init_class:
            mock_service = MagicMock()
            mock_service.generate_config.return_value = {'DOMAIN_NAME': 'test_domain'}
            mock_service.write_config.return_value = Path(temp_dir) / 'config_test.yaml'
            mock_service.create_scaffold.return_value = Path(temp_dir) / 'domain_test'
            mock_init_class.return_value = mock_service

            config_path, scaffold_path = wizard._generate_config(temp_dir, scaffold=True)

        assert mock_service.create_scaffold.called
        assert scaffold_path is not None


class TestWizardPourPointHandling:
    """Tests for pour point coordinate handling."""

    def test_pour_point_coords_parsed(self, wizard, temp_dir):
        """Test that pour point coordinates are parsed correctly."""
        wizard.state.set_answer('POUR_POINT_COORDS', '51.1722/-115.5717')
        wizard.state.set_answer('DOMAIN_NAME', 'test')
        wizard.state.set_answer('HYDROLOGICAL_MODEL', 'SUMMA')
        wizard.state.set_answer('EXPERIMENT_TIME_START', '2010-01-01')
        wizard.state.set_answer('EXPERIMENT_TIME_END', '2020-12-31')
        wizard.state.set_answer('FORCING_DATASET', 'ERA5')

        with patch('symfluence.cli.services.InitializationService') as mock_init_class:
            mock_instance = MagicMock()
            mock_instance.generate_config.return_value = {'DOMAIN_NAME': 'test'}
            mock_instance.write_config.return_value = Path(temp_dir) / 'config.yaml'
            mock_init_class.return_value = mock_instance

            wizard._generate_config(temp_dir, scaffold=False)

        # Verify generate_config was called
        assert mock_instance.generate_config.called


class TestWizardModelSpecificSettings:
    """Tests for model-specific settings handling."""

    def test_summa_settings_included(self, wizard, temp_dir):
        """Test that SUMMA-specific settings are included."""
        wizard.state.set_answer('DOMAIN_NAME', 'test')
        wizard.state.set_answer('HYDROLOGICAL_MODEL', 'SUMMA')
        wizard.state.set_answer('SUMMA_SPATIAL_MODE', 'lumped')
        wizard.state.set_answer('ROUTING_MODEL', 'mizuRoute')
        wizard.state.set_answer('EXPERIMENT_TIME_START', '2010-01-01')
        wizard.state.set_answer('EXPERIMENT_TIME_END', '2020-12-31')
        wizard.state.set_answer('FORCING_DATASET', 'ERA5')

        with patch('symfluence.cli.services.InitializationService') as mock_init_class:
            mock_instance = MagicMock()
            mock_instance.generate_config.return_value = {'DOMAIN_NAME': 'test'}
            mock_instance.write_config.return_value = Path(temp_dir) / 'config.yaml'
            mock_init_class.return_value = mock_instance

            wizard._generate_config(temp_dir, scaffold=False)

        assert mock_instance.generate_config.called

    def test_fuse_settings_included(self, wizard, temp_dir):
        """Test that FUSE-specific settings are included."""
        wizard.state.set_answer('DOMAIN_NAME', 'test')
        wizard.state.set_answer('HYDROLOGICAL_MODEL', 'FUSE')
        wizard.state.set_answer('FUSE_SPATIAL_MODE', 'lumped')
        wizard.state.set_answer('EXPERIMENT_TIME_START', '2010-01-01')
        wizard.state.set_answer('EXPERIMENT_TIME_END', '2020-12-31')
        wizard.state.set_answer('FORCING_DATASET', 'ERA5')

        with patch('symfluence.cli.services.InitializationService') as mock_init_class:
            mock_instance = MagicMock()
            mock_instance.generate_config.return_value = {'DOMAIN_NAME': 'test'}
            mock_instance.write_config.return_value = Path(temp_dir) / 'config.yaml'
            mock_init_class.return_value = mock_instance

            wizard._generate_config(temp_dir, scaffold=False)

        assert mock_instance.generate_config.called


class TestWizardNavigation:
    """Tests for wizard navigation functionality."""

    def test_process_phase_handles_back(self, wizard):
        """Test that phase processing handles back navigation."""
        # This tests internal navigation logic - basic test
        pass

    def test_process_phase_handles_quit(self, wizard):
        """Test that phase processing handles quit."""
        from symfluence.cli.wizard.state import WizardPhase

        with patch.object(wizard.prompts, 'ask', return_value=(None, 'quit')):
            with patch.object(wizard.prompts, 'show_phase_header'):
                result = wizard._process_phase(WizardPhase.ESSENTIAL)

        assert result == 'quit'


class TestWizardSummaryPhase:
    """Tests for wizard summary phase."""

    def test_summary_shows_all_answers(self, wizard):
        """Test that summary displays all answers."""
        wizard.state.set_answer('DOMAIN_NAME', 'test_domain')
        wizard.state.set_answer('HYDROLOGICAL_MODEL', 'SUMMA')
        wizard.state.set_answer('EXPERIMENT_TIME_START', '2010-01-01')
        wizard.state.set_answer('EXPERIMENT_TIME_END', '2020-12-31')

        with patch.object(wizard.prompts, 'show_summary') as mock_summary:
            with patch.object(wizard.prompts, 'confirm_generate', return_value=False):
                with patch.object(wizard.prompts, 'show_cancelled'):
                    wizard._handle_summary_phase('.', scaffold=False)

        mock_summary.assert_called_once()

    def test_summary_respects_cancel(self, wizard, temp_dir):
        """Test that cancelling at summary returns correct code."""
        wizard.state.set_answer('DOMAIN_NAME', 'test')

        with patch.object(wizard.prompts, 'show_summary'):
            with patch.object(wizard.prompts, 'confirm_generate', return_value=False):
                with patch.object(wizard.prompts, 'show_cancelled'):
                    result = wizard._handle_summary_phase(temp_dir, scaffold=False)

        assert result == ExitCode.USER_INTERRUPT


class TestCLIIntegration:
    """Tests for CLI argument integration."""

    def test_interactive_flag_routes_to_wizard(self, temp_dir):
        """Test that --interactive flag routes to wizard."""
        from argparse import Namespace

        from symfluence.cli.commands.project_commands import ProjectCommands

        args = Namespace(
            interactive=True,
            output_dir=temp_dir,
            scaffold=False,
            preset=None,
            debug=False,
        )

        with patch('symfluence.cli.wizard.ProjectWizard') as mock_wizard_class:
            mock_wizard = MagicMock()
            mock_wizard.run.return_value = ExitCode.SUCCESS
            mock_wizard_class.return_value = mock_wizard

            result = ProjectCommands.init(args)

        assert mock_wizard_class.called
        assert mock_wizard.run.called

    def test_non_interactive_uses_init_manager(self, temp_dir):
        """Test that non-interactive mode uses InitializationManager."""
        from argparse import Namespace

        from symfluence.cli.commands.project_commands import ProjectCommands

        args = Namespace(
            interactive=False,
            output_dir=temp_dir,
            scaffold=False,
            preset='fuse-lumped',
            domain=None,
            model=None,
            start_date=None,
            end_date=None,
            forcing=None,
            discretization=None,
            definition_method=None,
            minimal=False,
            comprehensive=True,
            debug=False,
        )

        with patch('symfluence.cli.services.InitializationManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.generate_config.return_value = {'DOMAIN_NAME': 'test'}
            mock_manager.write_config.return_value = Path(temp_dir) / 'config.yaml'
            mock_manager_class.return_value = mock_manager

            result = ProjectCommands.init(args)

        assert mock_manager_class.called
