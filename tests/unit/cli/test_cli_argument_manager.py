"""Unit tests for CLIArgumentManager."""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import argparse

from symfluence.utils.cli.cli_argument_manager import CLIArgumentManager

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestInitialization:
    """Test CLIArgumentManager initialization."""

    def test_initialization_creates_submanagers(self):
        """Test that initialization creates all required sub-managers."""
        manager = CLIArgumentManager()

        assert manager.binary_manager is not None
        assert manager.job_scheduler is not None
        assert manager.notebook_service is not None
        assert manager.parser is not None
        assert isinstance(manager.workflow_steps, dict)
        assert len(manager.workflow_steps) > 0

    def test_workflow_steps_defined(self):
        """Test that workflow steps are properly defined."""
        manager = CLIArgumentManager()

        # Check some expected steps exist
        expected_steps = ['setup_project', 'define_domain', 'discretize_domain',
                         'acquire_forcings', 'run_model', 'postprocess_results']

        for step in expected_steps:
            assert step in manager.workflow_steps
            assert 'description' in manager.workflow_steps[step]


class TestCoordinateValidation:
    """Test coordinate and bounding box validation logic."""

    @pytest.mark.parametrize("coords,expected", [
        ("51.1722/115.5717", True),
        ("51.1722/-115.5717", True),
        ("-51.1722/115.5717", True),
        ("-51.1722/-115.5717", True),
        ("0.0/0.0", True),
        ("90.0/180.0", True),
        ("-90.0/-180.0", True),
    ])
    def test_valid_coordinates(self, cli_manager, coords, expected):
        """Test parsing of valid coordinates."""
        result = cli_manager._validate_coordinates(coords)
        assert result == expected

    @pytest.mark.parametrize("coords", [
        "51.1722",              # Missing longitude
        "51.1722/115.5717/10",  # Too many values
        "abc/def",              # Non-numeric
        "91.0/0.0",             # Latitude out of range
        "0.0/181.0",            # Longitude out of range
        "-91.0/0.0",            # Latitude out of range (negative)
        "0.0/-181.0",           # Longitude out of range (negative)
        "",                     # Empty string
        "/",                    # Just separator
        "51.1722/",             # Missing longitude
        "/115.5717",            # Missing latitude
    ])
    def test_invalid_coordinates(self, cli_manager, coords):
        """Test rejection of invalid coordinates."""
        result = cli_manager._validate_coordinates(coords)
        assert result is False

    @pytest.mark.parametrize("bbox,expected", [
        ("52.0/-116.0/51.0/-115.0", True),  # lat_max/lon_min/lat_min/lon_max
        ("10.0/-10.0/-10.0/10.0", True),
        ("90.0/-180.0/-90.0/180.0", True),   # Max ranges
        ("0.0/0.0/0.0/0.0", True),           # Point bbox (degenerate but valid)
    ])
    def test_valid_bounding_box(self, cli_manager, bbox, expected):
        """Test valid bounding box validation."""
        result = cli_manager._validate_bounding_box(bbox)
        assert result == expected

    @pytest.mark.parametrize("bbox", [
        "50.0/-116.0/51.0/-115.0",  # lat_min > lat_max (swapped)
        "52.0/-115.0/51.0/-116.0",  # lon_min > lon_max (swapped)
        "91.0/-116.0/51.0/-115.0",  # Lat out of range
        "52.0/181.0/51.0/-115.0",   # Lon out of range
        "52.0/-116.0/51.0",         # Too few values
        "52.0/-116.0/51.0/-115.0/10.0",  # Too many values
        "abc/def/ghi/jkl",          # Non-numeric
        "",                         # Empty string
    ])
    def test_invalid_bounding_box(self, cli_manager, bbox):
        """Test rejection of invalid bounding boxes."""
        result = cli_manager._validate_bounding_box(bbox)
        assert result is False


class TestArgumentParsing:
    """Test basic argument parsing."""

    def test_parse_help_argument(self):
        """Test --help argument parsing."""
        manager = CLIArgumentManager()

        # --help exits the program, so we catch SystemExit
        with pytest.raises(SystemExit) as exc_info:
            manager.parse_arguments(['--help'])

        # Exit code 0 indicates help was displayed successfully
        assert exc_info.value.code == 0

    def test_parse_version_argument(self):
        """Test --version argument parsing."""
        manager = CLIArgumentManager()

        with pytest.raises(SystemExit) as exc_info:
            manager.parse_arguments(['--version'])

        assert exc_info.value.code == 0

    def test_parse_config_path(self):
        """Test --config argument parsing."""
        manager = CLIArgumentManager()

        args = manager.parse_arguments(['--config', '/path/to/config.yaml'])

        assert args.config == '/path/to/config.yaml'

    def test_parse_pour_point_coordinates(self):
        """Test --pour_point argument parsing."""
        manager = CLIArgumentManager()

        args = manager.parse_arguments([
            '--pour_point', '51.0/115.0',
            '--domain_name', 'test_domain'
        ])

        assert args.pour_point == '51.0/115.0'
        assert args.domain_name == 'test_domain'

    def test_parse_multiple_workflow_steps(self):
        """Test parsing multiple workflow step flags."""
        manager = CLIArgumentManager()

        args = manager.parse_arguments([
            '--config', 'test.yaml',
            '--define_domain',
            '--discretize_domain'
        ])

        assert args.define_domain is True
        assert args.discretize_domain is True

    def test_parse_force_rerun_flag(self):
        """Test --force_rerun flag parsing."""
        manager = CLIArgumentManager()

        args = manager.parse_arguments(['--config', 'test.yaml', '--force_rerun'])

        assert args.force_rerun is True

    def test_parse_debug_flag(self):
        """Test --debug flag parsing."""
        manager = CLIArgumentManager()

        args = manager.parse_arguments(['--config', 'test.yaml', '--debug'])

        assert args.debug is True


class TestArgumentValidation:
    """Test validate_arguments logic."""

    def test_conflicting_error_handling_flags(self, cli_manager):
        """Test that --stop_on_error and --continue_on_error conflict."""
        # Create args with both flags
        args = argparse.Namespace(
            stop_on_error=True,
            continue_on_error=True,
            config='test.yaml',
            pour_point=None,
            domain_def=None,
            resume_from=None,
            get_executables=None,
            validate_binaries=False,
            doctor=False,
            tools_info=False,
            list_templates=False,
            validate_environment=False,
            update_config=None,
            list_steps=False,
            validate_config=False,
            domain_name=None
        )

        # Both flags set should cause validation error
        valid, errors = cli_manager.validate_arguments(args)
        assert valid is False
        assert any('stop_on_error' in str(err).lower() and 'continue_on_error' in str(err).lower() for err in errors)

    def test_pour_point_requires_domain_name(self, cli_manager):
        """Test that --pour_point requires --domain_name and --domain_def."""
        args = argparse.Namespace(
            pour_point='51.0/115.0',
            domain_name=None,
            domain_def=None,
            config='test.yaml',
            resume_from=None,
            get_executables=None,
            validate_binaries=False,
            doctor=False,
            tools_info=False,
            list_templates=False,
            validate_environment=False,
            update_config=None,
            list_steps=False,
            validate_config=False,
            stop_on_error=False,
            continue_on_error=False
        )

        valid, errors = cli_manager.validate_arguments(args)
        assert valid is False
        assert any('domain_name' in str(err).lower() or 'domain_def' in str(err).lower() for err in errors)

    def test_invalid_pour_point_coordinates(self, cli_manager):
        """Test validation rejects invalid pour point coordinates."""
        args = argparse.Namespace(
            pour_point='invalid/coords',
            domain_name='test_domain',
            domain_def='delineate',
            config='test.yaml',
            resume_from=None,
            get_executables=None,
            validate_binaries=False,
            doctor=False,
            tools_info=False,
            list_templates=False,
            validate_environment=False,
            update_config=None,
            list_steps=False,
            validate_config=False,
            stop_on_error=False,
            continue_on_error=False
        )

        valid, errors = cli_manager.validate_arguments(args)
        assert valid is False
        assert any('coordinate' in str(err).lower() or 'invalid' in str(err).lower() or 'pour' in str(err).lower() for err in errors)

    def test_invalid_bounding_box(self, cli_manager):
        """Test validation rejects invalid bounding box."""
        args = argparse.Namespace(
            pour_point='51.0/115.0',
            bounding_box_coords='invalid/bbox/format',
            domain_name='test_domain',
            domain_def='delineate',
            config='test.yaml',
            resume_from=None,
            get_executables=None,
            validate_binaries=False,
            doctor=False,
            tools_info=False,
            list_templates=False,
            validate_environment=False,
            update_config=None,
            list_steps=False,
            validate_config=False,
            stop_on_error=False,
            continue_on_error=False
        )

        valid, errors = cli_manager.validate_arguments(args)
        assert valid is False
        assert any('bounding' in str(err).lower() or 'bbox' in str(err).lower() for err in errors)


class TestExecutionPlanGeneration:
    """Test get_execution_plan for different modes."""

    def test_execution_plan_has_settings(self):
        """Test that execution plan includes settings."""
        # Use actual parser to get properly formed args
        manager = CLIArgumentManager()
        args = manager.parse_arguments(['--config', 'test.yaml', '--force_rerun', '--debug'])

        plan = manager.get_execution_plan(args)

        assert 'settings' in plan
        assert plan['settings']['force_rerun'] is True
        assert plan['settings']['debug'] is True


class TestConfigOverrides:
    """Test apply_config_overrides."""

    def test_simple_override(self, cli_manager):
        """Test overriding a single config value."""
        config = {"DOMAIN_NAME": "old_value", "MODEL": "SUMMA"}
        overrides = {"DOMAIN_NAME": "new_value"}

        result = cli_manager.apply_config_overrides(config, overrides)

        assert result["DOMAIN_NAME"] == "new_value"
        assert result["MODEL"] == "SUMMA"  # Unchanged

    def test_multiple_overrides(self, cli_manager):
        """Test applying multiple overrides."""
        config = {
            "DOMAIN_NAME": "old_domain",
            "MODEL": "SUMMA",
            "START_DATETIME": "2020-01-01"
        }
        overrides = {
            "DOMAIN_NAME": "new_domain",
            "MODEL": "FUSE",
        }

        result = cli_manager.apply_config_overrides(config, overrides)

        assert result["DOMAIN_NAME"] == "new_domain"
        assert result["MODEL"] == "FUSE"
        assert result["START_DATETIME"] == "2020-01-01"  # Unchanged

    def test_new_key_added(self, cli_manager):
        """Test that new keys are added to config."""
        config = {"DOMAIN_NAME": "test"}
        overrides = {"NEW_KEY": "new_value"}

        result = cli_manager.apply_config_overrides(config, overrides)

        assert "NEW_KEY" in result
        assert result["NEW_KEY"] == "new_value"

    def test_empty_overrides(self, cli_manager):
        """Test that empty overrides don't change config."""
        config = {"DOMAIN_NAME": "test", "MODEL": "SUMMA"}
        overrides = {}

        result = cli_manager.apply_config_overrides(config, overrides)

        assert result == config


class TestPourPointSetup:
    """Test pour point workflow setup."""

    @patch('pathlib.Path.exists')
    def test_missing_template_error(self, mock_exists, cli_manager):
        """Test error when template not found."""
        # Mock all template paths as non-existent
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            cli_manager.setup_pour_point_workflow(
                coordinates='51.0/115.0',
                domain_def_method='delineate',
                domain_name='test_domain',
                bounding_box_coords=None
            )
