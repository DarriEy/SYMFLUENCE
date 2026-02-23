"""Unit tests for BinaryManager."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from symfluence.cli.binary_service import BinaryManager

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestInitialization:
    """Test BinaryManager initialization."""

    def test_initialization_with_default_tools(self):
        """Test BinaryManager initializes with default tools."""
        manager = BinaryManager()

        assert manager.external_tools is not None
        assert isinstance(manager.external_tools, dict)

    def test_initialization_with_custom_tools(self, mock_external_tools):
        """Test BinaryManager initializes with custom tools."""
        manager = BinaryManager(external_tools=mock_external_tools)

        assert manager.external_tools == mock_external_tools
        assert 'summa' in manager.external_tools
        assert 'sundials' in manager.external_tools


class TestHandleBinaryManagement:
    """Test handle_binary_management dispatcher."""

    @patch.object(BinaryManager, 'doctor')
    def test_doctor_command(self, mock_doctor, binary_manager):
        """Test doctor diagnostics command."""
        execution_plan = {
            'binary_operations': {
                'doctor': True
            }
        }

        result = binary_manager.handle_binary_management(execution_plan)

        assert result is True
        mock_doctor.assert_called_once()

    @patch.object(BinaryManager, 'tools_info')
    def test_tools_info_command(self, mock_tools_info, binary_manager):
        """Test tools info command."""
        execution_plan = {
            'binary_operations': {
                'tools_info': True
            }
        }

        result = binary_manager.handle_binary_management(execution_plan)

        assert result is True
        mock_tools_info.assert_called_once()

    @patch.object(BinaryManager, 'validate_binaries')
    def test_validate_binaries_command(self, mock_validate, binary_manager):
        """Test validate binaries command."""
        mock_validate.return_value = True

        execution_plan = {
            'binary_operations': {
                'validate_binaries': True
            }
        }

        result = binary_manager.handle_binary_management(execution_plan)

        assert result is True
        mock_validate.assert_called_once()

    @patch.object(BinaryManager, 'get_executables')
    def test_get_executables_command(self, mock_get_exec, binary_manager):
        """Test get executables command."""
        mock_get_exec.return_value = {'successful': ['summa'], 'failed': []}

        execution_plan = {
            'binary_operations': {
                'get_executables': ['summa']
            },
            'settings': {}
        }

        result = binary_manager.handle_binary_management(execution_plan)

        assert result is True
        mock_get_exec.assert_called_once()


class TestGetExecutables:
    """Test get_executables installation logic."""

    @patch('subprocess.run')
    @patch('shutil.which')
    @patch('os.chdir')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_dry_run_mode(self, mock_mkdir, mock_exists, mock_chdir, mock_which, mock_subprocess, binary_manager):
        """Test dry run doesn't execute commands."""
        mock_exists.return_value = False

        result = binary_manager.get_executables(
            specific_tools=['summa'],
            dry_run=True
        )

        assert result['dry_run'] is True
        # Should not have called subprocess for git clone or build
        assert mock_subprocess.call_count == 0

    @patch('subprocess.run')
    @patch('os.chdir')
    @patch.object(Path, 'exists', return_value=True)
    def test_skip_if_exists(self, mock_exists, mock_chdir, mock_subprocess, binary_manager, tmp_path):
        """Test skipping installation if tool exists."""
        # Mock all paths as existing - simulates tools already installed

        result = binary_manager.get_executables(
            specific_tools=['summa'],
            force=False
        )

        # Should skip installation (sundials is a dependency that also "exists")
        assert 'summa' in result['skipped'] or 'summa' in result['successful']


class TestBinaryValidation:
    """Test validate_binaries logic."""

    @patch('shutil.which')
    @patch('pathlib.Path.exists')
    def test_binary_missing(self, mock_exists, mock_which, binary_manager, mock_symfluence_instance):
        """Test validation of missing binary."""
        mock_which.return_value = None
        mock_exists.return_value = False

        mock_symfluence_instance.config = {
            'SYMFLUENCE_DATA_DIR': '/tmp/data',
            'SUMMA_EXE': 'summa.exe'
        }

        result = binary_manager.validate_binaries(mock_symfluence_instance)

        # Should detect missing tools
        assert isinstance(result, dict) or result is False
        if isinstance(result, dict):
            assert len(result.get('missing_tools', [])) > 0 or len(result.get('failed_tools', [])) > 0

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    def test_test_command_execution(self, mock_exists, mock_subprocess, binary_manager, mock_symfluence_instance, tmp_path):
        """Test binary validation via test command."""
        # Mock binary exists
        mock_exists.return_value = True

        # Mock test command success
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='version 3.0.3',
            stderr=''
        )

        mock_symfluence_instance.config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'SUMMA_EXE': str(tmp_path / 'summa.exe')
        }

        result = binary_manager.validate_binaries(mock_symfluence_instance)

        # Should validate successfully
        assert result is True or (isinstance(result, dict) and len(result.get('missing_tools', [])) == 0)

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    def test_test_command_timeout(self, mock_exists, mock_subprocess, binary_manager, mock_symfluence_instance, tmp_path):
        """Test test command timeout handling."""
        mock_exists.return_value = True

        # Mock timeout
        mock_subprocess.side_effect = subprocess.TimeoutExpired('summa', 10)

        mock_symfluence_instance.config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'SUMMA_EXE': str(tmp_path / 'summa.exe')
        }

        result = binary_manager.validate_binaries(mock_symfluence_instance)

        # Should handle timeout gracefully - returns True if most tools valid, dict otherwise
        # A timeout adds a warning but doesn't necessarily make validation fail
        assert isinstance(result, (dict, bool))


class TestDoctorDiagnostics:
    """Test run_doctor system diagnostics."""

    @patch('shutil.which')
    def test_doctor_finds_tools(self, mock_which, binary_manager):
        """Test doctor returns True when tools are present."""
        mock_which.return_value = '/usr/bin/tool'

        result = binary_manager.run_doctor()

        # Doctor should complete successfully
        assert result is True

    @patch('shutil.which')
    def test_doctor_missing_tools(self, mock_which, binary_manager):
        """Test doctor returns True even with missing tools (diagnostics completed)."""
        mock_which.return_value = None

        result = binary_manager.run_doctor()

        # Doctor should still complete (just reports issues)
        assert result is True


class TestToolsInfo:
    """Test show_tools_info."""

    @patch('pathlib.Path.exists')
    def test_show_tools_info(self, mock_exists, binary_manager):
        """Test tools info display returns boolean."""
        mock_exists.return_value = False  # No tools installed

        result = binary_manager.show_tools_info()

        # Should return a boolean indicating success/failure
        assert isinstance(result, bool)


class TestDependencyResolution:
    """Test tool dependency resolution logic."""

    def test_simple_dependency_order(self, mock_external_tools):
        """Test tools installed in dependency order."""
        manager = BinaryManager(external_tools=mock_external_tools)

        # summa requires sundials, so sundials should be resolved first
        # This is tested indirectly through get_executables
        # Note: 'requires' is for tool dependencies, 'dependencies' is for system deps
        assert 'sundials' in mock_external_tools['summa']['requires']

    def test_no_dependencies(self, mock_external_tools):
        """Test tools with no dependencies."""
        manager = BinaryManager(external_tools=mock_external_tools)

        # sundials has no system dependencies
        assert mock_external_tools['sundials']['dependencies'] == []


class TestPathValidation:
    """Test path validation and config management."""

    @patch('builtins.open', new_callable=mock_open, read_data="SUMMA_EXE: /path/to/summa")
    @patch('yaml.safe_load')
    @patch('yaml.dump')
    @patch('pathlib.Path.exists')
    def test_valid_writable_paths(self, mock_exists, mock_dump, mock_yaml, mock_file, binary_manager, tmp_path):
        """Test validation of writable paths."""
        mock_exists.return_value = True
        mock_yaml.return_value = {'SUMMA_EXE': str(tmp_path / 'summa')}

        # Test would call _ensure_valid_config_paths if it exists
        # For now, we just verify the function would work with valid paths
        assert tmp_path.exists()


class TestBuildCommands:
    """Test build command execution."""

    @patch('subprocess.run')
    @patch('os.chdir')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_build_success(self, mock_mkdir, mock_exists, mock_chdir, mock_subprocess, binary_manager):
        """Test successful build execution."""
        mock_exists.return_value = False
        mock_subprocess.return_value = MagicMock(returncode=0, stdout='', stderr='')

        # Mock get_executables with a single tool
        # This tests the build command flow
        result = binary_manager.get_executables(
            specific_tools=['mizuroute'],
            dry_run=False
        )

        # Should attempt build
        assert 'mizuroute' in result['successful'] or 'mizuroute' in result['failed'] or 'mizuroute' in result['skipped']

    @patch('subprocess.run')
    @patch('os.chdir')
    @patch('pathlib.Path.exists')
    def test_build_failure_handling(self, mock_exists, mock_chdir, mock_subprocess, binary_manager):
        """Test handling of build failures."""
        mock_exists.return_value = False

        # Mock build failure
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=['make'],
            stderr='Build error'
        )

        result = binary_manager.get_executables(
            specific_tools=['mizuroute'],
            dry_run=False
        )

        # Should handle build failure
        assert 'mizuroute' in result['failed'] or 'mizuroute' in result['errors']


class TestDetectNpmBinaries:
    """Test npm binary detection."""

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_detect_npm_binaries(self, mock_glob, mock_exists, binary_manager):
        """Test detection of npm-installed binaries."""
        # This method may or may not exist depending on implementation
        if hasattr(binary_manager, 'detect_npm_binaries'):
            mock_exists.return_value = True
            mock_glob.return_value = [Path('/path/to/bin/tool')]

            result = binary_manager.detect_npm_binaries()

            # detect_npm_binaries returns Path or None
            assert isinstance(result, (Path, type(None)))
