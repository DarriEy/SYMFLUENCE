"""Unit tests for binary command handlers."""

import pytest
import subprocess
from argparse import Namespace
from unittest.mock import MagicMock, patch

from symfluence.cli.commands.binary_commands import BinaryCommands
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestBinaryInstall:
    """Test binary install command."""

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_all_success(self, mock_binary_manager_class):
        """Test successful installation of all tools."""
        mock_manager = MagicMock()
        mock_manager.get_executables.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=None,  # Install all
            force=False,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.SUCCESS
        mock_manager.get_executables.assert_called_once_with(specific_tools=None, force=False)

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_specific_tools(self, mock_binary_manager_class):
        """Test installation of specific tools."""
        mock_manager = MagicMock()
        mock_manager.get_executables.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=['summa', 'mizuroute'],
            force=False,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.SUCCESS
        mock_manager.get_executables.assert_called_once_with(
            specific_tools=['summa', 'mizuroute'],
            force=False
        )

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_force_reinstall(self, mock_binary_manager_class):
        """Test force reinstall of tools."""
        mock_manager = MagicMock()
        mock_manager.get_executables.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=['taudem'],
            force=True,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.SUCCESS
        mock_manager.get_executables.assert_called_once_with(specific_tools=['taudem'], force=True)

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_failure(self, mock_binary_manager_class):
        """Test installation failure."""
        mock_manager = MagicMock()
        mock_manager.get_executables.return_value = False
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=['summa'],
            force=False,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.BINARY_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_build_error(self, mock_binary_manager_class):
        """Test installation with build error."""
        mock_manager = MagicMock()
        mock_manager.get_executables.side_effect = subprocess.CalledProcessError(1, 'make')
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=['summa'],
            force=False,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.BINARY_BUILD_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_install_permission_error(self, mock_binary_manager_class):
        """Test installation with permission error."""
        mock_manager = MagicMock()
        mock_manager.get_executables.side_effect = PermissionError("Cannot write to install dir")
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            tools=['summa'],
            force=False,
            debug=False
        )

        result = BinaryCommands.install(args)

        assert result == ExitCode.PERMISSION_ERROR


class TestBinaryValidate:
    """Test binary validate command."""

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_validate_success(self, mock_binary_manager_class):
        """Test successful binary validation."""
        mock_manager = MagicMock()
        mock_manager.validate_binaries.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            verbose=False,
            debug=False
        )

        result = BinaryCommands.validate(args)

        assert result == ExitCode.SUCCESS
        mock_manager.validate_binaries.assert_called_once_with(verbose=False)

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_validate_failure(self, mock_binary_manager_class):
        """Test binary validation failure."""
        mock_manager = MagicMock()
        mock_manager.validate_binaries.return_value = False
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            verbose=True,
            debug=False
        )

        result = BinaryCommands.validate(args)

        assert result == ExitCode.BINARY_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_validate_binary_not_found(self, mock_binary_manager_class):
        """Test validation when binary not found."""
        mock_manager = MagicMock()
        mock_manager.validate_binaries.side_effect = FileNotFoundError("summa.exe not found")
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(
            verbose=False,
            debug=False
        )

        result = BinaryCommands.validate(args)

        assert result == ExitCode.FILE_NOT_FOUND


class TestBinaryDoctor:
    """Test binary doctor command."""

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_doctor_success(self, mock_binary_manager_class):
        """Test successful diagnostics."""
        mock_manager = MagicMock()
        mock_manager.doctor.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = BinaryCommands.doctor(args)

        assert result == ExitCode.SUCCESS
        mock_manager.doctor.assert_called_once()

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_doctor_issues_found(self, mock_binary_manager_class):
        """Test diagnostics with issues found."""
        mock_manager = MagicMock()
        mock_manager.doctor.return_value = False
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = BinaryCommands.doctor(args)

        assert result == ExitCode.DEPENDENCY_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_doctor_import_error(self, mock_binary_manager_class):
        """Test diagnostics with import error."""
        mock_binary_manager_class.side_effect = ImportError("Cannot import BinaryManager")

        args = Namespace(debug=False)

        result = BinaryCommands.doctor(args)

        assert result == ExitCode.DEPENDENCY_ERROR


class TestBinaryInfo:
    """Test binary info command."""

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_info_success(self, mock_binary_manager_class):
        """Test successful tools info display."""
        mock_manager = MagicMock()
        mock_manager.tools_info.return_value = True
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = BinaryCommands.info(args)

        assert result == ExitCode.SUCCESS
        mock_manager.tools_info.assert_called_once()

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_info_failure(self, mock_binary_manager_class):
        """Test tools info failure."""
        mock_manager = MagicMock()
        mock_manager.tools_info.return_value = False
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = BinaryCommands.info(args)

        assert result == ExitCode.GENERAL_ERROR

    @patch('symfluence.cli.binary_service.BinaryManager')
    def test_info_directory_not_found(self, mock_binary_manager_class):
        """Test info when tools directory not found."""
        mock_manager = MagicMock()
        mock_manager.tools_info.side_effect = FileNotFoundError("Tools dir not found")
        mock_binary_manager_class.return_value = mock_manager

        args = Namespace(debug=False)

        result = BinaryCommands.info(args)

        assert result == ExitCode.FILE_NOT_FOUND
