"""Unit tests for example command handlers."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.cli.commands.example_commands import ExampleCommands
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestExampleLaunch:
    """Test example launch command."""

    @patch('symfluence.cli.services.NotebookService')
    def test_launch_success(self, mock_notebook_service_class):
        """Test successful notebook launch."""
        mock_service = MagicMock()
        mock_service.launch_example_notebook.return_value = 0
        mock_notebook_service_class.return_value = mock_service

        args = Namespace(
            example_id='01_quickstart',
            notebook=False,  # Use JupyterLab
            debug=False
        )

        result = ExampleCommands.launch(args)

        assert result == ExitCode.SUCCESS
        mock_service.launch_example_notebook.assert_called_once_with(
            example_id='01_quickstart',
            prefer_lab=True
        )

    @patch('symfluence.cli.services.NotebookService')
    def test_launch_classic_notebook(self, mock_notebook_service_class):
        """Test launch with classic Jupyter Notebook."""
        mock_service = MagicMock()
        mock_service.launch_example_notebook.return_value = 0
        mock_notebook_service_class.return_value = mock_service

        args = Namespace(
            example_id='02_advanced',
            notebook=True,  # Use classic notebook
            debug=False
        )

        result = ExampleCommands.launch(args)

        assert result == ExitCode.SUCCESS
        mock_service.launch_example_notebook.assert_called_once_with(
            example_id='02_advanced',
            prefer_lab=False
        )

    @patch('symfluence.cli.services.NotebookService')
    def test_launch_failure(self, mock_notebook_service_class):
        """Test notebook launch failure."""
        mock_service = MagicMock()
        mock_service.launch_example_notebook.return_value = 1  # Non-zero exit
        mock_notebook_service_class.return_value = mock_service

        args = Namespace(
            example_id='01_quickstart',
            notebook=False,
            debug=False
        )

        result = ExampleCommands.launch(args)

        assert result == ExitCode.GENERAL_ERROR

    @patch('symfluence.cli.services.NotebookService')
    def test_launch_import_error(self, mock_notebook_service_class):
        """Test launch with import error."""
        mock_notebook_service_class.side_effect = ImportError("Cannot import NotebookService")

        args = Namespace(
            example_id='01_quickstart',
            notebook=False,
            debug=False
        )

        result = ExampleCommands.launch(args)

        assert result == ExitCode.DEPENDENCY_ERROR

    @patch('symfluence.cli.services.NotebookService')
    def test_launch_file_not_found(self, mock_notebook_service_class):
        """Test launch when notebook not found."""
        mock_service = MagicMock()
        mock_service.launch_example_notebook.side_effect = FileNotFoundError(
            "Notebook not found"
        )
        mock_notebook_service_class.return_value = mock_service

        args = Namespace(
            example_id='nonexistent',
            notebook=False,
            debug=False
        )

        result = ExampleCommands.launch(args)

        assert result == ExitCode.FILE_NOT_FOUND

    @patch('symfluence.cli.services.NotebookService')
    def test_launch_permission_error(self, mock_notebook_service_class):
        """Test launch with permission error."""
        mock_service = MagicMock()
        mock_service.launch_example_notebook.side_effect = PermissionError(
            "Cannot execute Jupyter"
        )
        mock_notebook_service_class.return_value = mock_service

        args = Namespace(
            example_id='01_quickstart',
            notebook=False,
            debug=False
        )

        result = ExampleCommands.launch(args)

        assert result == ExitCode.PERMISSION_ERROR


class TestExampleListExamples:
    """Test example list command."""

    def test_list_examples_with_examples_dir(self, tmp_path):
        """Test listing examples when directory exists."""
        # Create mock examples directory structure
        examples_dir = tmp_path / 'examples'
        examples_dir.mkdir()
        (examples_dir / '01_quickstart').mkdir()
        (examples_dir / '01_quickstart' / 'quickstart.ipynb').write_text('{}')
        (examples_dir / '02_advanced').mkdir()
        (examples_dir / '02_advanced' / 'advanced.ipynb').write_text('{}')

        with patch.object(Path, 'parent', new_callable=lambda: property(lambda self: tmp_path)):
            # This is tricky to test since it relies on __file__ location
            # Instead, test the basic functionality
            args = Namespace(debug=False)

            result = ExampleCommands.list_examples(args)

            # Should succeed even if directory doesn't exist at expected location
            assert result == ExitCode.SUCCESS

    def test_list_examples_no_directory(self):
        """Test listing examples when directory doesn't exist."""
        args = Namespace(debug=False)

        # This should succeed (graceful handling of missing directory)
        result = ExampleCommands.list_examples(args)

        assert result == ExitCode.SUCCESS

    def test_list_examples_permission_error(self, tmp_path):
        """Test listing examples with permission error."""
        args = Namespace(debug=False)

        # Mock Path.rglob to raise PermissionError
        with patch.object(Path, 'rglob', side_effect=PermissionError("Access denied")):
            with patch.object(Path, 'exists', return_value=True):
                result = ExampleCommands.list_examples(args)

                assert result == ExitCode.PERMISSION_ERROR
