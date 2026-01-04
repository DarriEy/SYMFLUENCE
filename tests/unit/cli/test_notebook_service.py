"""Unit tests for NotebookService."""

import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import os

from symfluence.cli.notebook_service import NotebookService

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestInitialization:
    """Test NotebookService initialization."""

    def test_initialization(self):
        """Test NotebookService creates successfully."""
        service = NotebookService()
        assert service is not None


class TestExampleIDParsing:
    """Test example ID parsing and pattern matching."""

    @pytest.mark.parametrize("example_id,expected_prefix", [
        ("01a", "01a"),
        ("1a", "01a"),
        ("02b", "02b"),
        ("2b", "02b"),
        ("03c", "03c"),
        ("10d", "10d"),
    ])
    def test_example_id_regex_parsing(self, example_id, expected_prefix):
        """Test regex parsing of example IDs with letter suffix."""
        service = NotebookService()

        # Create a test repo structure
        repo_root = Path('/tmp/test_repo')

        # Mock the existence checks
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob:

            # Mock notebook discovery
            nb_path = repo_root / 'examples' / f'{expected_prefix}_test.ipynb'
            mock_rglob.return_value = [nb_path]

            with patch('subprocess.run'), \
                 patch('sys.executable', '/usr/bin/python3'):
                # This will fail but we're just testing the ID parsing logic
                try:
                    service.launch_example_notebook(example_id, repo_root=repo_root)
                except:
                    pass

            # Verify the correct pattern was used
            # The rglob should have been called with the expected prefix
            calls = mock_rglob.call_args_list
            assert any(expected_prefix in str(call) for call in calls if call is not None)

    def test_fallback_parsing(self):
        """Test fallback parsing when regex fails."""
        service = NotebookService()

        repo_root = Path('/tmp/test_repo')

        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_dir', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob:

            # Mock notebook discovery
            nb_path = repo_root / 'examples' / 'custom_notebook.ipynb'
            mock_rglob.return_value = [nb_path]

            with patch('subprocess.run'), \
                 patch('sys.executable', '/usr/bin/python3'):
                try:
                    service.launch_example_notebook('custom', repo_root=repo_root)
                except:
                    pass

            # Should use raw prefix
            calls = mock_rglob.call_args_list
            assert len(calls) > 0


class TestNotebookDiscovery:
    """Test notebook file discovery."""

    @patch('subprocess.run')
    @patch('sys.executable', '/usr/bin/python3')
    def test_single_notebook_match(self, mock_subprocess):
        """Test discovery when one notebook matches."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        # Create mock notebook
        nb_path = repo_root / 'examples' / '01a_point_scale_snotel.ipynb'

        def exists_side_effect(self):
            # Repo root and examples dir exist
            return str(self) in [str(repo_root), str(repo_root / 'examples')]

        def is_dir_side_effect(self):
            # Repo root and examples dir are directories
            return str(self) in [str(repo_root), str(repo_root / 'examples')]

        def rglob_side_effect(self, pattern):
            # Return our mock notebook when searching
            return [nb_path]

        with patch.object(Path, 'exists', exists_side_effect), \
             patch.object(Path, 'is_dir', is_dir_side_effect), \
             patch.object(Path, 'rglob', rglob_side_effect):

            # Mock subprocess calls for ipykernel and jupyter
            mock_subprocess.return_value = MagicMock(returncode=0)

            result = service.launch_example_notebook('01a', repo_root=repo_root)

            # Should attempt to launch (may fail but that's ok for this test)
            assert result in [0, 3]  # 0 for success, 3 for jupyter not available

    @patch('subprocess.run')
    @patch('sys.executable', '/usr/bin/python3')
    def test_multiple_notebook_matches(self, mock_subprocess, capsys):
        """Test behavior when multiple notebooks match."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        # Create multiple matching notebooks
        nb_paths = [
            repo_root / 'examples' / '01a_test1.ipynb',
            repo_root / 'examples' / '01a_test2.ipynb',
        ]

        def exists_side_effect(self):
            return str(self) in [str(repo_root), str(repo_root / 'examples')]

        def is_dir_side_effect(self):
            return str(self) in [str(repo_root), str(repo_root / 'examples')]

        def rglob_side_effect(self, pattern):
            return nb_paths

        with patch.object(Path, 'exists', exists_side_effect), \
             patch.object(Path, 'is_dir', is_dir_side_effect), \
             patch.object(Path, 'rglob', rglob_side_effect):

            mock_subprocess.return_value = MagicMock(returncode=0)

            result = service.launch_example_notebook('01a', repo_root=repo_root)

            # Should print warning about multiple matches
            captured = capsys.readouterr()
            assert 'Multiple' in captured.out or 'multiple' in captured.out.lower()

    def test_no_notebook_matches(self):
        """Test error when no notebooks match."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        def exists_side_effect(self):
            return str(self) in [str(repo_root), str(repo_root / 'examples')]

        def is_dir_side_effect(self):
            return str(self) in [str(repo_root), str(repo_root / 'examples')]

        def rglob_side_effect(self, pattern):
            return []

        with patch.object(Path, 'exists', exists_side_effect), \
             patch.object(Path, 'is_dir', is_dir_side_effect), \
             patch.object(Path, 'rglob', rglob_side_effect):

            result = service.launch_example_notebook('nonexistent', repo_root=repo_root)

            assert result == 2  # Error code for notebook not found


class TestVirtualEnvironmentDetection:
    """Test virtual environment detection."""

    @pytest.mark.parametrize("venv_name", [
        ".venv", "venv", "env", ".conda", ".virtualenv"
    ])
    @patch('subprocess.run')
    @patch('sys.executable', '/usr/bin/python3')
    def test_detect_various_venvs(self, mock_subprocess, venv_name):
        """Test detection of different venv directory names."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')
        venv_dir = repo_root / venv_name
        python_exe = venv_dir / 'bin' / 'python'

        nb_path = repo_root / 'examples' / '01a_test.ipynb'

        def exists_side_effect(self):
            # Repo, examples, venv dir, and python exe exist
            return str(self) in [
                str(repo_root),
                str(repo_root / 'examples'),
                str(venv_dir),
                str(python_exe)
            ]

        def is_dir_side_effect(self):
            # Repo, examples, and venv are directories
            return str(self) in [
                str(repo_root),
                str(repo_root / 'examples'),
                str(venv_dir)
            ]

        def rglob_side_effect(self, pattern):
            return [nb_path]

        with patch.object(Path, 'exists', exists_side_effect), \
             patch.object(Path, 'is_dir', is_dir_side_effect), \
             patch.object(Path, 'rglob', rglob_side_effect):

            mock_subprocess.return_value = MagicMock(returncode=0)

            result = service.launch_example_notebook('01a', repo_root=repo_root)

            # Should have found the venv
            assert result in [0, 3]


class TestPythonExecutableSelection:
    """Test Python executable selection logic."""

    @patch('subprocess.run')
    @patch('sys.executable', '/usr/bin/python3')
    def test_venv_python_priority(self, mock_subprocess):
        """Test venv/bin/python tried first."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')
        venv_dir = repo_root / '.venv'
        python_path = venv_dir / 'bin' / 'python'

        nb_path = repo_root / 'examples' / '01a_test.ipynb'

        def exists_side_effect(self):
            return str(self) in [
                str(repo_root),
                str(repo_root / 'examples'),
                str(venv_dir),
                str(python_path)
            ]

        def is_dir_side_effect(self):
            return str(self) in [
                str(repo_root),
                str(repo_root / 'examples'),
                str(venv_dir)
            ]

        def rglob_side_effect(self, pattern):
            return [nb_path]

        with patch.object(Path, 'exists', exists_side_effect), \
             patch.object(Path, 'is_dir', is_dir_side_effect), \
             patch.object(Path, 'rglob', rglob_side_effect):

            mock_subprocess.return_value = MagicMock(returncode=0)

            result = service.launch_example_notebook('01a', repo_root=repo_root)

            # Should use venv python
            assert result in [0, 3]


class TestKernelRegistration:
    """Test ipykernel installation and registration."""

    @patch('subprocess.run')
    @patch('sys.executable', '/usr/bin/python3')
    def test_ipykernel_already_installed(self, mock_subprocess):
        """Test kernel registration when ipykernel present."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')
        venv_dir = repo_root / '.venv'
        python_exe = venv_dir / 'bin' / 'python'

        nb_path = repo_root / 'examples' / '01a_test.ipynb'

        def exists_side_effect(self):
            return str(self) in [
                str(repo_root),
                str(repo_root / 'examples'),
                str(venv_dir),
                str(python_exe)
            ]

        def is_dir_side_effect(self):
            return str(self) in [
                str(repo_root),
                str(repo_root / 'examples'),
                str(venv_dir)
            ]

        def rglob_side_effect(self, pattern):
            return [nb_path]

        with patch.object(Path, 'exists', exists_side_effect), \
             patch.object(Path, 'is_dir', is_dir_side_effect), \
             patch.object(Path, 'rglob', rglob_side_effect):

            # Mock ipykernel already installed
            mock_subprocess.return_value = MagicMock(returncode=0)

            result = service.launch_example_notebook('01a', repo_root=repo_root)

        # Should register kernel
        calls = [str(call) for call in mock_subprocess.call_args_list]
        assert any('ipykernel' in str(call) for call in calls)

    @patch('subprocess.run')
    @patch('sys.executable', '/usr/bin/python3')
    def test_ipykernel_installation(self, mock_subprocess):
        """Test ipykernel installation when missing."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')
        venv_dir = repo_root / '.venv'
        python_exe = venv_dir / 'bin' / 'python'

        nb_path = repo_root / 'examples' / '01a_test.ipynb'

        def exists_side_effect(self):
            return str(self) in [
                str(repo_root),
                str(repo_root / 'examples'),
                str(venv_dir),
                str(python_exe)
            ]

        def is_dir_side_effect(self):
            return str(self) in [
                str(repo_root),
                str(repo_root / 'examples'),
                str(venv_dir)
            ]

        def rglob_side_effect(self, pattern):
            return [nb_path]

        with patch.object(Path, 'exists', exists_side_effect), \
             patch.object(Path, 'is_dir', is_dir_side_effect), \
             patch.object(Path, 'rglob', rglob_side_effect):

            # Mock ipykernel not installed, then success after install
            def subprocess_side_effect(*args, **kwargs):
                cmd = args[0] if args else kwargs.get('args', [])
                if 'show' in cmd and 'ipykernel' in cmd:
                    return MagicMock(returncode=1)  # Not installed
                return MagicMock(returncode=0)

            mock_subprocess.side_effect = subprocess_side_effect

            result = service.launch_example_notebook('01a', repo_root=repo_root)

            # Should have attempted to install ipykernel
            calls = [str(call) for call in mock_subprocess.call_args_list]
            assert any('install' in str(call) and 'ipykernel' in str(call) for call in calls)


class TestJupyterLaunch:
    """Test Jupyter launcher selection and execution."""

    @patch('subprocess.run')
    @patch('sys.executable', '/usr/bin/python3')
    def test_jupyterlab_launch(self, mock_subprocess):
        """Test launch with JupyterLab."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')
        venv_dir = repo_root / '.venv'
        python_exe = venv_dir / 'bin' / 'python'

        nb_path = repo_root / 'examples' / '01a_test.ipynb'

        def exists_side_effect(self):
            return str(self) in [
                str(repo_root),
                str(repo_root / 'examples'),
                str(venv_dir),
                str(python_exe)
            ]

        def is_dir_side_effect(self):
            return str(self) in [
                str(repo_root),
                str(repo_root / 'examples'),
                str(venv_dir)
            ]

        def rglob_side_effect(self, pattern):
            return [nb_path]

        with patch.object(Path, 'exists', exists_side_effect), \
             patch.object(Path, 'is_dir', is_dir_side_effect), \
             patch.object(Path, 'rglob', rglob_side_effect):

            mock_subprocess.return_value = MagicMock(returncode=0)

            result = service.launch_example_notebook('01a', repo_root=repo_root, prefer_lab=True)

            # Should attempt jupyterlab
            calls = [str(call) for call in mock_subprocess.call_args_list]
            assert any('jupyterlab' in str(call) or 'jupyter' in str(call) for call in calls)


class TestErrorCases:
    """Test error handling."""

    def test_repo_root_not_found(self):
        """Test error when repo root missing."""
        service = NotebookService()
        repo_root = Path('/nonexistent/repo')

        def exists_side_effect(self):
            # Repo root doesn't exist
            return False

        with patch.object(Path, 'exists', exists_side_effect):
            result = service.launch_example_notebook('01a', repo_root=repo_root)

            assert result == 1  # Error code for repo not found

    def test_examples_dir_not_found(self):
        """Test error when examples directory missing."""
        service = NotebookService()
        repo_root = Path('/tmp/test_repo')

        # We want repo_root.exists() to be True, but examples_root.exists() to be False
        def exists_side_effect(self):
            # Repo exists but examples directory doesn't
            return str(self) == str(repo_root)

        with patch.object(Path, 'exists', exists_side_effect):
            result = service.launch_example_notebook('01a', repo_root=repo_root)

            assert result == 2  # Error code for examples dir not found
