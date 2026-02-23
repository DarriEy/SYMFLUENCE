"""
Tests for the FileOperations class.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestFileOperationsInitialization:
    """Tests for FileOperations initialization."""

    def test_file_operations_can_be_imported(self):
        """Test that FileOperations can be imported."""
        from symfluence.agent.file_operations import FileOperations
        assert FileOperations is not None

    def test_initialization_default_repo_root(self):
        """Test initialization uses current directory by default."""
        import os

        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations()
        assert ops.repo_root == Path(os.getcwd()).resolve()

    def test_initialization_custom_repo_root(self, temp_dir):
        """Test initialization with custom repo root."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        assert ops.repo_root == temp_dir.resolve()


class TestReadFile:
    """Tests for file reading operations."""

    def test_read_file_success(self, temp_dir):
        """Test successful file reading."""
        from symfluence.agent.file_operations import FileOperations

        # Create test file
        test_file = temp_dir / 'test.py'
        test_file.write_text("print('hello')\nprint('world')\n")

        ops = FileOperations(repo_root=str(temp_dir))
        success, content = ops.read_file('test.py')

        assert success is True
        assert "print('hello')" in content
        assert "print('world')" in content

    def test_read_file_with_line_numbers(self, temp_dir):
        """Test file reading includes line numbers."""
        from symfluence.agent.file_operations import FileOperations

        test_file = temp_dir / 'test.py'
        test_file.write_text("line1\nline2\nline3\n")

        ops = FileOperations(repo_root=str(temp_dir))
        success, content = ops.read_file('test.py')

        assert success is True
        # Line numbers should be present
        assert "1" in content
        assert "2" in content

    def test_read_file_with_range(self, temp_dir):
        """Test file reading with line range."""
        from symfluence.agent.file_operations import FileOperations

        test_file = temp_dir / 'test.py'
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        ops = FileOperations(repo_root=str(temp_dir))
        success, content = ops.read_file('test.py', start_line=2, end_line=4)

        assert success is True
        assert "line2" in content
        assert "line3" in content
        assert "line4" in content
        # line1 and line5 should not be in the output
        assert "line1" not in content or "line5" not in content

    def test_read_file_not_found(self, temp_dir):
        """Test reading non-existent file."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        success, error = ops.read_file('nonexistent.py')

        assert success is False
        assert "not found" in error.lower()

    def test_read_file_outside_repo(self, temp_dir):
        """Test reading file outside repo root."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        success, error = ops.read_file('/etc/passwd')

        assert success is False
        assert "denied" in error.lower() or "outside" in error.lower()


class TestWriteFile:
    """Tests for file writing operations."""

    def test_write_file_success(self, temp_dir):
        """Test successful file writing."""
        from symfluence.agent.file_operations import FileOperations

        # Create src directory (allowed write root)
        src_dir = temp_dir / 'src'
        src_dir.mkdir()

        ops = FileOperations(repo_root=str(temp_dir))
        success, message = ops.write_file('src/new_file.py', "print('hello')")

        assert success is True
        assert (src_dir / 'new_file.py').exists()

    def test_write_file_creates_directories(self, temp_dir):
        """Test writing file creates parent directories."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        success, message = ops.write_file('src/subdir/new_file.py', "print('hello')")

        assert success is True
        assert (temp_dir / 'src' / 'subdir' / 'new_file.py').exists()

    def test_write_file_validates_python_syntax(self, temp_dir):
        """Test writing Python file validates syntax."""
        from symfluence.agent.file_operations import FileOperations

        src_dir = temp_dir / 'src'
        src_dir.mkdir()

        ops = FileOperations(repo_root=str(temp_dir))
        success, error = ops.write_file('src/bad.py', "def broken syntax(")

        assert success is False
        assert "syntax" in error.lower()

    def test_write_file_blocked(self, temp_dir):
        """Test writing to blocked file fails."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        success, error = ops.write_file('.env', "SECRET=key")

        assert success is False


class TestListDirectory:
    """Tests for directory listing operations."""

    def test_list_directory_success(self, temp_dir):
        """Test successful directory listing."""
        from symfluence.agent.file_operations import FileOperations

        # Create some files
        (temp_dir / 'file1.py').write_text("test")
        (temp_dir / 'file2.txt').write_text("test")
        subdir = temp_dir / 'subdir'
        subdir.mkdir()

        ops = FileOperations(repo_root=str(temp_dir))
        success, content = ops.list_directory('.')

        assert success is True
        assert 'file1.py' in content
        assert 'file2.txt' in content
        assert 'subdir' in content

    def test_list_directory_with_pattern(self, temp_dir):
        """Test directory listing with pattern filter."""
        from symfluence.agent.file_operations import FileOperations

        (temp_dir / 'file1.py').write_text("test")
        (temp_dir / 'file2.txt').write_text("test")
        (temp_dir / 'test_file.py').write_text("test")

        ops = FileOperations(repo_root=str(temp_dir))
        success, content = ops.list_directory('.', pattern='*.py')

        assert success is True
        assert 'file1.py' in content
        assert 'test_file.py' in content
        assert 'file2.txt' not in content

    def test_list_directory_not_found(self, temp_dir):
        """Test listing non-existent directory."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        success, error = ops.list_directory('nonexistent')

        assert success is False

    def test_list_directory_hides_hidden_files(self, temp_dir):
        """Test that hidden files are not listed."""
        from symfluence.agent.file_operations import FileOperations

        (temp_dir / '.hidden').write_text("test")
        (temp_dir / 'visible.py').write_text("test")

        ops = FileOperations(repo_root=str(temp_dir))
        success, content = ops.list_directory('.')

        assert success is True
        assert 'visible.py' in content
        assert '.hidden' not in content


class TestGitOperations:
    """Tests for git-related operations."""

    @patch('subprocess.run')
    def test_show_diff(self, mock_run, temp_dir):
        """Test showing git diff."""
        from symfluence.agent.file_operations import FileOperations

        mock_run.return_value = MagicMock(returncode=0, stdout="diff output", stderr="")

        ops = FileOperations(repo_root=str(temp_dir))
        success, diff = ops.show_diff('.')

        assert success is True
        assert diff == "diff output"

    @patch('subprocess.run')
    def test_show_diff_no_changes(self, mock_run, temp_dir):
        """Test showing diff when no changes."""
        from symfluence.agent.file_operations import FileOperations

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        ops = FileOperations(repo_root=str(temp_dir))
        success, diff = ops.show_diff('.')

        assert success is True
        assert "(no changes)" in diff

    @patch('subprocess.run')
    def test_stage_changes(self, mock_run, temp_dir):
        """Test staging changes."""
        from symfluence.agent.file_operations import FileOperations

        mock_run.return_value = MagicMock(returncode=0, stdout="M file.py", stderr="")

        ops = FileOperations(repo_root=str(temp_dir))
        success, message = ops.stage_changes()

        assert success is True


class TestSecurityChecks:
    """Tests for security-related checks."""

    def test_is_allowed_read_inside_repo(self, temp_dir):
        """Test allowed read inside repo root."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        # Create and use an actual file to ensure path resolution works
        test_file = temp_dir / 'test.py'
        test_file.touch()
        full_path = test_file.resolve()

        assert ops._is_allowed_read(full_path) is True

    def test_is_allowed_read_outside_repo(self, temp_dir):
        """Test disallowed read outside repo root."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        full_path = Path('/etc/passwd')

        assert ops._is_allowed_read(full_path) is False

    def test_is_allowed_write_in_src(self, temp_dir):
        """Test allowed write in src directory."""
        from symfluence.agent.file_operations import FileOperations

        src_dir = temp_dir / 'src'
        src_dir.mkdir()
        # Resolve the path to ensure it matches the repo root
        ops = FileOperations(repo_root=str(temp_dir.resolve()))
        full_path = (src_dir / 'test.py').resolve()

        assert ops._is_allowed_write(full_path) is True

    def test_is_allowed_write_blocked_file(self, temp_dir):
        """Test disallowed write to blocked file."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        full_path = temp_dir / '.env'

        assert ops._is_allowed_write(full_path) is False


class TestPythonSyntaxValidation:
    """Tests for Python syntax validation."""

    def test_validate_python_syntax_valid(self, temp_dir):
        """Test valid Python syntax passes."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        success, error = ops._validate_python_syntax("def hello(): pass")

        assert success is True
        assert error == ""

    def test_validate_python_syntax_invalid(self, temp_dir):
        """Test invalid Python syntax fails."""
        from symfluence.agent.file_operations import FileOperations

        ops = FileOperations(repo_root=str(temp_dir))
        success, error = ops._validate_python_syntax("def broken(")

        assert success is False
        assert len(error) > 0


class TestPatternMatching:
    """Tests for file pattern matching."""

    def test_matches_pattern_exact(self):
        """Test exact pattern match."""
        from symfluence.agent.file_operations import FileOperations

        assert FileOperations._matches_pattern("test.py", "test.py") is True

    def test_matches_pattern_wildcard(self):
        """Test wildcard pattern match."""
        from symfluence.agent.file_operations import FileOperations

        assert FileOperations._matches_pattern("test.py", "*.py") is True
        assert FileOperations._matches_pattern("test.txt", "*.py") is False

    def test_matches_pattern_prefix(self):
        """Test prefix pattern match."""
        from symfluence.agent.file_operations import FileOperations

        assert FileOperations._matches_pattern("test_file.py", "test_*") is True
        assert FileOperations._matches_pattern("main.py", "test_*") is False
