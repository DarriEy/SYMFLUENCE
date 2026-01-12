"""
File operations module for agent code reading and modification.

Provides safe file I/O operations with caching, syntax validation, and git integration.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from functools import lru_cache


class FileOperations:
    """Safe file operations for agent code reading and modification."""

    # Allowed directories for reading and writing
    ALLOWED_READ_ROOTS = ['src/', 'tests/', '.']
    ALLOWED_WRITE_ROOTS = ['src/', 'tests/']
    BLOCKED_FILES = {'.git', '.github', 'pyproject.toml', '.gitignore', '.env', '.env.local'}

    def __init__(self, repo_root: Optional[str] = None):
        """
        Initialize file operations.

        Args:
            repo_root: Repository root directory. If None, uses current directory.
        """
        if repo_root is None:
            repo_root = os.getcwd()
        self.repo_root = Path(repo_root).resolve()

    def read_file(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Read a source file safely with optional line range.

        Args:
            file_path: Path relative to repo root
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (1-indexed, inclusive)

        Returns:
            Tuple of (success, content_or_error)
        """
        try:
            full_path = self._resolve_path(file_path)

            # Security check
            if not self._is_allowed_read(full_path):
                return False, f"Access denied: {file_path} is outside allowed directories"

            # File existence check
            if not full_path.exists():
                return False, f"File not found: {file_path}"

            # Check file size (prevent reading huge files)
            if full_path.stat().st_size > 1_000_000:  # 1MB limit
                return False, f"File too large: {file_path} (max 1MB)"

            # Read file
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Apply line range
            if start_line is not None or end_line is not None:
                start = (start_line or 1) - 1
                end = (end_line or len(lines))
                lines = lines[start:end]

            # Format with line numbers
            if start_line:
                formatted = []
                for i, line in enumerate(lines, start=start_line):
                    formatted.append(f"{i:4d}→{line.rstrip()}")
                content = "\n".join(formatted)
            else:
                formatted = []
                for i, line in enumerate(lines, start=1):
                    formatted.append(f"{i:4d}→{line.rstrip()}")
                content = "\n".join(formatted)

            return True, content

        except Exception as e:
            return False, f"Error reading file: {str(e)}"

    def write_file(self, file_path: str, content: str) -> Tuple[bool, str]:
        """
        Write to a file safely with validation.

        Args:
            file_path: Path relative to repo root
            content: File content to write

        Returns:
            Tuple of (success, message_or_error)
        """
        try:
            full_path = self._resolve_path(file_path)

            # Security check
            if not self._is_allowed_write(full_path):
                return False, f"Cannot write to {file_path}: outside allowed directories or blocked file"

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # For Python files, validate syntax
            if full_path.suffix == '.py':
                success, error = self._validate_python_syntax(content)
                if not success:
                    return False, f"Syntax error in Python file: {error}"

            # Write file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True, f"Successfully wrote {file_path}"

        except Exception as e:
            return False, f"Error writing file: {str(e)}"

    def list_directory(
        self,
        directory: str = ".",
        recursive: bool = False,
        pattern: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        List directory contents safely.

        Args:
            directory: Directory path relative to repo root
            recursive: Show full tree if True
            pattern: File pattern filter (e.g., "*.py", "test_*")

        Returns:
            Tuple of (success, content_or_error)
        """
        try:
            full_path = self._resolve_path(directory)

            # Security check
            if not self._is_allowed_read(full_path):
                return False, f"Access denied: {directory}"

            if not full_path.is_dir():
                return False, f"Not a directory: {directory}"

            # List contents
            items = []
            for item in sorted(full_path.iterdir()):
                if item.name.startswith('.'):
                    continue

                # Apply pattern filter
                if pattern and not self._matches_pattern(item.name, pattern):
                    continue

                rel_path = item.relative_to(self.repo_root)
                if item.is_dir():
                    items.append(f"📁 {rel_path}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f}MB"
                    items.append(f"📄 {rel_path} ({size_str})")

            if not items:
                content = "Directory is empty"
            else:
                content = "\n".join(items)

            return True, content

        except Exception as e:
            return False, f"Error listing directory: {str(e)}"

    def show_diff(self, file_path: str) -> Tuple[bool, str]:
        """
        Show git diff for a file or staged changes.

        Args:
            file_path: File path or "." for all staged changes

        Returns:
            Tuple of (success, diff_or_error)
        """
        try:
            if file_path == ".":
                # Show all staged changes
                result = subprocess.run(
                    ["git", "diff", "--cached"],
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            else:
                # Show diff for specific file
                full_path = self._resolve_path(file_path)
                if not full_path.exists():
                    return False, f"File not found: {file_path}"

                result = subprocess.run(
                    ["git", "diff", "--cached", str(full_path)],
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                    timeout=10
                )

            if result.returncode not in (0, 1):
                return False, f"Git diff failed: {result.stderr}"

            output = result.stdout or "(no changes)"
            return True, output

        except Exception as e:
            return False, f"Error showing diff: {str(e)}"

    def stage_changes(self, file_paths: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Stage changes for commit.

        Args:
            file_paths: Files to stage. If None, stages all changes.

        Returns:
            Tuple of (success, message_or_error)
        """
        try:
            if file_paths is None:
                # Stage all
                result = subprocess.run(
                    ["git", "add", "-A"],
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            else:
                # Stage specific files
                for file_path in file_paths:
                    full_path = self._resolve_path(file_path)
                    result = subprocess.run(
                        ["git", "add", str(full_path)],
                        cwd=str(self.repo_root),
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode != 0:
                        return False, f"Failed to stage {file_path}: {result.stderr}"

            # Get status
            status = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=10
            )

            return True, f"Successfully staged changes\n\n{status.stdout}"

        except Exception as e:
            return False, f"Error staging changes: {str(e)}"

    def get_staged_changes(self) -> Tuple[bool, str]:
        """
        Get all staged changes as diff.

        Returns:
            Tuple of (success, diff_or_error)
        """
        return self.show_diff(".")

    # Helper methods

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a file path relative to repo root."""
        if file_path.startswith('/'):
            # Absolute path
            return Path(file_path).resolve()
        else:
            # Relative to repo root
            return (self.repo_root / file_path).resolve()

    def _is_allowed_read(self, full_path: Path) -> bool:
        """Check if a path is allowed for reading."""
        try:
            full_path.relative_to(self.repo_root)
            return True
        except ValueError:
            return False

    def _is_allowed_write(self, full_path: Path) -> bool:
        """Check if a path is allowed for writing."""
        # Check if outside repo
        try:
            rel_path = full_path.relative_to(self.repo_root)
        except ValueError:
            return False

        # Check if in blocked files
        if full_path.name in self.BLOCKED_FILES:
            return False

        # Check if in allowed write roots
        for root in self.ALLOWED_WRITE_ROOTS:
            root_path = self.repo_root / root
            try:
                full_path.relative_to(root_path)
                return True
            except ValueError:
                continue

        return False

    def _validate_python_syntax(self, content: str) -> Tuple[bool, str]:
        """Validate Python code syntax."""
        try:
            import py_compile
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                py_compile.compile(temp_file, doraise=True)
                return True, ""
            except py_compile.PyCompileError as e:
                return False, str(e)
            finally:
                os.unlink(temp_file)

        except Exception as e:
            return False, f"Syntax check failed: {str(e)}"

    @staticmethod
    def _matches_pattern(name: str, pattern: str) -> bool:
        """Simple pattern matching (supports * wildcard)."""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)
