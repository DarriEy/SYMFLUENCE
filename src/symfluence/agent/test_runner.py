# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Test runner module for executing pytest and parsing results.

Runs tests on modified code and provides formatted results.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


class TestRunner:
    """Run and manage tests for code modifications."""

    def __init__(self, repo_root: Optional[str] = None):
        """
        Initialize test runner.

        Args:
            repo_root: Repository root directory. If None, uses current directory.
        """
        if repo_root is None:
            import os
            repo_root = os.getcwd()
        self.repo_root = Path(repo_root).resolve()
        self.tests_dir = self.repo_root / "tests"

    def run_tests(
        self,
        test_pattern: Optional[str] = None,
        files: Optional[List[str]] = None,
        verbose: bool = False,
        quiet: bool = False
    ) -> Tuple[bool, str]:
        """
        Run tests using pytest.

        Args:
            test_pattern: pytest pattern (e.g., "test_agent", "tests/unit/test_*.py")
            files: Specific files to test (e.g., ["tests/unit/agent/test_file_ops.py"])
            verbose: Verbose output
            quiet: Quiet output (only summary)

        Returns:
            Tuple of (success, output_or_error)
        """
        try:
            if not self.tests_dir.exists():
                return False, f"Tests directory not found: {self.tests_dir}"

            # Build pytest command
            cmd = ["pytest"]

            # Add verbosity
            if verbose:
                cmd.append("-vv")
            elif not quiet:
                cmd.append("-v")

            # Add coverage report
            cmd.extend(["--tb=short"])

            # Specify test target
            if files:
                for file_path in files:
                    full_path = self.repo_root / file_path
                    if full_path.exists():
                        cmd.append(str(full_path))
                    else:
                        return False, f"Test file not found: {file_path}"
            elif test_pattern:
                cmd.append(str(self.tests_dir))
                cmd.extend(["-k", test_pattern])
            else:
                cmd.append(str(self.tests_dir))

            # Run pytest
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            # Parse output
            output = result.stdout + result.stderr
            success = result.returncode == 0

            # Format output
            formatted = self._format_test_output(output, success)

            return success, formatted

        except subprocess.TimeoutExpired:
            return False, "Tests timed out after 2 minutes"
        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error running tests: {str(e)}"

    def run_specific_test(self, test_path: str) -> Tuple[bool, str]:
        """
        Run a specific test file or test case.

        Args:
            test_path: Path to test file or test case (e.g., "tests/unit/test_agent.py::test_read_file")

        Returns:
            Tuple of (success, output_or_error)
        """
        try:
            full_path = self.repo_root / test_path
            if not full_path.exists() and "::" not in test_path:
                return False, f"Test not found: {test_path}"

            cmd = ["pytest", "-vv", test_path, "--tb=short"]

            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=60
            )

            output = result.stdout + result.stderr
            success = result.returncode == 0

            formatted = self._format_test_output(output, success)
            return success, formatted

        except subprocess.TimeoutExpired:
            return False, "Test timed out after 1 minute"
        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error running test: {str(e)}"

    def run_tests_for_files(self, modified_files: List[str]) -> Tuple[bool, str]:
        """
        Run tests for specific modified files.

        Args:
            modified_files: List of modified source files

        Returns:
            Tuple of (success, output_or_error)
        """
        try:
            # Find corresponding test files
            test_files = self._find_test_files(modified_files)

            if not test_files:
                return True, (
                    "No test files found for modified files.\n"
                    "Consider adding tests for these changes:\n"
                    + "\n".join(f"  - {f}" for f in modified_files)
                )

            # Run tests
            return self.run_tests(files=test_files, verbose=True)

        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error finding test files: {str(e)}"

    def check_syntax(self, python_file: str) -> Tuple[bool, str]:
        """
        Check Python file syntax without running tests.

        Args:
            python_file: Path to Python file

        Returns:
            Tuple of (success, message_or_error)
        """
        try:
            full_path = self.repo_root / python_file
            if not full_path.exists():
                return False, f"File not found: {python_file}"

            if not full_path.suffix == '.py':
                return False, "Not a Python file"

            # Use python -m py_compile
            result = subprocess.run(
                ["python", "-m", "py_compile", str(full_path)],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return True, f"✓ Syntax valid: {python_file}"
            else:
                return False, f"Syntax error: {result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "Syntax check timed out"
        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error checking syntax: {str(e)}"

    def get_test_coverage(self, source_path: str = "src") -> Tuple[bool, str]:
        """
        Get test coverage report for source code.

        Args:
            source_path: Path to source directory

        Returns:
            Tuple of (success, report_or_error)
        """
        try:
            cmd = [
                "pytest",
                str(self.tests_dir),
                f"--cov={source_path}",
                "--cov-report=term-missing",
                "-v"
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=120
            )

            if "No module named" in result.stderr:
                return False, "Coverage module not installed: pip install pytest-cov"

            output = result.stdout + result.stderr
            success = result.returncode == 0

            return success, output

        except subprocess.TimeoutExpired:
            return False, "Coverage analysis timed out"
        except Exception as e:  # noqa: BLE001 — agent resilience
            return False, f"Error getting coverage: {str(e)}"

    # Helper methods

    def _format_test_output(self, output: str, success: bool) -> str:
        """Format pytest output for readability."""
        lines = output.split('\n')

        # Extract summary line
        summary = None
        for line in lines:
            if 'passed' in line or 'failed' in line or 'error' in line:
                if '==' in line:  # Summary line
                    summary = line
                    break

        formatted = "Test Results\n" + "=" * 60 + "\n\n"

        if success:
            formatted += "✓ All tests passed\n"
        else:
            formatted += "✗ Some tests failed\n"

        formatted += "\n"

        # Add summary
        if summary:
            formatted += f"{summary}\n"
        else:
            # Try to extract pass/fail counts
            for line in lines[-20:]:  # Check last 20 lines
                if 'passed' in line or 'failed' in line:
                    formatted += f"{line}\n"

        # Add failed test details if any
        if not success and "FAILED" in output:
            formatted += "\nFailed Tests:\n"
            in_failures = False
            for line in lines:
                if "FAILED" in line:
                    in_failures = True
                if in_failures:
                    formatted += f"  {line}\n"
                    if "short test summary" in line.lower():
                        break

        return formatted

    def _find_test_files(self, source_files: List[str]) -> List[str]:
        """
        Find test files corresponding to source files.

        Args:
            source_files: List of source file paths

        Returns:
            List of test file paths
        """
        test_files = []

        for src_file in source_files:
            # Convert src/module.py to tests/unit/test_module.py
            src_path = Path(src_file)

            # Handle different test directory structures
            possible_test_paths = [
                self.tests_dir / "unit" / f"test_{src_path.stem}.py",
                self.tests_dir / f"test_{src_path.stem}.py",
                self.tests_dir / src_path.parent.name / f"test_{src_path.stem}.py",
            ]

            for test_path in possible_test_paths:
                if test_path.exists():
                    test_files.append(str(test_path.relative_to(self.repo_root)))
                    break

        return test_files
