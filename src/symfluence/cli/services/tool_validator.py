"""
Tool validation service for SYMFLUENCE.

Validates that required binary executables exist and are functional.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..console import Console
from .base import BaseService


class ToolValidator(BaseService):
    """
    Service for validating external tool installations.

    Handles:
    - Binary existence checks
    - Executable testing
    - Version verification
    """

    def __init__(
        self,
        external_tools: Optional[Dict[str, Any]] = None,
        console: Optional[Console] = None,
    ):
        """
        Initialize the ToolValidator.

        Args:
            external_tools: Dictionary of tool definitions. If None, loads on demand.
            console: Console instance for output.
        """
        super().__init__(console=console)
        self._external_tools = external_tools

    @property
    def external_tools(self) -> Dict[str, Any]:
        """Lazy load external tools definitions."""
        if self._external_tools is None:
            from ..external_tools_config import get_external_tools_definitions
            self._external_tools = get_external_tools_definitions()
        return self._external_tools

    def validate(
        self, symfluence_instance=None, verbose: bool = False
    ) -> Union[bool, Dict[str, Any]]:
        """
        Validate that required binary executables exist and are functional.

        Args:
            symfluence_instance: Optional SYMFLUENCE instance with config.
            verbose: If True, show detailed output.

        Returns:
            True if all tools valid, otherwise a dictionary with validation results.
        """
        self._console.panel("Validating External Tool Binaries", style="blue")

        validation_results: Dict[str, Any] = {
            "valid_tools": [],
            "missing_tools": [],
            "failed_tools": [],
            "skipped_tools": [],
            "warnings": [],
            "summary": {},
        }

        config = self._load_config(symfluence_instance)

        # Validate each tool (skip optional tools that aren't installed)
        for tool_name, tool_info in self.external_tools.items():
            if tool_info.get('hidden', False):
                validation_results["skipped_tools"].append(tool_name)
                continue
            if tool_info.get('optional', False):
                # Check if optional tool is actually installed before validating.
                # Use default_path_suffix (e.g. installs/clm/bin) which includes
                # the output subdirectory — NOT the install_dir root, which exists
                # whenever the repo was cloned even if the build failed.
                data_dir = self._get_data_dir(config)
                opt_path = data_dir / tool_info.get("default_path_suffix", "")
                if not opt_path.exists():
                    validation_results["skipped_tools"].append(tool_name)
                    continue  # Skip optional tools that aren't installed
            self._console.newline()
            self._console.info(f"Checking {tool_name.upper()}:")
            tool_result = {
                "name": tool_name,
                "description": tool_info.get("description", ""),
                "status": "unknown",
                "path": None,
                "executable": None,
                "version": None,
                "errors": [],
            }

            try:
                # Determine tool path (config override or default)
                config_path_key = tool_info.get("config_path_key")
                tool_path = (
                    config.get(config_path_key, "default") if config_path_key else "default"
                )
                if tool_path == "default":
                    data_dir = self._get_data_dir(config)
                    tool_path = data_dir / tool_info.get("default_path_suffix", "")
                else:
                    tool_path = Path(tool_path)
                tool_result["path"] = str(tool_path)

                # Check using verify_install block if present
                if self._check_verify_install(
                    tool_name, tool_info, tool_path, tool_result, validation_results
                ):
                    continue

                # Fallback: single-executable check
                exe_path = self._get_executable_path(tool_info, tool_path, config)
                if exe_path is None:
                    tool_result["status"] = "missing"
                    tool_result["errors"].append(f"Executable not found at: {tool_path}")
                    validation_results["missing_tools"].append(tool_name)
                    self._console.error(f"Not found: {tool_path}")
                    self._console.indent(
                        f"Try: symfluence binary install {tool_name}"
                    )
                else:
                    tool_result["executable"] = exe_path.name

                    test_cmd = tool_info.get("test_command")
                    if test_cmd is None:
                        tool_result["status"] = "valid"
                        tool_result["version"] = "Installed (existence verified)"
                        validation_results["valid_tools"].append(tool_name)
                        self._console.success(f"Found at: {exe_path}")
                        self._console.success("Status: Installed")
                    else:
                        self._run_test_command(
                            tool_name, exe_path, test_cmd, tool_result, validation_results
                        )

            except Exception as e:  # noqa: BLE001 — top-level fallback
                tool_result["status"] = "error"
                tool_result["errors"].append(f"Validation error: {str(e)}")
                validation_results["failed_tools"].append(tool_name)
                self._console.error(f"Validation error: {str(e)}")

            validation_results["summary"][tool_name] = tool_result

        # Print summary
        self._print_validation_summary(validation_results)

        if (
            len(validation_results["missing_tools"]) == 0
            and len(validation_results["failed_tools"]) == 0
        ):
            return True
        else:
            return validation_results

    def _check_verify_install(
        self,
        tool_name: str,
        tool_info: Dict[str, Any],
        tool_path: Path,
        tool_result: Dict[str, Any],
        validation_results: Dict[str, Any],
    ) -> bool:
        """
        Check tool using verify_install block.

        Args:
            tool_name: Name of the tool.
            tool_info: Tool configuration.
            tool_path: Path to the tool.
            tool_result: Result dictionary to update.
            validation_results: Overall results to update.

        Returns:
            True if check was performed and passed, False otherwise.
        """
        verify = tool_info.get("verify_install")
        if not verify or not isinstance(verify, dict):
            return False

        check_type = verify.get("check_type", "exists_all")
        file_paths = verify.get("file_paths", [])
        candidates = [tool_path / p for p in file_paths]

        # Also try from install_dir root (handles cases where default_path_suffix
        # includes a subdirectory like 'bin/' that overlaps with file_paths)
        install_dir = tool_info.get("install_dir", "")
        if install_dir:
            data_dir = self._get_data_dir(self._load_config(None))
            install_root = data_dir / "installs" / install_dir
            candidates.extend(install_root / p for p in file_paths)

        if check_type in ("python_module", "python_import"):
            module_name = verify.get("python_import", tool_name)
            return self._check_python_module(
                tool_name, module_name, tool_info, tool_path,
                tool_result, validation_results,
            )

        if check_type == "exists_any":
            found_path = None
            for p in candidates:
                if p.exists():
                    found_path = p
                    break
            exists_ok = found_path is not None
        elif check_type in ("exists_all", "exists"):
            exists_ok = all(p.exists() for p in candidates)
            if exists_ok:
                found_path = next(
                    (p for p in candidates if p.exists()),
                    candidates[0] if candidates else tool_path,
                )
        else:
            exists_ok = False
            found_path = None

        if exists_ok:
            test_cmd = tool_info.get("test_command")
            if test_cmd is None:
                tool_result["status"] = "valid"
                tool_result["version"] = "Installed (existence verified)"
                validation_results["valid_tools"].append(tool_name)
                self._console.success(f"Found at: {found_path}")
                self._console.success("Status: Installed")
                validation_results["summary"][tool_name] = tool_result
                return True

        return False

    def _check_python_module(
        self,
        tool_name: str,
        module_name: str,
        tool_info: Dict[str, Any],
        tool_path: Path,
        tool_result: Dict[str, Any],
        validation_results: Dict[str, Any],
    ) -> bool:
        """
        Check if a Python module is importable.

        Args:
            tool_name: Name of the tool.
            module_name: Python module name to import.
            tool_info: Tool configuration.
            tool_path: Path to the tool installation.
            tool_result: Result dictionary to update.
            validation_results: Overall results to update.

        Returns:
            True if check was performed (pass or fail), False otherwise.
        """
        try:
            # Build import snippet, honouring optional pre-imports
            # (e.g. cfuse requires torch to be imported first on Windows
            #  so that the correct DLL search paths are established)
            verify = tool_info.get("verify_install") or {}
            pre_imports = verify.get("pre_imports", [])
            import_lines = "".join(f"import {m}; " for m in pre_imports)
            import_snippet = (
                f"{import_lines}"
                f"import {module_name}; "
                f"print(getattr({module_name}, '__version__', 'installed'))"
            )

            # Set KMP_DUPLICATE_LIB_OK to avoid OpenMP runtime conflicts
            # on Windows when multiple libraries bundle their own OpenMP
            env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

            result = subprocess.run(
                [sys.executable, "-c", import_snippet],
                capture_output=True, text=True, timeout=15,
                env=env,
            )

            if result.returncode == 0:
                version = result.stdout.strip()[:100] if result.stdout.strip() else "Available"
                tool_result["status"] = "valid"
                tool_result["version"] = version
                validation_results["valid_tools"].append(tool_name)
                self._console.success(f"Found at: {tool_path}")
                self._console.success("Status: Working")
                validation_results["summary"][tool_name] = tool_result
                return True
            else:
                tool_result["status"] = "missing"
                error_msg = result.stderr.strip()[:200] if result.stderr else "Import failed"
                tool_result["errors"].append(error_msg)
                validation_results["missing_tools"].append(tool_name)
                self._console.error(f"Python module '{module_name}' not importable")
                self._console.indent(
                    f"Try: symfluence binary install {tool_name}"
                )
                validation_results["summary"][tool_name] = tool_result
                return True

        except subprocess.TimeoutExpired:
            tool_result["status"] = "timeout"
            tool_result["errors"].append("Import test timed out")
            validation_results["warnings"].append(f"{tool_name}: import test timed out")
            self._console.warning(f"Import test timed out for: {module_name}")
            validation_results["summary"][tool_name] = tool_result
            return True

    def _get_executable_path(
        self,
        tool_info: Dict[str, Any],
        tool_path: Path,
        config: Dict[str, Any],
    ) -> Optional[Path]:
        """
        Get the path to the tool's executable.

        Args:
            tool_info: Tool configuration.
            tool_path: Base path to the tool.
            config: Configuration dictionary.

        Returns:
            Path to the executable if found, None otherwise.
        """
        config_exe_key = tool_info.get("config_exe_key")
        if config_exe_key and config_exe_key in config:
            exe_name = config[config_exe_key]
        else:
            exe_name = tool_info.get("default_exe", "")

        if not exe_name:
            return None

        # Handle shared library extension on macOS
        if exe_name.endswith(".so") and sys.platform == "darwin":
            exe_name = exe_name.replace(".so", ".dylib")

        # Handle shared library extension on Windows
        if exe_name.endswith(".so") and sys.platform == "win32":
            exe_name = exe_name.replace(".so", ".dll")

        exe_path = tool_path / exe_name
        if exe_path.exists():
            return exe_path

        # Try without extension
        exe_name_no_ext = exe_name.replace(".exe", "")
        exe_path_no_ext = tool_path / exe_name_no_ext
        if exe_path_no_ext.exists():
            return exe_path_no_ext

        # On Windows, try adding .exe extension
        if sys.platform == "win32" and not exe_name.endswith(".exe"):
            exe_path_win = tool_path / (exe_name + ".exe")
            if exe_path_win.exists():
                return exe_path_win

        return None

    def _run_test_command(
        self,
        tool_name: str,
        exe_path: Path,
        test_cmd: str,
        tool_result: Dict[str, Any],
        validation_results: Dict[str, Any],
    ) -> None:
        """
        Run a test command on the executable.

        Args:
            tool_name: Name of the tool.
            exe_path: Path to the executable.
            test_cmd: Test command to run.
            tool_result: Result dictionary to update.
            validation_results: Overall results to update.
        """
        try:
            # On Windows, shell scripts need to be run through bash
            if sys.platform == "win32" and str(exe_path).endswith(".sh"):
                cmd = ["bash", str(exe_path), test_cmd]
            else:
                cmd = [str(exe_path), test_cmd]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if (
                result.returncode == 0
                or test_cmd in ("--help", "-h", "--version")
                or tool_name in ("taudem",)
            ):
                tool_result["status"] = "valid"
                tool_result["version"] = (
                    result.stdout.strip()[:100]
                    if result.stdout
                    else "Available"
                )
                validation_results["valid_tools"].append(tool_name)
                self._console.success(f"Found at: {exe_path}")
                self._console.success("Status: Working")
            else:
                tool_result["status"] = "failed"
                tool_result["errors"].append(
                    f"Test command failed: {result.stderr}"
                )
                validation_results["failed_tools"].append(tool_name)
                self._console.warning(f"Found but test failed: {exe_path}")
                self._console.warning(f"Error: {result.stderr[:100]}")

        except subprocess.TimeoutExpired:
            tool_result["status"] = "timeout"
            tool_result["errors"].append("Test command timed out")
            validation_results["warnings"].append(
                f"{tool_name}: test timed out"
            )
            self._console.warning(f"Found but test timed out: {exe_path}")
        except Exception as test_error:  # noqa: BLE001 — top-level fallback
            tool_result["status"] = "test_error"
            tool_result["errors"].append(f"Test error: {str(test_error)}")
            validation_results["warnings"].append(
                f"{tool_name}: {str(test_error)}"
            )
            self._console.warning(f"Found but couldn't test: {exe_path}")
            self._console.warning(f"Test error: {str(test_error)}")

    def _print_validation_summary(self, results: Dict[str, Any]) -> None:
        """
        Print validation summary.

        Args:
            results: Validation results dictionary.
        """
        total_tools = len(self.external_tools)
        skipped_count = len(results.get("skipped_tools", []))
        checked_count = total_tools - skipped_count
        valid_count = len(results["valid_tools"])
        missing_count = len(results["missing_tools"])
        failed_count = len(results["failed_tools"])

        self._console.newline()
        self._console.info("Binary Validation Summary:")
        self._console.indent(f"Valid: {valid_count}/{checked_count}")
        # Don't display skipped optional tools — they are on a need-to-know basis
        self._console.indent(f"Missing: {missing_count}/{checked_count}")
        self._console.indent(f"Failed: {failed_count}/{checked_count}")
