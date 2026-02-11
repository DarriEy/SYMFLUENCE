"""
Tool validation service for SYMFLUENCE.

Validates that required binary executables exist and are functional.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import BaseService
from ..console import Console


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
            "warnings": [],
            "summary": {},
        }

        config = self._load_config(symfluence_instance)

        # Validate each tool (skip optional tools that aren't installed)
        for tool_name, tool_info in self.external_tools.items():
            if tool_info.get('optional', False):
                # Check if optional tool is installed before validating
                data_dir = self._get_data_dir(config)
                opt_path = data_dir / tool_info.get("default_path_suffix", "")
                if not opt_path.exists():
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
                        f"Try: python SYMFLUENCE.py --get_executables {tool_name}"
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

            except Exception as e:
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
        candidates = [tool_path / p for p in verify.get("file_paths", [])]

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

        exe_path = tool_path / exe_name
        if exe_path.exists():
            return exe_path

        # Try without extension
        exe_name_no_ext = exe_name.replace(".exe", "")
        exe_path_no_ext = tool_path / exe_name_no_ext
        if exe_path_no_ext.exists():
            return exe_path_no_ext

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
            result = subprocess.run(
                [str(exe_path), test_cmd],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if (
                result.returncode == 0
                or test_cmd == "--help"
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
        except Exception as test_error:
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
        valid_count = len(results["valid_tools"])
        missing_count = len(results["missing_tools"])
        failed_count = len(results["failed_tools"])

        self._console.newline()
        self._console.info("Binary Validation Summary:")
        self._console.indent(f"Valid: {valid_count}/{total_tools}")
        self._console.indent(f"Missing: {missing_count}/{total_tools}")
        self._console.indent(f"Failed: {failed_count}/{total_tools}")
