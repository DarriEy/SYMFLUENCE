"""
Binary manager for external tool installation, validation, and execution.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .console import Console, console as global_console
from .external_tools_config import get_external_tools_definitions


class BinaryManager:
    """
    Manages external tool installation, validation, and execution.
    """

    def __init__(
        self,
        external_tools: Optional[Dict[str, Any]] = None,
        console: Optional[Console] = None,
    ):
        """
        Initialize the BinaryManager.

        Args:
            external_tools: Dictionary of tool definitions. If None, loads from config.
            console: Console instance for output. If None, uses global console.
        """
        self.external_tools = external_tools or get_external_tools_definitions()
        self._console = console or global_console

    def _load_config(self, symfluence_instance=None) -> Dict[str, Any]:
        """
        Load configuration from SYMFLUENCE instance or fall back to template.

        Args:
            symfluence_instance: Optional SYMFLUENCE instance with config attribute.

        Returns:
            Configuration dictionary.
        """
        if symfluence_instance and hasattr(symfluence_instance, "config"):
            return symfluence_instance.config
        if symfluence_instance and hasattr(symfluence_instance, "workflow_orchestrator"):
            return symfluence_instance.workflow_orchestrator.config

        try:
            from symfluence.resources import get_config_template

            config_path = get_config_template()
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return self._ensure_valid_config_paths(config, config_path)
        except (ImportError, FileNotFoundError, yaml.YAMLError) as e:
            self._console.debug(f"Could not load config: {e}")
            return {}

    def _get_data_dir(self, config: Dict[str, Any]) -> Path:
        """
        Get SYMFLUENCE data directory from environment or config.

        Args:
            config: Configuration dictionary.

        Returns:
            Path to data directory.
        """
        data_dir = os.getenv("SYMFLUENCE_DATA") or config.get("SYMFLUENCE_DATA_DIR", ".")
        return Path(data_dir)

    def get_executables(
        self,
        specific_tools: List[str] = None,
        symfluence_instance=None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Clone and install external tool repositories with dependency resolution.
        """
        action = "Planning" if dry_run else "Installing"
        self._console.panel(f"{action} External Tools", style="blue")

        if dry_run:
            self._console.info("[DRY RUN] No actual installation will occur")
            self._console.rule()

        installation_results = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "errors": [],
            "dry_run": dry_run,
        }

        config = self._load_config(symfluence_instance)
        install_base_dir = self._get_data_dir(config) / "installs"

        self._console.info(f"Installation directory: {install_base_dir}")

        if not dry_run:
            install_base_dir.mkdir(parents=True, exist_ok=True)

        # Determine which tools to install
        if specific_tools is None:
            tools_to_install = list(self.external_tools.keys())
        else:
            tools_to_install = []
            for tool in specific_tools:
                if tool in self.external_tools:
                    tools_to_install.append(tool)
                else:
                    self._console.warning(f"Unknown tool: {tool}")
                    installation_results["errors"].append(f"Unknown tool: {tool}")

        # Resolve dependencies and sort by install order
        tools_to_install = self._resolve_tool_dependencies(tools_to_install)

        self._console.info(f"Installing tools in order: {', '.join(tools_to_install)}")

        # Install each tool
        for tool_name in tools_to_install:
            tool_info = self.external_tools[tool_name]
            self._console.newline()
            self._console.info(f"[bold]{action} {tool_name.upper()}:[/bold]")
            self._console.indent(tool_info.get("description", ""))

            tool_install_dir = install_base_dir / tool_info.get("install_dir", tool_name)
            repository_url = tool_info.get("repository")
            branch = tool_info.get("branch")

            try:
                # Check if already exists
                if tool_install_dir.exists() and not force:
                    self._console.indent(f"Skipping - already exists at: {tool_install_dir}")
                    self._console.indent("Use --force_install to reinstall")
                    installation_results["skipped"].append(tool_name)
                    continue

                if dry_run:
                    self._console.indent(f"Would clone: {repository_url}")
                    if branch:
                        self._console.indent(f"Would checkout branch: {branch}")
                    self._console.indent(f"Target directory: {tool_install_dir}")
                    self._console.indent("Would run build commands:")
                    for cmd in tool_info.get("build_commands", []):
                        self._console.indent(f"  {cmd[:100]}...", level=2)
                    installation_results["successful"].append(f"{tool_name} (dry run)")
                    continue

                # Remove existing if force reinstall
                if tool_install_dir.exists() and force:
                    self._console.indent(f"Removing existing installation: {tool_install_dir}")
                    shutil.rmtree(tool_install_dir)

                # Clone repository or create directory for non-git tools
                if repository_url:
                    self._console.indent(f"Cloning from: {repository_url}")
                    if branch:
                        self._console.indent(f"Checking out branch: {branch}")
                        clone_cmd = [
                            "git",
                            "clone",
                            "-b",
                            branch,
                            repository_url,
                            str(tool_install_dir),
                        ]
                    else:
                        clone_cmd = ["git", "clone", repository_url, str(tool_install_dir)]

                    subprocess.run(clone_cmd, capture_output=True, text=True, check=True)
                    self._console.success("Clone successful")
                else:
                    self._console.indent("Creating installation directory")
                    tool_install_dir.mkdir(parents=True, exist_ok=True)
                    self._console.success(f"Directory created: {tool_install_dir}")

                # Check dependencies
                missing_deps = self._check_dependencies(tool_info.get("dependencies", []))
                if missing_deps:
                    self._console.warning(
                        f"Missing system dependencies: {', '.join(missing_deps)}"
                    )
                    self._console.indent(
                        "These may be available as modules - check with 'module avail'"
                    )
                    installation_results["errors"].append(
                        f"{tool_name}: missing system dependencies {missing_deps}"
                    )

                if tool_info.get("requires"):
                    required_tools = tool_info.get("requires", [])
                    for req_tool in required_tools:
                        req_tool_info = self.external_tools.get(req_tool, {})
                        req_tool_dir = install_base_dir / req_tool_info.get(
                            "install_dir", req_tool
                        )
                        if not req_tool_dir.exists():
                            error_msg = (
                                f"{tool_name} requires {req_tool} but it's not installed"
                            )
                            self._console.error(error_msg)
                            installation_results["errors"].append(error_msg)
                            installation_results["failed"].append(tool_name)
                            continue

                # Run build commands
                if tool_info.get("build_commands"):
                    self._console.indent("Running build commands...")

                    original_dir = os.getcwd()
                    os.chdir(tool_install_dir)

                    try:
                        combined_script = "\n".join(tool_info.get("build_commands", []))

                        build_result = subprocess.run(
                            combined_script,
                            shell=True,
                            check=True,
                            capture_output=True,
                            text=True,
                            executable="/bin/bash",
                        )

                        # Show output for critical tools
                        if tool_name in ["summa", "sundials", "mizuroute", "fuse", "ngen"]:
                            if build_result.stdout:
                                self._console.indent("=== Build Output ===", level=2)
                                for line in build_result.stdout.strip().split("\n"):
                                    self._console.indent(line, level=3)
                        else:
                            if build_result.stdout:
                                lines = build_result.stdout.strip().split("\n")
                                for line in lines[-10:]:
                                    self._console.indent(line, level=3)

                        self._console.success("Build successful")
                        installation_results["successful"].append(tool_name)

                    except subprocess.CalledProcessError as build_error:
                        self._console.error(f"Build failed: {build_error}")
                        if build_error.stdout:
                            self._console.indent("=== Build Output ===", level=2)
                            for line in build_error.stdout.strip().split("\n"):
                                self._console.indent(line, level=3)
                        if build_error.stderr:
                            self._console.indent("=== Error Output ===", level=2)
                            for line in build_error.stderr.strip().split("\n"):
                                self._console.indent(line, level=3)
                        installation_results["failed"].append(tool_name)
                        installation_results["errors"].append(f"{tool_name} build failed")

                    finally:
                        os.chdir(original_dir)
                else:
                    self._console.success("No build required")
                    installation_results["successful"].append(tool_name)

                # Verify installation
                self._verify_installation(tool_name, tool_info, tool_install_dir)

            except subprocess.CalledProcessError as e:
                if repository_url:
                    error_msg = f"Failed to clone {repository_url}: {e.stderr if e.stderr else str(e)}"
                else:
                    error_msg = f"Failed during installation: {e.stderr if e.stderr else str(e)}"
                self._console.error(error_msg)
                installation_results["failed"].append(tool_name)
                installation_results["errors"].append(f"{tool_name}: {error_msg}")

            except Exception as e:
                error_msg = f"Installation error: {str(e)}"
                self._console.error(error_msg)
                installation_results["failed"].append(tool_name)
                installation_results["errors"].append(f"{tool_name}: {error_msg}")

        # Print summary
        self._print_installation_summary(installation_results, dry_run)

        return installation_results

    def validate_binaries(
        self, symfluence_instance=None, verbose: bool = False
    ) -> Any:
        """
        Validate that required binary executables exist and are functional.
        """
        self._console.panel("Validating External Tool Binaries", style="blue")

        validation_results = {
            "valid_tools": [],
            "missing_tools": [],
            "failed_tools": [],
            "warnings": [],
            "summary": {},
        }

        config = self._load_config(symfluence_instance)

        # Validate each tool
        for tool_name, tool_info in self.external_tools.items():
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
                    data_dir = config.get("SYMFLUENCE_DATA_DIR", ".")
                    tool_path = Path(data_dir) / tool_info.get("default_path_suffix", "")
                else:
                    tool_path = Path(tool_path)
                tool_result["path"] = str(tool_path)

                # If tool defines a verify_install block, honor it first
                verify = tool_info.get("verify_install")
                found_path = None
                if verify and isinstance(verify, dict):
                    check_type = verify.get("check_type", "exists_all")
                    candidates = [tool_path / p for p in verify.get("file_paths", [])]

                    if check_type == "exists_any":
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

                    if exists_ok:
                        test_cmd = tool_info.get("test_command")
                        if test_cmd is None:
                            tool_result["status"] = "valid"
                            tool_result["version"] = "Installed (existence verified)"
                            validation_results["valid_tools"].append(tool_name)
                            self._console.success(f"Found at: {found_path}")
                            self._console.success("Status: Installed")
                            validation_results["summary"][tool_name] = tool_result
                            continue

                # Fallback: single-executable check
                config_exe_key = tool_info.get("config_exe_key")
                if config_exe_key and config_exe_key in config:
                    exe_name = config[config_exe_key]
                else:
                    exe_name = tool_info.get("default_exe", "")

                # Handle shared library extension on macOS
                if exe_name.endswith(".so") and sys.platform == "darwin":
                    exe_name = exe_name.replace(".so", ".dylib")

                tool_result["executable"] = exe_name

                exe_path = tool_path / exe_name
                if not exe_path.exists():
                    exe_name_no_ext = exe_name.replace(".exe", "")
                    exe_path_no_ext = tool_path / exe_name_no_ext
                    if exe_path_no_ext.exists():
                        exe_path = exe_path_no_ext
                        tool_result["executable"] = exe_name_no_ext

                if exe_path.exists():
                    test_cmd = tool_info.get("test_command")
                    if test_cmd is None:
                        tool_result["status"] = "valid"
                        tool_result["version"] = "Installed (existence verified)"
                        validation_results["valid_tools"].append(tool_name)
                        self._console.success(f"Found at: {exe_path}")
                        self._console.success("Status: Installed")
                    else:
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

                else:
                    tool_result["status"] = "missing"
                    tool_result["errors"].append(f"Executable not found at: {exe_path}")
                    validation_results["missing_tools"].append(tool_name)
                    self._console.error(f"Not found: {exe_path}")
                    self._console.indent(
                        f"Try: python SYMFLUENCE.py --get_executables {tool_name}"
                    )

            except Exception as e:
                tool_result["status"] = "error"
                tool_result["errors"].append(f"Validation error: {str(e)}")
                validation_results["failed_tools"].append(tool_name)
                self._console.error(f"Validation error: {str(e)}")

            validation_results["summary"][tool_name] = tool_result

        # Print summary
        total_tools = len(self.external_tools)
        valid_count = len(validation_results["valid_tools"])
        missing_count = len(validation_results["missing_tools"])
        failed_count = len(validation_results["failed_tools"])

        self._console.newline()
        self._console.info("Binary Validation Summary:")
        self._console.indent(f"Valid: {valid_count}/{total_tools}")
        self._console.indent(f"Missing: {missing_count}/{total_tools}")
        self._console.indent(f"Failed: {failed_count}/{total_tools}")

        if (
            len(validation_results["missing_tools"]) == 0
            and len(validation_results["failed_tools"]) == 0
        ):
            return True
        else:
            return validation_results

    def handle_binary_management(self, execution_plan: Dict[str, Any]) -> bool:
        """
        Legacy dispatcher for binary management operations.
        """
        ops = execution_plan.get("binary_operations", {})

        if ops.get("doctor"):
            self.run_doctor()
            return True
        if ops.get("tools_info"):
            self.show_tools_info()
            return True
        if ops.get("validate_binaries"):
            return self.validate_binaries() is True
        if ops.get("get_executables"):
            tools = ops.get("get_executables")
            if isinstance(tools, bool):
                tools = None
            result = self.get_executables(specific_tools=tools)
            return len(result.get("failed", [])) == 0

        return False

    def run_doctor(self) -> bool:
        """Alias for doctor() for backward compatibility."""
        return self.doctor()

    def show_tools_info(self) -> bool:
        """Alias for tools_info() for backward compatibility."""
        return self.tools_info()

    def _check_dependencies(self, dependencies: List[str]) -> List[str]:
        """Check which dependencies are missing from the system."""
        missing_deps = []
        for dep in dependencies:
            if not shutil.which(dep):
                missing_deps.append(dep)
        return missing_deps

    def _verify_installation(
        self, tool_name: str, tool_info: Dict[str, Any], install_dir: Path
    ) -> bool:
        """Verify that a tool was installed correctly."""
        try:
            verify = tool_info.get("verify_install")
            if verify and isinstance(verify, dict):
                check_type = verify.get("check_type", "exists_all")
                candidates = [install_dir / p for p in verify.get("file_paths", [])]

                if check_type == "exists_any":
                    ok = any(p.exists() for p in candidates)
                elif check_type in ("exists_all", "exists"):
                    ok = all(p.exists() for p in candidates)
                else:
                    ok = False

                status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
                self._console.indent(f"Install verification ({check_type}): {status}")
                for p in candidates:
                    check = "[green]Y[/green]" if p.exists() else "[red]N[/red]"
                    self._console.indent(f"  {check} {p}", level=2)
                return ok

            exe_name = tool_info.get("default_exe")
            if not exe_name:
                return False

            possible_paths = [
                install_dir / exe_name,
                install_dir / "bin" / exe_name,
                install_dir / "build" / exe_name,
                install_dir / "route" / "bin" / exe_name,
                install_dir / exe_name.replace(".exe", ""),
                install_dir / "install" / "sundials" / exe_name,
            ]

            for exe_path in possible_paths:
                if exe_path.exists():
                    self._console.success(f"Executable/library found: {exe_path}")
                    return True

            return False

        except Exception as e:
            self._console.warning(f"Verification error: {str(e)}")
            return False

    def _resolve_tool_dependencies(self, tools: List[str]) -> List[str]:
        """Resolve dependencies between tools and return sorted list."""
        tools_with_deps = set(tools)
        for tool in tools:
            if tool in self.external_tools and self.external_tools.get(tool, {}).get(
                "requires"
            ):
                required = self.external_tools.get(tool, {}).get("requires", [])
                tools_with_deps.update(required)

        return sorted(
            tools_with_deps,
            key=lambda t: (self.external_tools.get(t, {}).get("order", 999), t),
        )

    def _print_installation_summary(
        self, results: Dict[str, Any], dry_run: bool
    ) -> None:
        """Print installation summary."""
        successful_count = len(results["successful"])
        failed_count = len(results["failed"])
        skipped_count = len(results["skipped"])

        self._console.newline()
        self._console.info("Installation Summary:")
        if dry_run:
            self._console.indent(f"Would install: {successful_count} tools")
            self._console.indent(f"Would skip: {skipped_count} tools")
        else:
            self._console.indent(f"Successful: {successful_count} tools")
            self._console.indent(f"Failed: {failed_count} tools")
            self._console.indent(f"Skipped: {skipped_count} tools")

        if results["errors"]:
            self._console.newline()
            self._console.error("Errors encountered:")
            for error in results["errors"]:
                self._console.indent(f"- {error}")

    def _ensure_valid_config_paths(
        self, config: Dict[str, Any], config_path: Path
    ) -> Dict[str, Any]:
        """
        Ensure SYMFLUENCE_DATA_DIR and SYMFLUENCE_CODE_DIR paths exist and are valid.
        """
        data_dir = config.get("SYMFLUENCE_DATA_DIR")
        code_dir = config.get("SYMFLUENCE_CODE_DIR")

        data_dir_valid = False
        code_dir_valid = False

        if data_dir:
            try:
                data_path = Path(data_dir)
                if data_path.exists():
                    test_file = data_path / ".symfluence_test"
                    try:
                        test_file.touch()
                        test_file.unlink()
                        data_dir_valid = True
                    except (PermissionError, OSError):
                        pass
                else:
                    try:
                        data_path.mkdir(parents=True, exist_ok=True)
                        data_dir_valid = True
                    except (PermissionError, OSError):
                        pass
            except Exception:
                pass

        if code_dir:
            try:
                code_path = Path(code_dir)
                if code_path.exists() and os.access(code_path, os.R_OK):
                    code_dir_valid = True
            except Exception:
                pass

        if not data_dir_valid or not code_dir_valid:
            self._console.warning(
                "Detected invalid or inaccessible paths in config template:"
            )

            if not code_dir_valid:
                new_code_dir = Path.cwd().resolve()
                config["SYMFLUENCE_CODE_DIR"] = str(new_code_dir)
                self._console.success(f"SYMFLUENCE_CODE_DIR set to: {new_code_dir}")

            if not data_dir_valid:
                new_data_dir = (Path.cwd().parent / "SYMFLUENCE_data").resolve()
                config["SYMFLUENCE_DATA_DIR"] = str(new_data_dir)
                try:
                    new_data_dir.mkdir(parents=True, exist_ok=True)
                    self._console.success(f"SYMFLUENCE_DATA_DIR set to: {new_data_dir}")
                except Exception:
                    pass

            try:
                backup_path = config_path.with_name(
                    f"{config_path.stem}_backup{config_path.suffix}"
                )
                if config_path.exists():
                    shutil.copy2(config_path, backup_path)
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except Exception:
                pass

        return config

    def detect_npm_binaries(self) -> Optional[Path]:
        """
        Detect if SYMFLUENCE binaries are installed via npm.

        Returns:
            Path to npm-installed binaries, or None if not found
        """
        try:
            result = subprocess.run(
                ["npm", "root", "-g"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                npm_root = Path(result.stdout.strip())
                npm_bin_dir = npm_root / "symfluence" / "dist" / "bin"

                if npm_bin_dir.exists() and npm_bin_dir.is_dir():
                    return npm_bin_dir

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        return None

    def doctor(self) -> bool:
        """
        Run system diagnostics: check binaries, toolchain, and system libraries.
        """
        self._console.rule()

        # Check binaries
        self._console.newline()
        self._console.info("Checking binaries...")
        self._console.rule()

        config = self._load_config()
        symfluence_data = str(self._get_data_dir(config))

        npm_bin_dir = self.detect_npm_binaries()

        if npm_bin_dir:
            self._console.info(f"Detected npm-installed binaries: {npm_bin_dir}")
        if symfluence_data:
            self._console.info(f"Checking source installs in: {symfluence_data}")

        found_binaries = 0
        total_binaries = 0

        # Build table rows for binary status
        binary_rows = []

        for name, tool_info in self.external_tools.items():
            if name == "sundials":
                continue  # Skip library-only tool
            total_binaries += 1

            found = False
            location = None

            # 1. Check in SYMFLUENCE_DATA (installed from source)
            if symfluence_data:
                rel_path_suffix = tool_info.get("default_path_suffix", "")
                exe_name = tool_info.get("default_exe", "")

                full_path = Path(symfluence_data) / rel_path_suffix

                if name in ("taudem",):
                    if full_path.exists() and full_path.is_dir():
                        found = True
                        location = full_path
                elif exe_name:
                    if exe_name.endswith(".so") and sys.platform == "darwin":
                        exe_name_mac = exe_name.replace(".so", ".dylib")
                        candidates = [exe_name, exe_name_mac]
                    else:
                        candidates = [exe_name]

                    for cand in candidates:
                        exe_path = full_path / cand
                        if exe_path.exists():
                            found = True
                            location = exe_path
                            break

                        exe_path_no_ext = full_path / cand.replace(".exe", "")
                        if exe_path_no_ext.exists():
                            found = True
                            location = exe_path_no_ext
                            break

            # 2. Check npm installation as fallback
            if not found and npm_bin_dir:
                npm_path = npm_bin_dir / name
                if npm_path.exists():
                    found = True
                    location = npm_path
                else:
                    exe_name = tool_info.get("default_exe", "")
                    if exe_name:
                        for candidate in [exe_name, exe_name.replace(".exe", "")]:
                            npm_exe_path = npm_bin_dir / candidate
                            if npm_exe_path.exists():
                                found = True
                                location = npm_exe_path
                                break

            status = "[green]OK[/green]" if found else "[red]MISSING[/red]"
            loc_str = str(location) if location else "-"
            binary_rows.append([name, status, loc_str])

            if found:
                found_binaries += 1

        self._console.table(
            columns=["Tool", "Status", "Location"],
            rows=binary_rows,
            title="Binary Status",
        )

        # Check toolchain metadata
        self._console.newline()
        self._console.info("Toolchain metadata...")
        self._console.rule()

        toolchain_locations = []
        if symfluence_data:
            toolchain_locations.append(
                Path(symfluence_data) / "installs" / "toolchain.json"
            )
        if npm_bin_dir:
            toolchain_locations.append(npm_bin_dir.parent / "toolchain.json")

        toolchain_found = False
        for toolchain_path in toolchain_locations:
            if toolchain_path.exists():
                try:
                    import json

                    with open(toolchain_path) as f:
                        toolchain = json.load(f)

                    platform = toolchain.get("platform", "unknown")
                    build_date = toolchain.get("build_date", "unknown")
                    fortran = toolchain.get("compilers", {}).get("fortran", "unknown")

                    self._console.success(f"Found: {toolchain_path}")
                    self._console.indent(f"Platform: {platform}")
                    self._console.indent(f"Build date: {build_date}")
                    self._console.indent(f"Fortran: {fortran}")
                    toolchain_found = True
                    break
                except Exception as e:
                    self._console.warning(f"Error reading {toolchain_path}: {e}")

        if not toolchain_found:
            self._console.error("No toolchain metadata found")

        # Check system libraries
        self._console.newline()
        self._console.info("System libraries...")
        self._console.rule()

        system_tools = {
            "nc-config": "NetCDF",
            "nf-config": "NetCDF-Fortran",
            "h5cc": "HDF5",
            "gdal-config": "GDAL",
            "mpirun": "MPI",
        }

        lib_rows = []
        found_libs = 0
        for tool, name in system_tools.items():
            location = shutil.which(tool)
            if location:
                lib_rows.append([name, "[green]OK[/green]", location])
                found_libs += 1
            else:
                lib_rows.append([name, "[red]MISSING[/red]", "-"])

        self._console.table(
            columns=["Library", "Status", "Location"],
            rows=lib_rows,
            title="System Libraries",
        )

        # Summary
        self._console.newline()
        self._console.rule()
        self._console.info("Summary:")
        self._console.indent(f"Binaries: {found_binaries}/{total_binaries} found")
        tc_status = "[green]Found[/green]" if toolchain_found else "[red]Not found[/red]"
        self._console.indent(f"Toolchain metadata: {tc_status}")
        self._console.indent(f"System libraries: {found_libs}/{len(system_tools)} found")

        if found_binaries == total_binaries and toolchain_found and found_libs >= 3:
            self._console.newline()
            self._console.success("System is ready for SYMFLUENCE!")
        elif found_binaries == 0:
            self._console.newline()
            self._console.warning("No binaries found. Install with:")
            self._console.indent("npm install -g symfluence (for pre-built binaries)")
            self._console.indent("./symfluence --get_executables (to build from source)")
        else:
            self._console.newline()
            self._console.warning("Some components missing. Review output above.")

        self._console.rule()
        return True

    def tools_info(self) -> bool:
        """
        Display installed tools information from toolchain metadata.
        """
        symfluence_data = os.getenv("SYMFLUENCE_DATA")
        npm_bin_dir = self.detect_npm_binaries()

        toolchain_locations = []
        if symfluence_data:
            toolchain_locations.append(
                Path(symfluence_data) / "installs" / "toolchain.json"
            )
        if npm_bin_dir:
            toolchain_locations.append(npm_bin_dir.parent / "toolchain.json")

        toolchain_path = None
        for path in toolchain_locations:
            if path.exists():
                toolchain_path = path
                break

        if not toolchain_path:
            self._console.error("No toolchain metadata found.")
            self._console.newline()
            self._console.info("Toolchain metadata is generated during installation.")
            self._console.indent("Install binaries with:")
            self._console.indent("  npm install -g symfluence")
            self._console.indent("  ./symfluence --get_executables")
            return False

        try:
            import json

            with open(toolchain_path) as f:
                toolchain = json.load(f)

            self._console.rule()
            self._console.info(f"Platform: {toolchain.get('platform', 'unknown')}")
            self._console.info(f"Build Date: {toolchain.get('build_date', 'unknown')}")
            self._console.info(f"Toolchain file: {toolchain_path}")

            # Compilers
            if "compilers" in toolchain:
                self._console.newline()
                self._console.info("Compilers:")
                self._console.rule()
                compilers = toolchain["compilers"]
                compiler_rows = [
                    [key.capitalize(), value] for key, value in compilers.items()
                ]
                self._console.table(
                    columns=["Compiler", "Version"], rows=compiler_rows
                )

            # Libraries
            if "libraries" in toolchain:
                self._console.newline()
                self._console.info("Libraries:")
                self._console.rule()
                libraries = toolchain["libraries"]
                lib_rows = [[key.capitalize(), value] for key, value in libraries.items()]
                self._console.table(columns=["Library", "Version"], rows=lib_rows)

            # Tools
            if "tools" in toolchain:
                self._console.newline()
                self._console.info("Installed Tools:")
                self._console.rule()
                for tool_name, tool_info in toolchain["tools"].items():
                    self._console.newline()
                    self._console.info(f"  {tool_name.upper()}:")
                    if "commit" in tool_info:
                        commit_short = (
                            tool_info.get("commit", "")[:8]
                            if len(tool_info.get("commit", "")) > 8
                            else tool_info.get("commit", "")
                        )
                        self._console.indent(f"Commit: {commit_short}", level=2)
                    if "branch" in tool_info:
                        self._console.indent(
                            f"Branch: {tool_info.get('branch', '')}", level=2
                        )
                    if "executable" in tool_info:
                        self._console.indent(
                            f"Executable: {tool_info.get('executable', '')}", level=2
                        )

            self._console.newline()
            self._console.rule()
            return True

        except Exception as e:
            self._console.error(f"Error reading toolchain file: {e}")
            return False
