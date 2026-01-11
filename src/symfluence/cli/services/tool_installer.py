"""
Tool installation service for SYMFLUENCE.

Handles cloning repositories, running build commands, and verifying installations.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseService
from ..console import Console


class ToolInstaller(BaseService):
    """
    Service for installing external tools from source.

    Handles:
    - Repository cloning
    - Build command execution
    - Dependency resolution
    - Installation verification
    """

    def __init__(
        self,
        external_tools: Optional[Dict[str, Any]] = None,
        console: Optional[Console] = None,
    ):
        """
        Initialize the ToolInstaller.

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

    def install(
        self,
        specific_tools: Optional[List[str]] = None,
        symfluence_instance=None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Clone and install external tool repositories with dependency resolution.

        Args:
            specific_tools: List of specific tools to install. If None, installs all.
            symfluence_instance: Optional SYMFLUENCE instance with config.
            force: If True, reinstall even if already exists.
            dry_run: If True, only show what would be done.

        Returns:
            Dictionary with installation results.
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
        tools_to_install = self._resolve_dependencies(tools_to_install)

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

                # Clone repository or create directory
                if not self._clone_repository(
                    repository_url, branch, tool_install_dir
                ):
                    installation_results["failed"].append(tool_name)
                    continue

                # Check dependencies
                missing_deps = self._check_system_dependencies(
                    tool_info.get("dependencies", [])
                )
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

                # Check required tools
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
                    success = self._run_build_commands(
                        tool_name, tool_info, tool_install_dir
                    )
                    if success:
                        installation_results["successful"].append(tool_name)
                    else:
                        installation_results["failed"].append(tool_name)
                        installation_results["errors"].append(f"{tool_name} build failed")
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

    def _clone_repository(
        self,
        repository_url: Optional[str],
        branch: Optional[str],
        target_dir: Path,
    ) -> bool:
        """
        Clone a git repository.

        Args:
            repository_url: URL of the repository to clone.
            branch: Branch to checkout. If None, uses default branch.
            target_dir: Target directory for the clone.

        Returns:
            True if successful, False otherwise.
        """
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
                    str(target_dir),
                ]
            else:
                clone_cmd = ["git", "clone", repository_url, str(target_dir)]

            subprocess.run(clone_cmd, capture_output=True, text=True, check=True)
            self._console.success("Clone successful")
        else:
            self._console.indent("Creating installation directory")
            target_dir.mkdir(parents=True, exist_ok=True)
            self._console.success(f"Directory created: {target_dir}")

        return True

    def _run_build_commands(
        self,
        tool_name: str,
        tool_info: Dict[str, Any],
        install_dir: Path,
    ) -> bool:
        """
        Run build commands for a tool.

        Args:
            tool_name: Name of the tool being built.
            tool_info: Tool configuration dictionary.
            install_dir: Installation directory.

        Returns:
            True if build successful, False otherwise.
        """
        self._console.indent("Running build commands...")

        original_dir = os.getcwd()
        os.chdir(install_dir)

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
            return True

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
            return False

        finally:
            os.chdir(original_dir)

    def _verify_installation(
        self, tool_name: str, tool_info: Dict[str, Any], install_dir: Path
    ) -> bool:
        """
        Verify that a tool was installed correctly.

        Args:
            tool_name: Name of the tool.
            tool_info: Tool configuration dictionary.
            install_dir: Installation directory.

        Returns:
            True if verification passed, False otherwise.
        """
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

    def _resolve_dependencies(self, tools: List[str]) -> List[str]:
        """
        Resolve dependencies between tools and return sorted list.

        Args:
            tools: List of tool names to install.

        Returns:
            Sorted list with dependencies included.
        """
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

    def _check_system_dependencies(self, dependencies: List[str]) -> List[str]:
        """
        Check which system dependencies are missing.

        Args:
            dependencies: List of required system binaries.

        Returns:
            List of missing dependencies.
        """
        missing_deps = []
        for dep in dependencies:
            if not shutil.which(dep):
                missing_deps.append(dep)
        return missing_deps

    def _print_installation_summary(
        self, results: Dict[str, Any], dry_run: bool
    ) -> None:
        """
        Print installation summary.

        Args:
            results: Installation results dictionary.
            dry_run: Whether this was a dry run.
        """
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
