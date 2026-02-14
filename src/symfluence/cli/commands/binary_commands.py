"""
Binary/tool management command handlers for SYMFLUENCE CLI.

This module implements handlers for external tool installation and validation.
"""

from argparse import Namespace
import subprocess

from .base import BaseCommand, cli_exception_handler
from ..exit_codes import ExitCode


class BinaryCommands(BaseCommand):
    """Handlers for binary/tool management commands."""

    @staticmethod
    @cli_exception_handler
    def install(args: Namespace) -> int:
        """
        Execute: symfluence binary install [TOOL1 TOOL2 ...]

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.binary_service import BinaryManager

        binary_manager = BinaryManager()

        # Get tools to install
        tools = args.tools if args.tools else None  # None means install all
        force = args.force
        patched = getattr(args, 'patched', False)

        if tools:
            BaseCommand._console.info(f"Installing tools: {', '.join(tools)}")
        else:
            BaseCommand._console.info("Installing all available tools...")

        if force:
            BaseCommand._console.indent("(Force reinstall mode)")

        if patched:
            BaseCommand._console.indent("(SYMFLUENCE patches enabled)")

        # Handle subprocess errors specifically
        try:
            success = binary_manager.get_executables(
                specific_tools=tools,
                force=force,
                patched=patched
            )
        except subprocess.CalledProcessError as e:
            BaseCommand._console.error(f"Build command failed: {e}")
            if BaseCommand.get_arg(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.BINARY_BUILD_ERROR

        if success:
            BaseCommand._console.success("Tool installation completed successfully")
            return ExitCode.SUCCESS
        else:
            BaseCommand._console.error("Tool installation failed or was incomplete")
            return ExitCode.BINARY_ERROR

    @staticmethod
    @cli_exception_handler
    def validate(args: Namespace) -> int:
        """
        Execute: symfluence binary validate

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.binary_service import BinaryManager

        binary_manager = BinaryManager()

        verbose = BaseCommand.get_arg(args, 'verbose', False)

        BaseCommand._console.info("Validating installed binaries...")

        # Handle subprocess errors specifically
        try:
            success = binary_manager.validate_binaries(verbose=verbose)
        except subprocess.CalledProcessError as e:
            BaseCommand._console.error(f"Binary test command failed: {e}")
            return ExitCode.BINARY_ERROR

        if success:
            BaseCommand._console.success("All binaries validated successfully")
            return ExitCode.SUCCESS
        else:
            BaseCommand._console.error("Binary validation failed")
            return ExitCode.BINARY_ERROR

    @staticmethod
    @cli_exception_handler
    def doctor(args: Namespace) -> int:
        """
        Execute: symfluence binary doctor

        Run system diagnostics to check environment and dependencies.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.binary_service import BinaryManager

        binary_manager = BinaryManager()

        BaseCommand._console.info("Running system diagnostics...")
        BaseCommand._console.rule()

        # Call doctor function from binary manager
        success = binary_manager.doctor()

        if success:
            BaseCommand._console.rule()
            BaseCommand._console.success("System diagnostics completed")
            return ExitCode.SUCCESS
        else:
            BaseCommand._console.rule()
            BaseCommand._console.error("System diagnostics found issues")
            return ExitCode.DEPENDENCY_ERROR

    @staticmethod
    @cli_exception_handler
    def install_sysdeps(args: Namespace) -> int:
        """
        Execute: symfluence binary install-sysdeps

        Install platform-appropriate system dependencies using the detected
        package manager.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.services.system_deps import SystemDepsRegistry

        registry = SystemDepsRegistry()
        tool = getattr(args, 'tool', None)
        dry_run = getattr(args, 'dry_run', False)

        BaseCommand._console.info(
            f"Detected platform: {registry.platform.value}"
        )

        if tool:
            results = registry.check_deps_for_tool(tool)
            if not results:
                BaseCommand._console.error(f"Unknown tool: {tool}")
                return ExitCode.GENERAL_ERROR
        else:
            results = registry.check_all_deps()

        missing = [r for r in results if not r.found]
        if not missing:
            BaseCommand._console.success("All system dependencies are already installed!")
            return ExitCode.SUCCESS

        BaseCommand._console.warning(
            f"Missing {len(missing)} dependencies: "
            + ", ".join(r.display_name for r in missing)
        )

        script = registry.generate_install_script(tool_name=tool)
        if not script:
            BaseCommand._console.error(
                "Could not generate install commands for your platform. "
                "See docs/SYSTEM_REQUIREMENTS.md for manual instructions."
            )
            return ExitCode.DEPENDENCY_ERROR

        BaseCommand._console.newline()
        BaseCommand._console.info("Install commands:")
        BaseCommand._console.rule()
        for line in script.strip().splitlines():
            if line and not line.startswith("#") and not line.startswith("set"):
                BaseCommand._console.indent(line)
        BaseCommand._console.rule()

        if dry_run:
            BaseCommand._console.info(
                "[DRY RUN] Commands printed above but not executed."
            )
            return ExitCode.SUCCESS

        BaseCommand._console.newline()
        BaseCommand._console.info("Running install commands...")

        try:
            from symfluence.cli.services.system_deps import Platform

            platform = registry.platform

            if platform in (Platform.CONDA, Platform.WSL):
                # Conda install runs natively (no bash wrapper needed).
                # WSL script contains `wsl -e ...` — run directly on Windows.
                # Script is generated internally, not from user input.
                result = subprocess.run(
                    script, shell=True, text=True, timeout=600,  # nosec B602
                )
            elif platform == Platform.MSYS2:
                # MSYS2 has its own bash — run pacman script through it
                from symfluence.cli.services.tool_installer import ToolInstaller
                bash = ToolInstaller._find_bash()
                if bash:
                    result = subprocess.run(
                        [bash, "-c", script], text=True, timeout=600,
                    )
                else:
                    result = subprocess.run(
                        script, shell=True, text=True, timeout=600,  # nosec B602
                    )
            elif platform == Platform.UNKNOWN:
                # Don't attempt execution — just show commands
                BaseCommand._console.warning(
                    "Unknown platform. Commands printed above but not executed."
                )
                return ExitCode.SUCCESS
            else:
                # APT, DNF, BREW, HPC_MODULE — use bash
                from symfluence.cli.services.tool_installer import ToolInstaller
                bash = ToolInstaller._find_bash() or "bash"
                result = subprocess.run(
                    [bash, "-c", script], text=True, timeout=600,
                )

            if result.returncode == 0:
                BaseCommand._console.success(
                    "System dependencies installed successfully"
                )
                return ExitCode.SUCCESS
            else:
                BaseCommand._console.error(
                    "Some packages failed to install. "
                    "Check the output above and retry manually."
                )
                return ExitCode.DEPENDENCY_ERROR
        except subprocess.TimeoutExpired:
            BaseCommand._console.error("Installation timed out after 10 minutes")
            return ExitCode.GENERAL_ERROR

    @staticmethod
    @cli_exception_handler
    def info(args: Namespace) -> int:
        """
        Execute: symfluence binary info

        Display information about installed tools.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.binary_service import BinaryManager

        binary_manager = BinaryManager()

        BaseCommand._console.info("Installed Tools Information:")
        BaseCommand._console.rule()

        # Call info function from binary manager
        success = binary_manager.tools_info()

        if success:
            BaseCommand._console.rule()
            return ExitCode.SUCCESS
        else:
            BaseCommand._console.error("Failed to retrieve tools information")
            return ExitCode.GENERAL_ERROR
