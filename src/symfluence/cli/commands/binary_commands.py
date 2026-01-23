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

        if tools:
            BaseCommand._console.info(f"Installing tools: {', '.join(tools)}")
        else:
            BaseCommand._console.info("Installing all available tools...")

        if force:
            BaseCommand._console.indent("(Force reinstall mode)")

        # Handle subprocess errors specifically
        try:
            success = binary_manager.get_executables(
                specific_tools=tools,
                force=force
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
