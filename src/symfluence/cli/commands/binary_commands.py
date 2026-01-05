"""
Binary/tool management command handlers for SYMFLUENCE CLI.

This module implements handlers for external tool installation and validation.
"""

import sys
from argparse import Namespace
from typing import List, Optional

from .base import BaseCommand


class BinaryCommands(BaseCommand):
    """Handlers for binary/tool management commands."""

    @staticmethod
    def install(args: Namespace) -> int:
        """
        Execute: symfluence binary install [TOOL1 TOOL2 ...]

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.cli.binary_manager import BinaryManager

            binary_manager = BinaryManager()

            # Get tools to install
            tools = args.tools if args.tools else None  # None means install all
            force = args.force

            # Validate tool names
            if tools:
                from symfluence.cli.argument_parser import EXTERNAL_TOOLS
                invalid_tools = [t for t in tools if t not in EXTERNAL_TOOLS]
                if invalid_tools:
                    BaseCommand.print_error(f"Invalid tool names: {', '.join(invalid_tools)}")
                    BaseCommand.print_info(f"Available tools: {', '.join(EXTERNAL_TOOLS)}")
                    return 1

            if tools:
                BaseCommand.print_info(f"📦 Installing tools: {', '.join(tools)}")
            else:
                BaseCommand.print_info("📦 Installing all available tools...")

            if force:
                BaseCommand.print_info("   (Force reinstall mode)")

            # Call binary manager to install
            success = binary_manager.get_executables(
                specific_tools=tools,
                force=force
            )

            if success:
                BaseCommand.print_success("Tool installation completed successfully")
                return 0
            else:
                BaseCommand.print_error("Tool installation failed or was incomplete")
                return 1

        except Exception as e:
            BaseCommand.print_error(f"Installation failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def validate(args: Namespace) -> int:
        """
        Execute: symfluence binary validate

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.cli.binary_manager import BinaryManager

            binary_manager = BinaryManager()

            BaseCommand.print_info("🔍 Validating installed binaries...")

            # Call binary manager validation
            success = binary_manager.validate_binaries()

            if success:
                BaseCommand.print_success("All binaries validated successfully")
                return 0
            else:
                BaseCommand.print_error("Binary validation failed")
                return 1

        except Exception as e:
            BaseCommand.print_error(f"Validation failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def doctor(args: Namespace) -> int:
        """
        Execute: symfluence binary doctor

        Run system diagnostics to check environment and dependencies.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.cli.binary_manager import BinaryManager

            binary_manager = BinaryManager()

            BaseCommand.print_info("🏥 Running system diagnostics...")
            BaseCommand.print_info("=" * 70)

            # Call doctor function from binary manager
            success = binary_manager.doctor()

            if success:
                BaseCommand.print_info("=" * 70)
                BaseCommand.print_success("System diagnostics completed")
                return 0
            else:
                BaseCommand.print_info("=" * 70)
                BaseCommand.print_error("System diagnostics found issues")
                return 1

        except Exception as e:
            BaseCommand.print_error(f"Diagnostics failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def info(args: Namespace) -> int:
        """
        Execute: symfluence binary info

        Display information about installed tools.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.cli.binary_manager import BinaryManager

            binary_manager = BinaryManager()

            BaseCommand.print_info("📋 Installed Tools Information:")
            BaseCommand.print_info("=" * 70)

            # Call info function from binary manager
            success = binary_manager.tools_info()

            if success:
                BaseCommand.print_info("=" * 70)
                return 0
            else:
                BaseCommand.print_error("Failed to retrieve tools information")
                return 1

        except Exception as e:
            BaseCommand.print_error(f"Failed to get tools info: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def execute(args: Namespace) -> int:
        """
        Main execution dispatcher (required by BaseCommand).

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code
        """
        if hasattr(args, 'func'):
            return args.func(args)
        else:
            BaseCommand.print_error("No binary action specified")
            return 1
