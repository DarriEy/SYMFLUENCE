"""
Configuration management command handlers for SYMFLUENCE CLI.

This module implements handlers for configuration file management and validation.
"""

import sys
from argparse import Namespace
from pathlib import Path

from .base import BaseCommand


class ConfigCommands(BaseCommand):
    """Handlers for configuration management commands."""

    @staticmethod
    def list_templates(args: Namespace) -> int:
        """
        Execute: symfluence config list-templates

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Load templates from package data
            from symfluence.resources import list_config_templates

            BaseCommand.print_info("Available configuration templates:")
            BaseCommand.print_info("=" * 70)

            templates = list_config_templates()
            if templates:
                for i, template in enumerate(templates, 1):
                    BaseCommand.print_info(f"{i:2}. {template.name}")
                BaseCommand.print_info("=" * 70)
                BaseCommand.print_info(f"Total: {len(templates)} templates")
            else:
                BaseCommand.print_info("No template files found")

            return 0

        except Exception as e:
            BaseCommand.print_error(f"Failed to list templates: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def update(args: Namespace) -> int:
        """
        Execute: symfluence config update CONFIG_FILE

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            config_file = args.config_file

            # Validate file exists
            if not Path(config_file).exists():
                BaseCommand.print_error(f"Config file not found: {config_file}")
                return 1

            BaseCommand.print_info(f"📝 Updating configuration: {config_file}")

            if getattr(args, 'interactive', False):
                BaseCommand.print_info("   (Interactive mode)")
                BaseCommand.print_info("Interactive config update not yet implemented")
                return 1
            else:
                BaseCommand.print_info("Config update functionality not yet implemented")
                BaseCommand.print_info("You can edit the file directly with your preferred editor")

            return 0

        except Exception as e:
            BaseCommand.print_error(f"Config update failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def validate(args: Namespace) -> int:
        """
        Execute: symfluence config validate

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            config_path = BaseCommand.get_config_path(args)

            BaseCommand.print_info(f"✓ Validating configuration: {config_path}")

            # Load config to validate YAML syntax
            config = BaseCommand.load_config(config_path, required=True)
            if config is None:
                return 1

            BaseCommand.print_info("✓ Configuration file is valid YAML")

            # Try to initialize SYMFLUENCE to validate structure
            from symfluence.core import SYMFLUENCE
            try:
                symfluence = SYMFLUENCE(config_path, debug_mode=args.debug)
                BaseCommand.print_success("Configuration validated successfully")
                return 0
            except Exception as e:
                BaseCommand.print_error(f"Configuration structure validation failed: {e}")
                return 1

        except Exception as e:
            BaseCommand.print_error(f"Validation failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def validate_env(args: Namespace) -> int:
        """
        Execute: symfluence config validate-env

        Validate system environment for SYMFLUENCE.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            BaseCommand.print_info("🔍 Validating system environment...")
            BaseCommand.print_info("=" * 70)

            import sys
            import platform

            BaseCommand.print_info(f"Python version: {sys.version}")
            BaseCommand.print_info(f"Platform: {platform.platform()}")
            BaseCommand.print_info(f"Python executable: {sys.executable}")

            # Check for required packages
            required_packages = ['yaml', 'numpy', 'pandas', 'xarray', 'geopandas']
            missing_packages = []

            for package in required_packages:
                try:
                    __import__(package)
                    BaseCommand.print_info(f"✓ {package:15s} - installed")
                except ImportError:
                    BaseCommand.print_info(f"✗ {package:15s} - MISSING")
                    missing_packages.append(package)

            BaseCommand.print_info("=" * 70)

            if missing_packages:
                BaseCommand.print_error(f"Missing packages: {', '.join(missing_packages)}")
                return 1
            else:
                BaseCommand.print_success("Environment validation passed")
                return 0

        except Exception as e:
            BaseCommand.print_error(f"Environment validation failed: {e}")
            if args.debug:
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
            BaseCommand.print_error("No config action specified")
            return 1
