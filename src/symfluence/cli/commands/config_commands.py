"""
Configuration management command handlers for SYMFLUENCE CLI.

This module implements handlers for configuration file management and validation.
"""

from argparse import Namespace
from pathlib import Path

from .base import BaseCommand
from ..exit_codes import ExitCode


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
            # Get template files from SYMFLUENCE resources
            from symfluence.resources import list_config_templates

            BaseCommand._console.info("Available configuration templates:")
            BaseCommand._console.rule()

            templates = list_config_templates()
            if templates:
                for i, template_path in enumerate(templates, 1):
                    template_name = Path(template_path).name
                    # Skip backup files
                    if template_name.endswith('_backup.yaml'):
                        continue
                    description = ""
                    if 'quickstart_minimal' in template_name:
                        description = " (Minimal template - 10 required fields only)"
                    elif 'comprehensive' in template_name:
                        description = " (Complete reference - 406+ options)"
                    elif 'camels' in template_name:
                        description = " (CAMELS dataset template)"
                    elif 'fluxnet' in template_name:
                        description = " (FLUXNET sites template)"
                    elif 'norswe' in template_name:
                        description = " (Norwegian SWE template)"
                    elif template_name.startswith('config_template.yaml'):
                        description = " (Standard template with common options)"
                    BaseCommand._console.info(f"{i:2}. {template_name}{description}")
                BaseCommand._console.rule()
                BaseCommand._console.info(f"Total: {len([t for t in templates if not Path(t).name.endswith('_backup.yaml')])} templates")
            else:
                BaseCommand._console.info("No template files found")

            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Failed to list templates: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.GENERAL_ERROR

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
                BaseCommand._console.error(f"Config file not found: {config_file}")
                return ExitCode.FILE_NOT_FOUND

            BaseCommand._console.info(f"Updating configuration: {config_file}")

            if getattr(args, 'interactive', False):
                BaseCommand._console.indent("(Interactive mode)")
                BaseCommand._console.info("Interactive config update not yet implemented")
                return ExitCode.GENERAL_ERROR
            else:
                BaseCommand._console.info("Config update functionality not yet implemented")
                BaseCommand._console.info("You can edit the file directly with your preferred editor")

            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Config update failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.GENERAL_ERROR

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

            BaseCommand._console.info(f"Validating configuration: {config_path}")

            # Load config using typed system to validate
            config = BaseCommand.load_typed_config(config_path, required=True)
            if config is None:
                return ExitCode.CONFIG_ERROR

            BaseCommand._console.success("Configuration file is valid YAML")

            # Try to initialize SYMFLUENCE to validate structure
            from symfluence.core import SYMFLUENCE
            try:
                SYMFLUENCE(config_path, debug_mode=args.debug)
                BaseCommand._console.success("Configuration validated successfully")
                return ExitCode.SUCCESS
            except Exception as e:
                BaseCommand._console.error(f"Configuration structure validation failed: {e}")
                return ExitCode.CONFIG_ERROR

        except Exception as e:
            BaseCommand._console.error(f"Validation failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.VALIDATION_ERROR

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
            BaseCommand._console.info("Validating system environment...")
            BaseCommand._console.rule()

            import sys
            import platform

            BaseCommand._console.info(f"Python version: {sys.version}")
            BaseCommand._console.info(f"Platform: {platform.platform()}")
            BaseCommand._console.info(f"Python executable: {sys.executable}")

            # Check for required packages
            required_packages = ['yaml', 'numpy', 'pandas', 'xarray', 'geopandas']
            missing_packages = []

            for package in required_packages:
                try:
                    __import__(package)
                    BaseCommand._console.success(f"{package:15s} - installed")
                except ImportError:
                    BaseCommand._console.error(f"{package:15s} - MISSING")
                    missing_packages.append(package)

            BaseCommand._console.rule()

            if missing_packages:
                BaseCommand._console.error(f"Missing packages: {', '.join(missing_packages)}")
                return ExitCode.DEPENDENCY_ERROR
            else:
                BaseCommand._console.success("Environment validation passed")
                return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Environment validation failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.GENERAL_ERROR

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
            BaseCommand._console.error("No config action specified")
            return ExitCode.USAGE_ERROR
