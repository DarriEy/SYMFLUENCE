"""
Base command class for SYMFLUENCE CLI commands.

This module provides the base class that all command handlers inherit from,
providing common utilities and interfaces.
"""

import sys
import yaml
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

from ..validators import validate_config_exists


class BaseCommand(ABC):
    """
    Base class for all CLI command handlers.

    Provides common functionality for loading configuration, handling errors,
    and executing commands.
    """

    @staticmethod
    def load_config(config_path: str, required: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file
            required: Whether config file is required. If True, raises error if not found.

        Returns:
            Configuration dictionary, or None if not required and not found

        Raises:
            SystemExit: If required=True and config file doesn't exist or is invalid
        """
        path = Path(config_path)

        if not path.exists():
            if required:
                print(f"Error: Config file not found: {config_path}", file=sys.stderr)
                sys.exit(1)
            else:
                return None

        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML in config file: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to load config file: {e}", file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def validate_config(config_path: str, required: bool = True) -> bool:
        """
        Validate that config file exists and is readable.

        Args:
            config_path: Path to configuration file
            required: Whether config is required

        Returns:
            True if valid (or not required and doesn't exist), False otherwise
        """
        is_valid, error_msg = validate_config_exists(config_path)

        if not is_valid:
            if required:
                print(f"Error: {error_msg}", file=sys.stderr)
                return False
            else:
                # Not required and doesn't exist is OK
                return True

        return True

    @staticmethod
    def get_config_path(args: Namespace) -> str:
        """
        Get configuration file path from args, with fallback to default.

        Args:
            args: Parsed arguments namespace

        Returns:
            Path to configuration file
        """
        if hasattr(args, 'config') and args.config:
            return args.config
        else:
            # Default config path
            return './config.yaml'

    @staticmethod
    def print_error(message: str) -> None:
        """
        Print error message to stderr.

        Args:
            message: Error message to print
        """
        print(f"Error: {message}", file=sys.stderr)

    @staticmethod
    def print_success(message: str) -> None:
        """
        Print success message to stdout.

        Args:
            message: Success message to print
        """
        print(f"✓ {message}")

    @staticmethod
    def print_info(message: str) -> None:
        """
        Print informational message to stdout.

        Args:
            message: Info message to print
        """
        print(message)

    @staticmethod
    @abstractmethod
    def execute(args: Namespace) -> int:
        """
        Execute the command.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        pass
