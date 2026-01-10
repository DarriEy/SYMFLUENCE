"""
Base command class for SYMFLUENCE CLI commands.

This module provides the base class that all command handlers inherit from,
providing common utilities and interfaces.
"""

import sys
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, TYPE_CHECKING

from ..console import Console, console as global_console
from ..validators import validate_config_exists

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseCommand(ABC):
    """
    Base class for all CLI command handlers.

    Provides common functionality for loading configuration, handling errors,
    and executing commands.

    Attributes:
        _console: Shared console instance for all commands
    """

    _console: ClassVar[Console] = global_console

    @classmethod
    def set_console(cls, console: Console) -> None:
        """
        Set the console instance for all commands.

        Useful for testing or configuring output behavior.

        Args:
            console: Console instance to use
        """
        cls._console = console

    @staticmethod
    def load_typed_config(
        config_path: str,
        required: bool = True,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional["SymfluenceConfig"]:
        """
        Load configuration using the typed SymfluenceConfig system.

        This is the preferred method for loading configuration as it provides
        type-safe access and validation.

        Args:
            config_path: Path to configuration file
            required: Whether config file is required
            overrides: Optional overrides to apply

        Returns:
            SymfluenceConfig instance, or None if not required and not found
        """
        from symfluence.core.config.models import SymfluenceConfig
        from symfluence.core.exceptions import ConfigurationError

        path = Path(config_path)
        if not path.exists():
            if required:
                BaseCommand._console.error(f"Config file not found: {config_path}")
                return None
            return None

        try:
            return SymfluenceConfig.from_file(path, overrides=overrides)
        except ConfigurationError as e:
            BaseCommand._console.error(f"Configuration error: {e}")
            return None
        except Exception as e:
            BaseCommand._console.error(f"Failed to load config: {e}")
            return None

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
        result = validate_config_exists(config_path)

        if result.is_err:
            if required:
                error = result.first_error()
                BaseCommand._console.error(error.message if error else "Config validation failed")
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
            return './0_config_files/config_template.yaml'

    # Backward compatibility aliases for deprecated methods
    @classmethod
    def print_error(cls, message: str) -> None:
        """Deprecated: Use _console.error() instead."""
        cls._console.error(message)

    @classmethod
    def print_success(cls, message: str) -> None:
        """Deprecated: Use _console.success() instead."""
        cls._console.success(message)

    @classmethod
    def print_info(cls, message: str) -> None:
        """Deprecated: Use _console.info() instead."""
        cls._console.info(message)

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
