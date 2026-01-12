"""
Binary service facade for external tool management.

This module provides a unified interface to the modular services:
- ToolInstaller: For installing tools from source
- ToolValidator: For validating installed tools
- SystemDiagnostics: For system health checks
"""

from typing import Any, Dict, List, Optional, Union

from .console import Console, console as global_console
from .external_tools_config import get_external_tools_definitions
from .services import ToolInstaller, ToolValidator, SystemDiagnostics
from .services.base import BaseService


class BinaryService(BaseService):
    """
    Facade for binary tool management.

    Coordinates the ToolInstaller, ToolValidator, and SystemDiagnostics
    services to provide a unified interface for binary management operations.

    This is the successor to BinaryManager, providing the same interface
    but delegating to focused services internally.
    """

    def __init__(
        self,
        external_tools: Optional[Dict[str, Any]] = None,
        console: Optional[Console] = None,
    ):
        """
        Initialize the BinaryService.

        Args:
            external_tools: Dictionary of tool definitions. If None, loads from config.
            console: Console instance for output. If None, uses global console.
        """
        # Initialize parent class (BaseService)
        super().__init__(console=console or global_console)

        self._external_tools = external_tools

        # Initialize services with shared console
        self._installer = ToolInstaller(
            external_tools=external_tools,
            console=self._console,
        )
        self._validator = ToolValidator(
            external_tools=external_tools,
            console=self._console,
        )
        self._diagnostics = SystemDiagnostics(
            external_tools=external_tools,
            console=self._console,
        )

    @property
    def external_tools(self) -> Dict[str, Any]:
        """Get external tools definitions."""
        if self._external_tools is None:
            self._external_tools = get_external_tools_definitions()
        return self._external_tools

    # =========================================================================
    # Installation Methods (delegated to ToolInstaller)
    # =========================================================================

    def get_executables(
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
        return self._installer.install(
            specific_tools=specific_tools,
            symfluence_instance=symfluence_instance,
            force=force,
            dry_run=dry_run,
        )

    def install(
        self,
        tools: Optional[List[str]] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Install external tools (alias for get_executables).

        Args:
            tools: List of specific tools to install. If None, installs all.
            force: If True, reinstall even if already exists.
            dry_run: If True, only show what would be done.

        Returns:
            Dictionary with installation results.
        """
        return self._installer.install(
            specific_tools=tools,
            force=force,
            dry_run=dry_run,
        )

    # =========================================================================
    # Validation Methods (delegated to ToolValidator)
    # =========================================================================

    def validate_binaries(
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
        return self._validator.validate(
            symfluence_instance=symfluence_instance,
            verbose=verbose,
        )

    def validate(self, verbose: bool = False) -> Union[bool, Dict[str, Any]]:
        """
        Validate installed tools (alias for validate_binaries).

        Args:
            verbose: If True, show detailed output.

        Returns:
            True if all tools valid, otherwise a dictionary with validation results.
        """
        return self._validator.validate(verbose=verbose)

    # =========================================================================
    # Diagnostics Methods (delegated to SystemDiagnostics)
    # =========================================================================

    def doctor(self) -> bool:
        """
        Run system diagnostics: check binaries, toolchain, and system libraries.

        Returns:
            True if diagnostics completed successfully.
        """
        return self._diagnostics.run_diagnostics()

    def run_doctor(self) -> bool:
        """Alias for doctor() for backward compatibility."""
        return self.doctor()

    def tools_info(self) -> bool:
        """
        Display installed tools information from toolchain metadata.

        Returns:
            True if tools info was displayed, False if no metadata found.
        """
        return self._diagnostics.get_tools_info()

    def show_tools_info(self) -> bool:
        """Alias for tools_info() for backward compatibility."""
        return self.tools_info()

    def detect_npm_binaries(self):
        """
        Detect if SYMFLUENCE binaries are installed via npm.

        Returns:
            Path to npm-installed binaries, or None if not found.
        """
        return self._diagnostics.detect_npm_binaries()

    # =========================================================================
    # Legacy Dispatcher (for backward compatibility)
    # =========================================================================

    def handle_binary_management(self, execution_plan: Dict[str, Any]) -> bool:
        """
        Legacy dispatcher for binary management operations.

        Args:
            execution_plan: Dictionary containing binary_operations.

        Returns:
            True if operation succeeded, False otherwise.
        """
        ops = execution_plan.get("binary_operations", {})

        if ops.get("doctor"):
            self.doctor()
            return True
        if ops.get("tools_info"):
            self.tools_info()
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


# Backward compatibility alias
BinaryManager = BinaryService
