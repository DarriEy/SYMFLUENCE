"""
Exit codes for SYMFLUENCE CLI commands.

This module defines standard exit codes for consistent error reporting
across all CLI commands. Using specific exit codes allows scripts and
CI/CD systems to distinguish between different failure modes.
"""

from enum import IntEnum


class ExitCode(IntEnum):
    """
    Standard exit codes for CLI commands.

    These codes follow common Unix conventions where possible:
    - 0: Success
    - 1: General error
    - 2: Misuse of shell command (invalid arguments)
    - 128+N: Terminated by signal N

    Custom codes (3-127) are used for SYMFLUENCE-specific errors.
    """

    # Success
    SUCCESS = 0

    # General errors
    GENERAL_ERROR = 1
    USAGE_ERROR = 2  # Invalid arguments/usage

    # SYMFLUENCE-specific errors (3-29)
    CONFIG_ERROR = 3  # Configuration file issues
    VALIDATION_ERROR = 4  # Input validation failed
    FILE_NOT_FOUND = 5  # Required file missing
    DIRECTORY_NOT_FOUND = 6  # Required directory missing
    BINARY_ERROR = 7  # External binary issues (not found, failed)
    BINARY_BUILD_ERROR = 8  # Failed to build external binary
    NETWORK_ERROR = 9  # Network/download failures
    PERMISSION_ERROR = 10  # Permission denied
    TIMEOUT_ERROR = 11  # Operation timed out
    DEPENDENCY_ERROR = 12  # Missing dependency
    MODEL_ERROR = 13  # Hydrological model execution error
    WORKFLOW_ERROR = 14  # Workflow execution error
    DATA_ERROR = 15  # Data processing error

    # HPC/Job errors (20-29)
    JOB_SUBMIT_ERROR = 20  # Failed to submit job
    JOB_EXECUTION_ERROR = 21  # Job execution failed

    # Signal-based termination (128+signal)
    USER_INTERRUPT = 130  # Ctrl+C (128 + SIGINT=2)
    SIGTERM = 143  # Terminated (128 + SIGTERM=15)

    @classmethod
    def from_exception(cls, exc: Exception) -> "ExitCode":
        """
        Determine exit code from exception type.

        Args:
            exc: The exception that was raised

        Returns:
            Appropriate exit code for the exception type
        """
        from symfluence.core.exceptions import (
            ConfigurationError,
            SYMFLUENCEError,
        )

        if isinstance(exc, FileNotFoundError):
            return cls.FILE_NOT_FOUND
        if isinstance(exc, PermissionError):
            return cls.PERMISSION_ERROR
        if isinstance(exc, TimeoutError):
            return cls.TIMEOUT_ERROR
        if isinstance(exc, KeyboardInterrupt):
            return cls.USER_INTERRUPT
        if isinstance(exc, ConfigurationError):
            return cls.CONFIG_ERROR
        if isinstance(exc, SYMFLUENCEError):
            return cls.GENERAL_ERROR
        return cls.GENERAL_ERROR
