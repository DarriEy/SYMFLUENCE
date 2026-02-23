"""
Worker Environment Configuration

Manages environment variables for worker processes to control threading
and file locking behavior.

This module provides a class-based interface to the centralized HDF5 safety
configuration in core.hdf5_safety, allowing for custom overrides while
maintaining consistency across the codebase.
"""

import os
from typing import Dict

from symfluence.core.hdf5_safety import (
    HDF5_ENV_VARS,
    THREAD_ENV_VARS,
    get_worker_environment,
)


class WorkerEnvironmentConfig:
    """
    Manages environment variables for parallel worker processes.

    Controls threading behavior for numerical libraries and HDF5/NetCDF
    file locking to prevent conflicts during parallel execution.

    This class wraps the centralized configuration from core.hdf5_safety
    while allowing custom overrides for specific worker types.
    """

    # Default environment variables derived from centralized module
    DEFAULT_ENV_VARS: Dict[str, str] = {**HDF5_ENV_VARS, **THREAD_ENV_VARS}

    def __init__(self, custom_vars: Dict[str, str] = None):
        """
        Initialize worker environment configuration.

        Args:
            custom_vars: Optional custom environment variables to add/override
        """
        self._env_vars = get_worker_environment(include_thread_limits=True)
        if custom_vars:
            self._env_vars.update(custom_vars)

    def get_environment(self) -> Dict[str, str]:
        """
        Get environment variables for worker processes.

        Returns:
            Dictionary of environment variables to set
        """
        return self._env_vars.copy()

    def apply_to_current_process(self) -> None:
        """Apply worker environment variables to current process."""
        os.environ.update(self._env_vars)

    def merge_with_current_env(self) -> Dict[str, str]:
        """
        Create a copy of current environment merged with worker settings.

        Returns:
            Complete environment dictionary for subprocess execution
        """
        env = os.environ.copy()
        env.update(self._env_vars)
        return env
