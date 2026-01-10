"""
Worker Environment Configuration

Manages environment variables for worker processes to control threading
and file locking behavior.
"""

import os
from typing import Dict


class WorkerEnvironmentConfig:
    """
    Manages environment variables for parallel worker processes.

    Controls threading behavior for numerical libraries and HDF5/NetCDF
    file locking to prevent conflicts during parallel execution.
    """

    # Default environment variables for worker processes
    DEFAULT_ENV_VARS: Dict[str, str] = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'NETCDF_DISABLE_LOCKING': '1',
        'HDF5_USE_FILE_LOCKING': 'FALSE',
        'HDF5_DISABLE_VERSION_CHECK': '1',
    }

    def __init__(self, custom_vars: Dict[str, str] = None):
        """
        Initialize worker environment configuration.

        Args:
            custom_vars: Optional custom environment variables to add/override
        """
        self._env_vars = self.DEFAULT_ENV_VARS.copy()
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
        for key, value in self._env_vars.items():
            os.environ[key] = value

    def merge_with_current_env(self) -> Dict[str, str]:
        """
        Create a copy of current environment merged with worker settings.

        Returns:
            Complete environment dictionary for subprocess execution
        """
        env = os.environ.copy()
        env.update(self._env_vars)
        return env
