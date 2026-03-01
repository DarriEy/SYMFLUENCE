# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Centralized HDF5/netCDF4 thread safety configuration.

This module provides a single source of truth for HDF5 file locking workarounds
and thread safety settings. Previously, this logic was duplicated across 8+ files
with inconsistent implementations.

The HDF5 library is not thread-safe by default, and concurrent access from
background threads (like tqdm's monitor) can cause segmentation faults during
netCDF/HDF5 file operations.

Usage:
    # At application startup (done automatically in symfluence/__init__.py)
    from symfluence.core.hdf5_safety import configure_hdf5_safety
    configure_hdf5_safety()

    # In worker processes
    from symfluence.core.hdf5_safety import apply_worker_environment
    apply_worker_environment()

    # Get environment dict for subprocess
    from symfluence.core.hdf5_safety import get_worker_environment
    env = get_worker_environment()
    subprocess.run(cmd, env=env)
"""

import gc
import os
from typing import Dict

# =============================================================================
# Environment Variable Constants
# =============================================================================

HDF5_ENV_VARS: Dict[str, str] = {
    'HDF5_USE_FILE_LOCKING': 'FALSE',
    'HDF5_DISABLE_VERSION_CHECK': '1',
    'NETCDF_DISABLE_LOCKING': '1',
}
"""Environment variables for HDF5/netCDF file locking safety."""


THREAD_ENV_VARS: Dict[str, str] = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'KMP_DUPLICATE_LIB_OK': 'TRUE',  # Prevent OpenMP conflicts on macOS
}
"""Environment variables to force single-threaded execution in numerical libraries."""


# =============================================================================
# Configuration Functions
# =============================================================================

def configure_hdf5_safety(disable_tqdm_monitor: bool = True) -> None:
    """
    Configure HDF5/netCDF4 thread safety at application startup.

    This function must be called BEFORE any HDF5/netCDF4/xarray imports occur.
    It is automatically called by symfluence/__init__.py.

    Args:
        disable_tqdm_monitor: If True, disable tqdm's background monitor thread
                              which can cause segfaults with netCDF4/HDF5.
    """
    # Set ALL HDF5 and threading environment variables
    # These must be set BEFORE importing libraries to prevent thread pool creation.
    # HDF5 locking vars are FORCE-SET (not setdefault) because on HPC systems
    # the run session often differs from the install session, and the HDF5 C
    # library reads HDF5_USE_FILE_LOCKING at dlopen time.  If the var is unset
    # (or the module environment sets it to something else), file-locking errors
    # on parallel filesystems (Lustre/GPFS/BeeGFS) are almost guaranteed.
    env_vars = get_worker_environment(include_thread_limits=True)
    for key, value in env_vars.items():
        if key in HDF5_ENV_VARS:
            # Force-set HDF5/netCDF locking vars — these are critical on HPC
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)

    # Configure xarray to minimize file caching and prefer h5netcdf backend
    # h5netcdf is more stable than netCDF4 for concurrent access
    try:
        import xarray as xr
        xr.set_options(
            file_cache_maxsize=1,  # Minimal file cache
        )
        # Try to set h5netcdf as default backend if available
        try:
            import h5netcdf  # noqa: F401
            # h5netcdf is available, will be used when backend='h5netcdf'
        except ImportError:
            pass  # h5netcdf not available, will fall back to netCDF4
    except (ImportError, AttributeError):
        pass  # xarray not yet imported or doesn't support this option

    # Disable tqdm monitor thread to prevent segfaults
    if disable_tqdm_monitor:
        _disable_tqdm_monitor()


def get_worker_environment(include_thread_limits: bool = True) -> Dict[str, str]:
    """
    Get environment variables for worker processes.

    Returns a dictionary of environment variables that should be set in
    worker processes to ensure HDF5/netCDF safety and single-threaded
    execution of numerical libraries.

    Args:
        include_thread_limits: If True, include variables that limit threading
                               in numerical libraries (OMP, MKL, etc.)

    Returns:
        Dictionary of environment variables to set
    """
    env_vars = HDF5_ENV_VARS.copy()
    if include_thread_limits:
        env_vars.update(THREAD_ENV_VARS)
    return env_vars


def apply_worker_environment() -> None:
    """
    Apply worker environment variables to the current process.

    Call this at the start of worker processes to ensure HDF5 safety
    and single-threaded execution.
    """
    env_vars = get_worker_environment(include_thread_limits=True)
    os.environ.update(env_vars)


def merge_with_current_env(include_thread_limits: bool = True) -> Dict[str, str]:
    """
    Create a copy of the current environment merged with worker settings.

    Useful for subprocess execution where you need the full environment.

    Args:
        include_thread_limits: If True, include thread limiting variables

    Returns:
        Complete environment dictionary for subprocess execution
    """
    env = os.environ.copy()
    env.update(get_worker_environment(include_thread_limits))
    return env


def clear_xarray_cache() -> None:
    """
    Clear xarray's file manager cache to prevent stale file handles.

    This should be called after intensive file operations, especially
    in worker processes that may have residual file handles from
    previous iterations.
    """
    try:
        import xarray as xr

        # Try different cache clearing approaches for various xarray versions
        # xarray >= 0.19: CachingFileManager uses a module-level cache
        if hasattr(xr.backends, 'file_manager'):
            fm = xr.backends.file_manager
            # Old style cache (xarray < 2022)
            if hasattr(fm, 'FILE_CACHE'):
                fm.FILE_CACHE.clear()
            # Try to find CachingFileManager's cache
            if hasattr(fm, 'CachingFileManager'):
                cfm = fm.CachingFileManager
                cache = getattr(cfm, '_cache', None)
                if cache is not None:
                    try:
                        cache.clear()
                    except (TypeError, AttributeError):
                        pass

        # Also try closing any open file handles via the backends
        for backend_name in ['netcdf4', 'h5netcdf', 'scipy']:
            try:
                backend_mod = getattr(xr.backends, f'{backend_name}_', None)
                if backend_mod and hasattr(backend_mod, '_clear_cache'):
                    backend_mod._clear_cache()
            except (AttributeError, TypeError):
                pass

    except (ImportError, AttributeError):
        pass  # xarray internals may vary by version

    # Also try to clear netCDF4's internal cache if available
    try:
        import netCDF4 as nc4
        # netCDF4 doesn't have a public cache API, but we can trigger cleanup
        if hasattr(nc4, '_clear_cache'):
            nc4._clear_cache()
    except (ImportError, AttributeError):
        pass

    # Force garbage collection to release file handles
    gc.collect()
    gc.collect()  # Second pass for cyclic references


# =============================================================================
# Internal Helpers
# =============================================================================

def _disable_tqdm_monitor() -> None:
    """Disable tqdm's background monitor thread."""
    try:
        import tqdm
        tqdm.tqdm.monitor_interval = 0
        if tqdm.tqdm.monitor is not None:
            try:
                tqdm.tqdm.monitor.exit()
            except (AttributeError, RuntimeError):
                pass  # Monitor may already be stopped
            tqdm.tqdm.monitor = None
    except ImportError:
        pass


def ensure_thread_safety() -> None:
    """
    Ensure thread-safe environment for netCDF4/HDF5 operations.

    This is a convenience function that combines environment setup
    with cache clearing. Call before intensive HDF5 operations.

    Note:
        This is equivalent to calling:
        - apply_worker_environment()
        - _disable_tqdm_monitor()
        - clear_xarray_cache()
    """
    apply_worker_environment()
    _disable_tqdm_monitor()
    clear_xarray_cache()


def prepare_for_netcdf_operation() -> None:
    """
    Aggressive preparation before intensive netCDF/HDF5 operations.

    This function should be called before operations that are known to
    cause issues with HDF5 thread safety (e.g., easymore remapping).
    It performs a more thorough cleanup than ensure_thread_safety().

    This function:
    1. Sets all HDF5/netCDF safety environment variables
    2. Disables tqdm's monitor thread
    3. Clears xarray's file cache
    4. Forces garbage collection of netCDF4 Dataset objects
    5. Attempts to close any lingering file handles
    6. Disables xarray's file caching to prevent cache-related segfaults
    """
    # Apply all environment variables
    apply_worker_environment()

    # Disable tqdm monitor
    _disable_tqdm_monitor()

    # Disable xarray's file caching to prevent cache-related segfaults
    try:
        import xarray as xr
        # Set cache size to 0 to disable caching
        if hasattr(xr.backends, 'file_manager'):
            xr.set_options(file_cache_maxsize=1)  # Minimal cache
    except (ImportError, AttributeError):
        pass

    # First GC pass to find unreferenced Dataset objects
    gc.collect()

    # Try to close any open netCDF4 datasets
    try:
        import netCDF4 as nc4
        # netCDF4 tracks open datasets internally in some versions
        if hasattr(nc4, '_active_datasets'):
            for ds in list(nc4._active_datasets):
                try:
                    ds.close()
                except Exception:  # noqa: BLE001 — must-not-raise contract
                    pass
    except (ImportError, AttributeError):
        pass

    # Clear xarray cache
    clear_xarray_cache()

    # Additional GC passes for thorough cleanup
    gc.collect()
    gc.collect()
