"""
Global profiler context management for SYMFLUENCE.

Provides thread-safe global access to profiler instances and convenience
functions for checking profiling state. Supports cross-process profiling
by having worker processes write their profile data to files.

Manages two profilers:
1. IOProfiler: Python-level I/O operations
2. SystemIOProfiler: External tool I/O (SUMMA, mizuRoute, etc.)
"""

import atexit
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .io_profiler import IOProfiler
from .system_io_profiler import SystemIOProfiler

# Thread-local storage for profiler context
_profiler_context = threading.local()

# Global profiler instances (shared across threads)
_global_profiler: Optional[IOProfiler] = None
_global_system_profiler: Optional[SystemIOProfiler] = None
_profiler_lock = threading.Lock()

# Environment variable names
ENV_PROFILE_ENABLED = 'SYMFLUENCE_PROFILE'
ENV_PROFILE_DIR = 'SYMFLUENCE_PROFILE_DIR'
ENV_PROFILE_STACKS = 'SYMFLUENCE_PROFILE_STACKS'

# Flag to track if worker export has been registered
_worker_export_registered = False


def get_profiler() -> IOProfiler:
    """
    Get the global profiler instance.

    Creates a disabled profiler if none exists. The profiler must be explicitly
    enabled via set_profiler() or the --profile CLI flag.

    For worker processes, checks environment variables and registers an
    atexit handler to export profile data when the process exits.

    Returns:
        The global IOProfiler instance
    """
    global _global_profiler, _worker_export_registered

    if _global_profiler is None:
        with _profiler_lock:
            if _global_profiler is None:
                # Check environment variable for profiling
                env_enabled = os.environ.get(ENV_PROFILE_ENABLED, '').lower() in ('1', 'true', 'yes')
                capture_stacks = os.environ.get(ENV_PROFILE_STACKS, '').lower() in ('1', 'true', 'yes')
                _global_profiler = IOProfiler(enabled=env_enabled, capture_stack_traces=capture_stacks)

                # If profiling is enabled via environment, register atexit handler
                # This ensures worker processes export their profile data
                if env_enabled and not _worker_export_registered:
                    _register_worker_export()
                    _worker_export_registered = True

    return _global_profiler


def _register_worker_export():
    """Register atexit handler to export worker profile data."""
    def export_on_exit():
        global _global_profiler
        if _global_profiler is None or not _global_profiler.enabled:
            return

        # Check if there are any operations to export
        if len(_global_profiler._operations) == 0:
            return

        # Get profile directory from environment
        profile_dir = os.environ.get(ENV_PROFILE_DIR, '')
        if not profile_dir:
            return

        try:
            profile_dir = Path(profile_dir)
            profile_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename for this worker
            pid = os.getpid()
            profile_file = profile_dir / f"worker_profile_{pid}.json"

            _global_profiler.export_to_file(str(profile_file))
        except OSError:
            # Silently fail - don't want to crash on exit
            pass

    try:
        atexit.register(export_on_exit)
    except Exception:
        pass  # interpreter may already be shutting down


def set_profiler(profiler: Optional[IOProfiler]) -> None:
    """
    Set the global profiler instance.

    Args:
        profiler: The profiler to use globally, or None to clear
    """
    global _global_profiler

    with _profiler_lock:
        _global_profiler = profiler


def profiling_enabled() -> bool:
    """
    Check if profiling is currently enabled.

    Returns:
        True if profiling is enabled
    """
    profiler = get_profiler()
    return profiler.enabled


def enable_profiling(capture_stack_traces: bool = False) -> IOProfiler:
    """
    Enable profiling and return the profiler.

    Args:
        capture_stack_traces: Whether to capture call stacks (expensive)

    Returns:
        The enabled profiler instance
    """
    profiler = get_profiler()
    profiler.enabled = True
    profiler.capture_stack_traces = capture_stack_traces
    return profiler


def disable_profiling() -> None:
    """Disable profiling."""
    profiler = get_profiler()
    profiler.enabled = False


def setup_profiling_environment(profile_dir: str, capture_stacks: bool = False) -> None:
    """
    Set up environment variables for cross-process profiling.

    Should be called in the main process before spawning workers. Workers
    will detect these environment variables and export their profile data
    to the specified directory when they exit.

    Args:
        profile_dir: Directory where worker profile files should be written
        capture_stacks: Whether workers should capture stack traces
    """
    os.environ[ENV_PROFILE_ENABLED] = '1'
    os.environ[ENV_PROFILE_DIR] = str(profile_dir)
    if capture_stacks:
        os.environ[ENV_PROFILE_STACKS] = '1'

    # Create the profile directory
    Path(profile_dir).mkdir(parents=True, exist_ok=True)


def get_profile_directory() -> Optional[str]:
    """
    Get the profile directory from environment.

    Returns:
        Profile directory path or None if not set
    """
    return os.environ.get(ENV_PROFILE_DIR)


# ============================================================================
# System-Level Profiler (External Tools)
# ============================================================================

def get_system_profiler() -> SystemIOProfiler:
    """
    Get the global system I/O profiler instance.

    Creates a disabled profiler if none exists. The profiler must be explicitly
    enabled via enable_profiling() or the --profile CLI flag.

    Returns:
        The global SystemIOProfiler instance
    """
    global _global_system_profiler

    if _global_system_profiler is None:
        with _profiler_lock:
            if _global_system_profiler is None:
                # Check environment variable for profiling
                env_enabled = os.environ.get(ENV_PROFILE_ENABLED, '').lower() in ('1', 'true', 'yes')
                _global_system_profiler = SystemIOProfiler(enabled=env_enabled)

    return _global_system_profiler


def set_system_profiler(profiler: Optional[SystemIOProfiler]) -> None:
    """
    Set the global system profiler instance.

    Args:
        profiler: The system profiler to use globally, or None to clear
    """
    global _global_system_profiler

    with _profiler_lock:
        _global_system_profiler = profiler


def enable_system_profiling() -> SystemIOProfiler:
    """
    Enable system-level profiling and return the profiler.

    Returns:
        The enabled system profiler instance
    """
    profiler = get_system_profiler()
    profiler.enabled = True
    return profiler


def disable_system_profiling() -> None:
    """Disable system-level profiling."""
    profiler = get_system_profiler()
    profiler.enabled = False


@contextmanager
def profiling_scope(
    enabled: bool = True,
    component: Optional[str] = None,
    capture_stack_traces: bool = False
) -> Iterator[IOProfiler]:
    """
    Context manager for a profiling scope.

    Useful for enabling profiling for a specific code section.

    Args:
        enabled: Whether to enable profiling in this scope
        component: Component name for operations in this scope
        capture_stack_traces: Whether to capture call stacks

    Yields:
        The profiler instance
    """
    profiler = get_profiler()
    previous_enabled = profiler.enabled
    previous_stack_traces = profiler.capture_stack_traces
    previous_component = profiler._current_component

    try:
        profiler.enabled = enabled
        profiler.capture_stack_traces = capture_stack_traces
        if component:
            profiler._current_component = component
        yield profiler
    finally:
        profiler.enabled = previous_enabled
        profiler.capture_stack_traces = previous_stack_traces
        profiler._current_component = previous_component


class ProfilerContext:
    """
    Class-based context manager for profiling integration.

    Provides methods for tracking I/O operations that can be used as
    no-ops when profiling is disabled, avoiding overhead.

    Usage:
        ctx = ProfilerContext("parameter_manager")

        with ctx.track_netcdf_write("trialParams.nc"):
            write_netcdf(...)
    """

    def __init__(self, component_name: str):
        """
        Initialize profiler context for a component.

        Args:
            component_name: Name of the component using this context
        """
        self.component_name = component_name

    @property
    def profiler(self) -> IOProfiler:
        """Get the global profiler."""
        return get_profiler()

    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.profiler.enabled

    def set_iteration(self, iteration: int) -> None:
        """Set the current calibration iteration."""
        self.profiler.set_iteration(iteration)

    @contextmanager
    def track_file_write(self, path: str, size_bytes: Optional[int] = None) -> Iterator[None]:
        """Track a file write operation."""
        with self.profiler.track_file_write(path, size_bytes, self.component_name):
            yield

    @contextmanager
    def track_file_read(self, path: str, size_bytes: Optional[int] = None) -> Iterator[None]:
        """Track a file read operation."""
        with self.profiler.track_file_read(path, size_bytes, self.component_name):
            yield

    @contextmanager
    def track_netcdf_write(self, path: str, size_bytes: Optional[int] = None) -> Iterator[None]:
        """Track a NetCDF write operation."""
        with self.profiler.track_netcdf_write(path, size_bytes, self.component_name):
            yield

    @contextmanager
    def track_pickle_write(self, path: str, size_bytes: Optional[int] = None) -> Iterator[None]:
        """Track a pickle write operation."""
        with self.profiler.track_pickle_write(path, size_bytes, self.component_name):
            yield

    @contextmanager
    def track_dir_create(self, path: str) -> Iterator[None]:
        """Track directory creation."""
        with self.profiler.track_dir_create(path, self.component_name):
            yield

    @contextmanager
    def track_file_copy(
        self, src_path: str, dst_path: str, size_bytes: Optional[int] = None
    ) -> Iterator[None]:
        """Track a file copy operation."""
        with self.profiler.track_file_copy(src_path, dst_path, size_bytes, self.component_name):
            yield
