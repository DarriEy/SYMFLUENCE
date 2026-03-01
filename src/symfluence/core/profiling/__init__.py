# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE Profiling Module.

Provides comprehensive I/O and performance profiling for SYMFLUENCE workflows,
particularly useful for diagnosing IOPS issues on HPC shared filesystems during
large-scale calibration jobs.

Two-Level Profiling:
    1. Python-level I/O: Tracks Python file operations (NetCDF, pickle, etc.)
    2. System-level I/O: Tracks external tool execution (SUMMA, mizuRoute, etc.)

Usage:
    # Enable via CLI flag
    symfluence workflow run --config config.yaml --profile

    # Python-level profiling
    from symfluence.core.profiling import IOProfiler, get_profiler

    profiler = get_profiler()
    with profiler.track_file_write("trialParams.nc", size_bytes=1024):
        write_netcdf_file(...)

    # System-level profiling (external tools)
    from symfluence.core.profiling import get_system_profiler

    sys_profiler = get_system_profiler()
    with sys_profiler.profile_subprocess(
        command=['summa.exe', '-m', 'fileManager.txt'],
        component='summa',
        iteration=1
    ) as proc:
        result = proc.run(stdout=log_file, stderr=subprocess.STDOUT)

    # Generate combined report
    profiler.generate_report("/path/to/python_io_report.json")
    sys_profiler.generate_report("/path/to/system_io_report.json")
"""

from .io_profiler import IOOperation, IOProfiler
from .profiler_context import (
    ProfilerContext,
    disable_profiling,
    disable_system_profiling,
    enable_profiling,
    enable_system_profiling,
    get_profile_directory,
    get_profiler,
    get_system_profiler,
    profiling_enabled,
    set_profiler,
    set_system_profiler,
    setup_profiling_environment,
)
from .system_io_profiler import ProcessIOStats, SystemIOProfiler

__all__ = [
    # Python-level profiling
    'IOProfiler',
    'IOOperation',
    'ProfilerContext',
    'get_profiler',
    'set_profiler',
    'profiling_enabled',
    'enable_profiling',
    'disable_profiling',
    'setup_profiling_environment',
    'get_profile_directory',
    # System-level profiling
    'SystemIOProfiler',
    'ProcessIOStats',
    'get_system_profiler',
    'set_system_profiler',
    'enable_system_profiling',
    'disable_system_profiling',
]
