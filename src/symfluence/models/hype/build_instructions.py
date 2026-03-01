# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HYPE build instructions for SYMFLUENCE.

This module defines how to build HYPE from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria

HYPE (Hydrological Predictions for the Environment) is a semi-distributed
hydrological model developed by SMHI (Swedish Meteorological and
Hydrological Institute).
"""

from symfluence.cli.services import (
    BuildInstructionsRegistry,
    get_common_build_environment,
    get_netcdf_detection,
)


@BuildInstructionsRegistry.register('hype')
def get_hype_build_instructions():
    """
    Get HYPE build instructions.

    HYPE can be built with or without NetCDF support. The build uses
    make and requires a Fortran compiler.

    Returns:
        Dictionary with complete build configuration for HYPE.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()

    return {
        'description': 'HYPE - Hydrological Predictions for the Environment',
        'config_path_key': 'HYPE_INSTALL_PATH',
        'config_exe_key': 'HYPE_EXE',
        'default_path_suffix': 'installs/hype/bin',
        'default_exe': 'hype',
        'repository': 'git://git.code.sf.net/p/hype/code',
        'branch': None,
        'install_dir': 'hype',
        'build_commands': [
            common_env,
            netcdf_detect,
            r'''
# Build HYPE from SourceForge git repository
set -e
mkdir -p bin

if [ -z "${NETCDF_FORTRAN}" ]; then
    echo "NetCDF not found, building basic version..."
    make hype FC="${FC:-gfortran}" || { echo "HYPE compilation failed"; exit 1; }
else
    echo "Building HYPE with NetCDF support..."
    export NCDF_PATH="${NETCDF_FORTRAN}"
    make hype libs=netcdff FC="${FC:-gfortran}" || {
        echo "NetCDF build failed, trying basic build..."
        make clean || true
        make hype FC="${FC:-gfortran}" || { echo "HYPE compilation failed"; exit 1; }
    }
fi

# Stage binary (handles both Linux 'hype' and Windows 'hype.exe')
if [ -f "hype.exe" ]; then
    mv hype.exe bin/hype
elif [ -f "hype" ]; then
    mv hype bin/
elif [ ! -f "bin/hype" ] && [ ! -f "bin/hype.exe" ]; then
    echo "HYPE binary not found after build"
    exit 1
fi
# Normalise name: ensure bin/hype exists (may be bin/hype.exe on Windows)
if [ ! -f "bin/hype" ] && [ -f "bin/hype.exe" ]; then
    cp bin/hype.exe bin/hype
fi
chmod +x bin/hype 2>/dev/null || true
echo "HYPE build successful"
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,  # HYPE exits with error when run without args
        'verify_install': {
            'file_paths': ['bin/hype'],
            'check_type': 'exists'
        },
        'order': 11
    }
