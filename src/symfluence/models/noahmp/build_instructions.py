# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Noah-MP (noah-owp-modular) build instructions for SYMFLUENCE."""
from __future__ import annotations

from symfluence.core.registries import R


@R.build_instructions.add('noahmp')
def get_noahmp_build_instructions():
    """Get Noah-MP standalone install instructions.

    noah-owp-modular is a Fortran model requiring a Fortran compiler
    and NetCDF libraries.  Build produces noah_owp_modular.exe in run/.
    """
    return {
        'description': 'Noah-MP Standalone Land Surface Model (NOAA-OWP)',
        'config_path_key': 'NOAHMP_INSTALL_PATH',
        'config_exe_key': 'NOAHMP_EXE',
        'default_path_suffix': 'installs/noah-owp-modular',
        'default_exe': 'noah_owp_modular.exe',
        'repository': 'https://github.com/NOAA-OWP/noah-owp-modular.git',
        'branch': 'main',
        'install_dir': 'noah-owp-modular',
        'build_commands': [
            r"""
# Noah-MP (noah-owp-modular) Install Script for SYMFLUENCE
set -e
echo "=== Noah-MP Installation Starting ==="

# Check for Fortran compiler
if ! command -v gfortran &>/dev/null; then
    echo "ERROR: gfortran not found. Install a Fortran compiler first."
    echo "  macOS:  brew install gcc"
    echo "  Ubuntu: sudo apt-get install gfortran"
    exit 1
fi

# Check for NetCDF
if ! command -v nf-config &>/dev/null && ! command -v nc-config &>/dev/null; then
    echo "ERROR: NetCDF-Fortran not found."
    echo "  macOS:  brew install netcdf netcdf-fortran"
    echo "  Ubuntu: sudo apt-get install libnetcdf-dev libnetcdff-dev"
    exit 1
fi

# Configure (select appropriate config for platform)
if [ "$(uname)" = "Darwin" ]; then
    # macOS gfortran
    if [ -f "config/user_build_options.macos.gfortran" ]; then
        cp config/user_build_options.macos.gfortran user_build_options
    elif [ -f "config/user_build_options.bigsur.gfortran" ]; then
        cp config/user_build_options.bigsur.gfortran user_build_options
    else
        echo "WARNING: No macOS config found, trying ./configure"
        ./configure 2>/dev/null || true
    fi
else
    # Linux gfortran
    if [ -f "config/user_build_options.pgf90.linux" ]; then
        cp config/user_build_options.pgf90.linux user_build_options
    else
        ./configure 2>/dev/null || true
    fi
fi

make clean 2>/dev/null || true
make

if [ -f "run/noah_owp_modular.exe" ]; then
    echo "=== Noah-MP Installation Complete ==="
    echo "Executable: run/noah_owp_modular.exe"
else
    echo "ERROR: Build failed — noah_owp_modular.exe not found in run/"
    exit 1
fi
            """.strip()
        ],
        'dependencies': ['gfortran', 'netcdf-fortran'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['run/noah_owp_modular.exe'],
            'check_type': 'exists',
        },
        'order': 24,
        'optional': True,
    }
