"""
mizuRoute build instructions for SYMFLUENCE.

This module defines how to build mizuRoute from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria

mizuRoute is a river network routing model developed at NCAR.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import (
    get_common_build_environment,
    get_netcdf_detection,
)


@BuildInstructionsRegistry.register('mizuroute')
def get_mizuroute_build_instructions():
    """
    Get mizuRoute build instructions.

    mizuRoute requires NetCDF libraries. The build uses make and
    requires editing the Makefile directly.

    Returns:
        Dictionary with complete build configuration for mizuRoute.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()

    return {
        'description': 'Mizukami routing model for river network routing',
        'config_path_key': 'INSTALL_PATH_MIZUROUTE',
        'config_exe_key': 'EXE_NAME_MIZUROUTE',
        'default_path_suffix': 'installs/mizuRoute/route/bin',
        'default_exe': 'mizuRoute.exe',
        'repository': 'https://github.com/ESCOMP/mizuRoute.git',
        'branch': 'serial',
        'install_dir': 'mizuRoute',
        'build_commands': [
            common_env,
            netcdf_detect,
            r'''
# Build mizuRoute - edit Makefile directly (it doesn't use env vars)
cd route/build
mkdir -p ../bin

F_MASTER_PATH="$(cd .. && pwd)"
echo "F_MASTER: $F_MASTER_PATH/"

# Validate NetCDF was detected
if [ -z "${NETCDF_FORTRAN}" ]; then
    echo "ERROR: Could not find NetCDF installation"
    exit 1
fi

# Edit the Makefile in-place
echo "=== Configuring Makefile ==="
perl -i -pe "s|^FC\s*=.*$|FC = gnu|" Makefile
perl -i -pe "s|^FC_EXE\s*=.*$|FC_EXE = ${FC:-gfortran}|" Makefile
perl -i -pe "s|^EXE\s*=.*$|EXE = mizuRoute.exe|" Makefile
perl -i -pe "s|^F_MASTER\s*=.*$|F_MASTER = $F_MASTER_PATH/|" Makefile
perl -i -pe "s|^\s*NCDF_PATH\s*=.*$| NCDF_PATH = ${NETCDF_FORTRAN}|" Makefile
perl -i -pe "s|^isOpenMP\s*=.*$|isOpenMP = no|" Makefile

# Fix LIBNETCDF for separate C/Fortran libs (e.g., macOS Homebrew)
if [ "${NETCDF_C}" != "${NETCDF_FORTRAN}" ]; then
    echo "Fixing LIBNETCDF for separate C/Fortran paths"
    perl -i -pe "s|^LIBNETCDF\s*=.*$|LIBNETCDF = -L${NETCDF_FORTRAN}/lib -lnetcdff -L${NETCDF_C}/lib -lnetcdf|" Makefile
fi

# Build
make clean || true
echo "Building mizuRoute..."
make 2>&1 | tee build.log || true

if [ -f "../bin/mizuRoute.exe" ]; then
    echo "Build successful - executable at ../bin/mizuRoute.exe"
else
    echo "ERROR: Executable not found at ../bin/mizuRoute.exe"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['route/bin/mizuRoute.exe'],
            'check_type': 'exists'
        },
        'order': 3
    }
