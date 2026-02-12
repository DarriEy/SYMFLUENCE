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

# Ensure NetCDF paths are set, preferring conda installation
# CONDA_LIB_PREFIX is set by the common build environment snippet
# and handles Windows conda layout (Library/ subdir).
if [ -n "$CONDA_PREFIX" ] && [ -f "${CONDA_LIB_PREFIX:-$CONDA_PREFIX}/bin/nf-config" ]; then
    export NETCDF_FORTRAN="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
    export NETCDF_C="${NETCDF_C:-${CONDA_LIB_PREFIX:-$CONDA_PREFIX}}"
    echo "Using conda NetCDF-Fortran at: $NETCDF_FORTRAN"
fi

# Validate NetCDF was detected
if [ -z "${NETCDF_FORTRAN}" ] || [ ! -d "${NETCDF_FORTRAN}/include" ]; then
    echo "ERROR: Could not find NetCDF installation"
    echo "NETCDF_FORTRAN=${NETCDF_FORTRAN}"
    echo "CONDA_PREFIX=${CONDA_PREFIX:-not set}"
    if [ -n "$CONDA_PREFIX" ]; then
        echo "Contents of CONDA_PREFIX/include:"
        ls -la "$CONDA_PREFIX/include" 2>/dev/null | head -10 || true
        echo "Contents of CONDA_PREFIX/bin/nf-config:"
        ls -la "$CONDA_PREFIX/bin/nf-config" 2>/dev/null || echo "nf-config not found"
    fi
    exit 1
fi

# On Windows (MSYS2/MinGW), convert backslash paths to forward slashes.
# Perl interprets \U, \e, \c etc. as escape sequences in replacement strings,
# corrupting Windows paths like C:\Users\... into C:SERS...
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        NETCDF_FORTRAN="${NETCDF_FORTRAN//\\//}"
        NETCDF_C="${NETCDF_C//\\//}"
        ;;
esac

# Edit the Makefile in-place
echo "=== Configuring Makefile ==="
perl -i -pe "s|^FC\s*=.*$|FC = gnu|" Makefile
perl -i -pe "s|^FC_EXE\s*=.*$|FC_EXE = ${FC:-gfortran}|" Makefile
perl -i -pe "s|^EXE\s*=.*$|EXE = mizuRoute.exe|" Makefile
perl -i -pe "s|^F_MASTER\s*=.*$|F_MASTER = $F_MASTER_PATH/|" Makefile
perl -i -pe "s|^\s*NCDF_PATH\s*=.*$|NCDF_PATH = ${NETCDF_FORTRAN}|" Makefile
perl -i -pe "s|^isOpenMP\s*=.*$|isOpenMP = no|" Makefile

# Fix LIBNETCDF for separate C/Fortran libs (e.g., macOS Homebrew, HPC with separate installs)
# Note: LIBNETCDF in mizuRoute is a multi-line definition with backslash continuation,
# so we must also remove the orphaned continuation line after replacement.
if [ "${NETCDF_C}" != "${NETCDF_FORTRAN}" ]; then
    echo "Fixing LIBNETCDF for separate C/Fortran paths"
    perl -i -pe "s|^LIBNETCDF\s*=.*$|LIBNETCDF = -L${NETCDF_FORTRAN}/lib -lnetcdff -L${NETCDF_C}/lib -lnetcdf|" Makefile
    # Remove the orphaned continuation line (starts with whitespace, contains -L and NCDF_PATH)
    perl -i -ne "print unless /^\s+-L.*NCDF_PATH/" Makefile
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
