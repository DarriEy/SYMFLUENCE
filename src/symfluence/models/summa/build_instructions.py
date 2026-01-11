"""
SUMMA build instructions for SYMFLUENCE.

This module defines how to build SUMMA from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria
- Dependencies (requires SUNDIALS)

SUMMA (Structure for Unifying Multiple Modeling Alternatives) is a
land surface model that uses SUNDIALS for solving differential equations.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import get_common_build_environment


@BuildInstructionsRegistry.register('summa')
def get_summa_build_instructions():
    """
    Get SUMMA build instructions.

    SUMMA requires SUNDIALS to be installed first. The build uses CMake
    and links against NetCDF and LAPACK.

    Returns:
        Dictionary with complete build configuration for SUMMA.
    """
    common_env = get_common_build_environment()

    return {
        'description': 'Structure for Unifying Multiple Modeling Alternatives (with SUNDIALS)',
        'config_path_key': 'SUMMA_INSTALL_PATH',
        'config_exe_key': 'SUMMA_EXE',
        'default_path_suffix': 'installs/summa/bin',
        'default_exe': 'summa_sundials.exe',
        'repository': 'https://github.com/CH-Earth/summa.git',
        'branch': 'develop_sundials',
        'install_dir': 'summa',
        'requires': ['sundials'],
        'build_commands': [
            common_env,
            r'''
# Build SUMMA against SUNDIALS + NetCDF, leverage SUMMA's CMake-based build
set -e

export SUNDIALS_DIR="$(realpath ../sundials/install/sundials)"
echo "Using SUNDIALS from: $SUNDIALS_DIR"

# Determine LAPACK strategy based on platform
SPECIFY_LINKS=OFF

# macOS: Use manual LAPACK specification (Homebrew OpenBLAS isn't reliably detected by CMake)
if [ "$(uname)" == "Darwin" ]; then
    echo "macOS detected - using manual LAPACK specification"
    SPECIFY_LINKS=ON
    export LIBRARY_LINKS='-llapack'
# HPC with OpenBLAS module loaded
elif command -v module >/dev/null 2>&1 && module list 2>&1 | grep -qi openblas; then
    echo "OpenBLAS module loaded - using auto-detection"
    SPECIFY_LINKS=OFF
# Linux with system OpenBLAS
elif pkg-config --exists openblas 2>/dev/null || [ -f "/usr/lib64/libopenblas.so" ] || [ -f "/usr/lib/libopenblas.so" ]; then
    echo "System OpenBLAS found - using auto-detection"
    SPECIFY_LINKS=OFF
else
    # Fallback to manual LAPACK
    echo "Using manual LAPACK specification"
    SPECIFY_LINKS=ON
    export LIBRARY_LINKS='-llapack -lblas'
fi

rm -rf cmake_build && mkdir -p cmake_build

cmake -S build -B cmake_build \
  -DUSE_SUNDIALS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPECIFY_LAPACK_LINKS=$SPECIFY_LINKS \
  -DCMAKE_PREFIX_PATH="$SUNDIALS_DIR" \
  -DSUNDIALS_ROOT="$SUNDIALS_DIR" \
  -DNETCDF_PATH="${NETCDF:-/usr}" \
  -DNETCDF_FORTRAN_PATH="${NETCDF_FORTRAN:-/usr}" \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_Fortran_FLAGS="-ffree-form -ffree-line-length-none"

# Build all targets (repo scripts use 'all', not just 'summa_sundials')
cmake --build cmake_build --target all -j ${NCORES:-4}

# Stage binary into bin/ and provide standard name
if [ -f "bin/summa_sundials.exe" ]; then
    cd bin
    ln -sf summa_sundials.exe summa.exe
    cd ..
elif [ -f "cmake_build/bin/summa_sundials.exe" ]; then
    mkdir -p bin
    cp cmake_build/bin/summa_sundials.exe bin/
    cd bin
    ln -sf summa_sundials.exe summa.exe
    cd ..
elif [ -f "cmake_build/bin/summa.exe" ]; then
    mkdir -p bin
    cp cmake_build/bin/summa.exe bin/
fi
            '''.strip()
        ],
        'dependencies': [],
        'test_command': '--version',
        'verify_install': {
            'file_paths': [
                'bin/summa.exe',
                'bin/summa_sundials.exe',
                'cmake_build/bin/summa.exe',
                'cmake_build/bin/summa_sundials.exe'
            ],
            'check_type': 'exists_any'
        },
        'order': 2
    }
