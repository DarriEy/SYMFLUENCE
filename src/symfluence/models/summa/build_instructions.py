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

# Ensure NetCDF paths are set correctly for CMake
# On Windows conda, libraries live under CONDA_PREFIX/Library (not CONDA_PREFIX).
CONDA_LIB_PREFIX="${CONDA_PREFIX}"
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        if [ -d "${CONDA_PREFIX}/Library/lib" ]; then
            CONDA_LIB_PREFIX="${CONDA_PREFIX}/Library"
        fi
        ;;
esac

if [ -n "$CONDA_PREFIX" ]; then
    export CMAKE_PREFIX_PATH="${CONDA_LIB_PREFIX}:${CMAKE_PREFIX_PATH:-}"
    export NETCDF="${NETCDF:-$CONDA_LIB_PREFIX}"
    export NETCDF_FORTRAN="${NETCDF_FORTRAN:-$CONDA_LIB_PREFIX}"
    echo "Using conda NetCDF at: $NETCDF"
fi

# Validate NetCDF installation
if [ ! -f "${NETCDF}/include/netcdf.h" ] && [ ! -f "${NETCDF}/include/netcdf.inc" ]; then
    echo "WARNING: NetCDF headers not found at ${NETCDF}/include"
    echo "Available:"
    ls -la "${NETCDF}/include"/netcdf* 2>/dev/null | head -10 || true
fi

# Determine LAPACK strategy based on platform
SPECIFY_LINKS=OFF

case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        # Windows/MinGW: link directly against conda DLLs by full path.
        # MinGW's linker can link against DLLs directly.
        # NOTE: CMake uses semicolons as list separators — spaces would be
        # treated as part of a single filename and break the build.
        if [ -f "${CONDA_LIB_PREFIX}/bin/openblas.dll" ]; then
            echo "Using OpenBLAS DLL directly (Windows/MinGW)"
            SPECIFY_LINKS=ON
            export LIBRARY_LINKS="${CONDA_LIB_PREFIX}/bin/openblas.dll"
        elif [ -f "${CONDA_LIB_PREFIX}/bin/liblapack.dll" ]; then
            echo "Using manual LAPACK specification (Windows/MinGW)"
            SPECIFY_LINKS=ON
            export LIBRARY_LINKS="${CONDA_LIB_PREFIX}/bin/liblapack.dll;${CONDA_LIB_PREFIX}/bin/libblas.dll"
        else
            echo "Using manual LAPACK specification (Windows fallback)"
            SPECIFY_LINKS=ON
            export LIBRARY_LINKS="-llapack;-lblas"
        fi
        ;;
    Darwin)
        echo "macOS detected - using manual LAPACK specification"
        SPECIFY_LINKS=ON
        export LIBRARY_LINKS='-llapack'
        ;;
    *)
        # HPC with OpenBLAS module loaded
        if command -v module >/dev/null 2>&1 && module list 2>&1 | grep -qi openblas; then
            echo "OpenBLAS module loaded - using auto-detection"
            SPECIFY_LINKS=OFF
        # Conda environment with OpenBLAS
        elif [ -n "$CONDA_PREFIX" ] && [ -f "${CONDA_LIB_PREFIX}/lib/libopenblas.so" -o -f "${CONDA_LIB_PREFIX}/lib/libopenblas.dylib" ]; then
            echo "Conda OpenBLAS found at ${CONDA_LIB_PREFIX}/lib - adding to cmake search path"
            SPECIFY_LINKS=OFF
            export CMAKE_PREFIX_PATH="${CONDA_LIB_PREFIX}:${CMAKE_PREFIX_PATH:-}"
            export LIBRARY_PATH="${CONDA_LIB_PREFIX}/lib:${LIBRARY_PATH:-}"
            export OPENBLAS_ROOT="$CONDA_LIB_PREFIX"
            export OpenBLAS_HOME="$CONDA_LIB_PREFIX"
        # Linux with system OpenBLAS
        elif pkg-config --exists openblas 2>/dev/null || [ -f "/usr/lib64/libopenblas.so" ] || [ -f "/usr/lib/libopenblas.so" ]; then
            echo "System OpenBLAS found - using auto-detection"
            SPECIFY_LINKS=OFF
        else
            # Fallback to manual LAPACK
            echo "Using manual LAPACK specification"
            SPECIFY_LINKS=ON
            export LIBRARY_LINKS="-llapack;-lblas"
        fi
        ;;
esac

# No SUMMA source patches needed — SUNDIALS is built with INDEX_SIZE=64
# to match SUMMA's INTEGER(8) declarations for sunindextype arguments.

# On Windows, SUNDIALS DLLs do not export Fortran module procedure symbols
# (only the C wrapper functions are exported). We must link against the
# static SUNDIALS libraries instead.
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        if grep -q 'fida_mod_shared' build/CMakeLists.txt 2>/dev/null; then
            echo "Patching CMakeLists.txt: using static SUNDIALS targets (Windows DLL export limitation)"
            sed -i 's/fida_mod_shared/fida_mod_static/g; s/fkinsol_mod_shared/fkinsol_mod_static/g' build/CMakeLists.txt
        fi
        ;;
esac

rm -rf cmake_build && mkdir -p cmake_build

# Build CMAKE_PREFIX_PATH with all relevant paths
SUMMA_PREFIX_PATH="$SUNDIALS_DIR"
SUMMA_EXTRA_CMAKE=""
if [ -n "${CONDA_PREFIX:-}" ]; then
    SUMMA_PREFIX_PATH="${CONDA_LIB_PREFIX};${SUMMA_PREFIX_PATH}"
    # Pass OpenBLAS hints for SUMMA's FindOpenBLAS.cmake
    if [ -n "${OPENBLAS_ROOT:-}" ]; then
        SUMMA_EXTRA_CMAKE="-DOPENBLAS_ROOT=${OPENBLAS_ROOT} -DOpenBLAS_HOME=${OPENBLAS_ROOT}"
    fi
fi

cmake -S build -B cmake_build \
  -DUSE_SUNDIALS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_FLAGS_RELEASE="-O2 -DNDEBUG" \
  -DSPECIFY_LAPACK_LINKS=$SPECIFY_LINKS \
  -DCMAKE_PREFIX_PATH="${SUMMA_PREFIX_PATH}" \
  -DSUNDIALS_ROOT="$SUNDIALS_DIR" \
  -DNETCDF_PATH="${NETCDF:-/usr}" \
  -DNETCDF_FORTRAN_PATH="${NETCDF_FORTRAN:-/usr}" \
  -DNetCDF_ROOT="${NETCDF:-/usr}" \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_Fortran_FLAGS="-ffree-form -ffree-line-length-none" \
  $SUMMA_EXTRA_CMAKE

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
