"""
mHM build instructions for SYMFLUENCE.

This module defines how to build mHM from source, including:
- Repository and branch information
- Build commands (CMake + gfortran + NetCDF-Fortran)
- Installation verification criteria

mHM (mesoscale Hydrological Model) is built using CMake with Fortran support.
It requires gfortran and NetCDF-Fortran libraries.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import (
    get_common_build_environment,
    get_netcdf_detection,
)


@BuildInstructionsRegistry.register('mhm')
def get_mhm_build_instructions():
    """
    Get mHM build instructions.

    mHM requires CMake, gfortran, and NetCDF-Fortran for building.
    The build produces the `mhm` executable.

    Returns:
        Dictionary with complete build configuration for mHM.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()

    return {
        'description': 'mesoscale Hydrological Model (mHM)',
        'config_path_key': 'MHM_INSTALL_PATH',
        'config_exe_key': 'MHM_EXE',
        'default_path_suffix': 'installs/mhm/bin',
        'default_exe': 'mhm',
        'repository': 'https://git.ufz.de/mhm/mhm.git',
        'branch': 'main',
        'install_dir': 'mhm',
        'build_commands': [
            common_env,
            netcdf_detect,
            r'''
# mHM Build Script for SYMFLUENCE
# Builds mHM using CMake + gfortran + NetCDF-Fortran

set -e

echo "=== mHM Build Starting ==="
echo "Building mHM with CMake + gfortran + NetCDF-Fortran"

# Ensure we have NetCDF
if [ -z "$NETCDF_C" ]; then
    echo "ERROR: NetCDF-C not found. Please install netcdf-c."
    exit 1
fi

echo "Using NetCDF from: $NETCDF_C"

# Platform detection
UNAME_S=$(uname -s)
echo "Platform: $UNAME_S"

# Check for required tools
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: CMake not found. Please install cmake."
    exit 1
fi

if ! command -v gfortran >/dev/null 2>&1; then
    echo "ERROR: gfortran not found. Please install gfortran."
    exit 1
fi

echo "CMake version: $(cmake --version | head -1)"
echo "gfortran version: $(gfortran --version | head -1)"

# Check for NetCDF-Fortran
NETCDF_FORTRAN=""
if command -v nf-config >/dev/null 2>&1; then
    NETCDF_FORTRAN="$(nf-config --prefix 2>/dev/null || echo '')"
elif [ -d "/opt/homebrew/opt/netcdf-fortran" ]; then
    NETCDF_FORTRAN="/opt/homebrew/opt/netcdf-fortran"
elif [ -d "/usr/local/opt/netcdf-fortran" ]; then
    NETCDF_FORTRAN="/usr/local/opt/netcdf-fortran"
elif [ -d "/usr" ] && [ -f "/usr/include/netcdf.mod" ]; then
    NETCDF_FORTRAN="/usr"
fi

if [ -z "$NETCDF_FORTRAN" ]; then
    echo "ERROR: NetCDF-Fortran not found. Please install netcdf-fortran."
    echo "  macOS: brew install netcdf-fortran"
    echo "  Ubuntu: apt install libnetcdff-dev"
    exit 1
fi

echo "Using NetCDF-Fortran from: $NETCDF_FORTRAN"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring mHM with CMake..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=../install
    -DCMAKE_Fortran_COMPILER=gfortran
)

# Add NetCDF paths if available
if [ -n "$NETCDF_C" ]; then
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="${NETCDF_C};${NETCDF_FORTRAN}")
fi

# Platform-specific flags
if [ "$UNAME_S" = "Darwin" ]; then
    echo "macOS detected - adding macOS-specific flags..."
    CMAKE_ARGS+=(-DCMAKE_Fortran_FLAGS="-ffree-line-length-none -cpp")
else
    CMAKE_ARGS+=(-DCMAKE_Fortran_FLAGS="-ffree-line-length-none -cpp")
fi

cmake "${CMAKE_ARGS[@]}" ..

if [ $? -ne 0 ]; then
    echo "CMake configuration failed"
    exit 1
fi

# Build mHM
echo "Building mHM..."
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
cmake --build . -j${NCORES}

if [ $? -ne 0 ]; then
    echo "mHM build failed"
    exit 1
fi

echo "Build successful!"

# Install
cmake --install . 2>/dev/null || true

# Find and copy executable
MHM_EXE=""
for candidate in mhm mHM mhm.exe; do
    if [ -f "$candidate" ]; then
        MHM_EXE="$candidate"
        break
    fi
    if [ -f "src/$candidate" ]; then
        MHM_EXE="src/$candidate"
        break
    fi
done

if [ -z "$MHM_EXE" ]; then
    echo "ERROR: mHM executable not found after build"
    find . -name "mhm*" -type f 2>/dev/null
    exit 1
fi

# Create bin directory and install
mkdir -p ../bin
cp "$MHM_EXE" ../bin/mhm
chmod +x ../bin/mhm

echo "=== mHM Build Complete ==="
echo "Installed to: bin/mhm"

# Verify installation
if [ -f "../bin/mhm" ]; then
    echo "Verification: mhm executable exists"
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'gfortran', 'nf-config'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/mhm'],
            'check_type': 'exists'
        },
        'order': 16,  # After VIC
        'optional': True,  # Not installed by default with --install
    }
