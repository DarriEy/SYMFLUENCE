"""
VIC build instructions for SYMFLUENCE.

This module defines how to build VIC from source, including:
- Repository and branch information
- Build commands (Makefile-based with MPI)
- Installation verification criteria

VIC 5.x uses Makefiles for building. The image driver is built by default,
which supports NetCDF input/output for distributed simulations.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import (
    get_common_build_environment,
    get_netcdf_detection,
)


@BuildInstructionsRegistry.register('vic')
def get_vic_build_instructions():
    """
    Get VIC build instructions.

    VIC requires NetCDF-C library and MPI. The build uses Makefiles and produces
    the vic_image.exe executable for grid-based simulations.

    Returns:
        Dictionary with complete build configuration for VIC.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()

    return {
        'description': 'Variable Infiltration Capacity Model',
        'config_path_key': 'VIC_INSTALL_PATH',
        'config_exe_key': 'VIC_EXE',
        'default_path_suffix': 'installs/vic/bin',
        'default_exe': 'vic_image.exe',
        'repository': 'https://github.com/UW-Hydro/VIC.git',
        'branch': 'develop',  # Use develop branch for VIC 5.x
        'install_dir': 'vic',
        'build_commands': [
            common_env,
            netcdf_detect,
            r'''
# VIC Build Script for SYMFLUENCE
# Builds VIC 5.x with image driver (NetCDF support)
# VIC uses Makefiles, not CMake

echo "=== VIC Build Starting ==="
echo "Building VIC image driver with NetCDF support"

# Ensure we have NetCDF
if [ -z "$NETCDF_C" ]; then
    echo "ERROR: NetCDF-C not found. Please install netcdf-c."
    exit 1
fi

echo "Using NetCDF from: $NETCDF_C"

# Platform detection
UNAME_S=$(uname -s)
echo "Platform: $UNAME_S"

# Navigate to the image driver directory
cd vic/drivers/image

if [ ! -f "Makefile" ]; then
    echo "ERROR: Makefile not found in vic/drivers/image"
    exit 1
fi

echo "Building in: $(pwd)"

# Clean any previous build
make clean 2>/dev/null || true

# Backup original Makefile
cp Makefile Makefile.orig

# VIC has global variables defined in headers causing duplicate symbols
# Add -fcommon to allow this (was default behavior before GCC 10)
sed -i.bak 's/CFLAGS  =  ${INCLUDES}/CFLAGS  =  -fcommon ${INCLUDES}/' Makefile

# Platform-specific configuration
if [ "$UNAME_S" = "Darwin" ]; then
    echo "macOS detected - configuring for clang/OpenMP..."

    # Check for MPI
    if command -v mpicc >/dev/null 2>&1; then
        export MPICC="mpicc"
        echo "Found MPI compiler: $MPICC"
    else
        echo "WARNING: mpicc not found. Install open-mpi: brew install open-mpi"
        export MPICC="clang"
    fi

    # Check for libomp (OpenMP for clang)
    LIBOMP_PATH=""
    if [ -d "/opt/homebrew/opt/libomp" ]; then
        LIBOMP_PATH="/opt/homebrew/opt/libomp"
    elif [ -d "/usr/local/opt/libomp" ]; then
        LIBOMP_PATH="/usr/local/opt/libomp"
    fi

    if [ -n "$LIBOMP_PATH" ]; then
        echo "Found libomp at: $LIBOMP_PATH"
        # Replace -fopenmp with clang-compatible version
        sed -i.bak 's/-fopenmp/-Xpreprocessor -fopenmp/g' Makefile
        # Add libomp library path
        sed -i.bak "s|LIBRARY = -lm \${NC_LIBS}|LIBRARY = -lm -L${LIBOMP_PATH}/lib -lomp \${NC_LIBS}|g" Makefile
        # Add libomp include path to NC_CFLAGS
        export NC_CFLAGS="$(nc-config --cflags 2>/dev/null || echo "-I${NETCDF_C}/include") -I${LIBOMP_PATH}/include"
    else
        echo "WARNING: libomp not found. Install it: brew install libomp"
        echo "Disabling OpenMP..."
        sed -i.bak 's/-fopenmp//g' Makefile
        export NC_CFLAGS="$(nc-config --cflags 2>/dev/null || echo "-I${NETCDF_C}/include")"
    fi

    export NC_LIBS="$(nc-config --libs 2>/dev/null || echo "-L${NETCDF_C}/lib -lnetcdf")"
else
    # Linux - use mpicc if available
    if command -v mpicc >/dev/null 2>&1; then
        export MPICC="mpicc"
        echo "Found MPI compiler: $MPICC"
    elif command -v gcc >/dev/null 2>&1; then
        export MPICC="gcc"
        echo "Using gcc (parallel execution disabled)"
    else
        echo "ERROR: No C compiler found (need mpicc or gcc)"
        exit 1
    fi

    export NC_LIBS="$(nc-config --libs 2>/dev/null || echo "-L${NETCDF_C}/lib -lnetcdf")"
    export NC_CFLAGS="$(nc-config --cflags 2>/dev/null || echo "-I${NETCDF_C}/include")"
fi

echo "NC_LIBS: $NC_LIBS"
echo "NC_CFLAGS: $NC_CFLAGS"
echo "MPICC: $MPICC"

# Build VIC
echo "Running make..."
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
make -j${NCORES}

# Check build result
if [ $? -ne 0 ]; then
    echo "VIC build failed"
    exit 1
fi

# Check if executable was created
if [ ! -f "vic_image.exe" ]; then
    echo "ERROR: vic_image.exe not found after build"
    ls -la
    exit 1
fi

echo "Build successful!"

# Create bin directory and install
mkdir -p ../../../bin
cp vic_image.exe ../../../bin/
chmod +x ../../../bin/vic_image.exe

echo "=== VIC Build Complete ==="
echo "Installed to: bin/vic_image.exe"

# Verify installation
if [ -f "../../../bin/vic_image.exe" ]; then
    echo "Verification: vic_image.exe exists"
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['nc-config', 'mpicc'],
        'test_command': 'vic_image.exe -v',
        'verify_install': {
            'file_paths': ['bin/vic_image.exe'],
            'check_type': 'exists'
        },
        'order': 15  # After core models
    }
