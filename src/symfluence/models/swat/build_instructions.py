"""
SWAT build instructions for SYMFLUENCE.

This module defines how to build SWAT from source, including:
- Repository and branch information
- Build commands (CMake + gfortran)
- Installation verification criteria

SWAT is built from Fortran source code using CMake + gfortran. The build
produces the swat_rel.exe executable for watershed simulations.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import get_common_build_environment


@BuildInstructionsRegistry.register('swat')
def get_swat_build_instructions():
    """
    Get SWAT build instructions.

    SWAT is compiled from Fortran source using CMake + gfortran.
    The repo includes a CMakeLists.txt that handles module dependency
    ordering (modparm.f must be compiled before other sources).

    Returns:
        Dictionary with complete build configuration for SWAT.
    """
    common_env = get_common_build_environment()

    return {
        'description': 'Soil and Water Assessment Tool',
        'config_path_key': 'SWAT_INSTALL_PATH',
        'config_exe_key': 'SWAT_EXE',
        'default_path_suffix': 'installs/swat/bin',
        'default_exe': 'swat_rel.exe',
        'repository': 'https://github.com/WatershedModels/SWAT.git',
        'branch': 'master',
        'install_dir': 'swat',
        'build_commands': [
            common_env,
            r'''
# SWAT Build Script for SYMFLUENCE
# Builds SWAT using CMake + gfortran (handles module dependencies)

set -e

echo "=== SWAT Build Starting ==="
echo "Building SWAT with CMake + gfortran"

# Check for required tools
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: CMake not found. Please install cmake."
    echo "  macOS: brew install cmake"
    echo "  Ubuntu: sudo apt-get install cmake"
    exit 1
fi

if ! command -v gfortran >/dev/null 2>&1; then
    echo "ERROR: gfortran not found. Please install gfortran."
    echo "  macOS: brew install gcc"
    echo "  Ubuntu: sudo apt-get install gfortran"
    exit 1
fi

echo "CMake version: $(cmake --version | head -1)"
echo "gfortran version: $(gfortran --version | head -1)"

# Platform detection
UNAME_S=$(uname -s)
echo "Platform: $UNAME_S"

# Set up Fortran flags for gfortran 10+ compatibility
EXTRA_FFLAGS="-fno-automatic -fno-align-commons"
GFORTRAN_MAJOR=$(gfortran -dumpversion | cut -d. -f1)
if [ "$GFORTRAN_MAJOR" -ge 10 ] 2>/dev/null; then
    echo "gfortran >= 10 detected, adding -fallow-argument-mismatch"
    EXTRA_FFLAGS="$EXTRA_FFLAGS -fallow-argument-mismatch"
fi

# Patch known source code bugs before building
echo "Applying source code patches..."

# Fix missing comma in std3.f format descriptor (line 47)
# Original: t97'------   Fixed: t97,'------
if grep -q "t97'" src/std3.f 2>/dev/null; then
    sed -i.bak "s/t97'/t97,'/" src/std3.f
    echo "  Patched std3.f: missing comma in format 1300"
fi

# Remove -static flag for HPC compatibility when static libc/libm are unavailable.
# On Spack/module-based HPC systems, static libc.a and libm.a are typically not
# installed, so -static linking fails with "cannot find -lm".
# Only patch when static linking is actually broken to avoid regressions on
# desktop Linux/macOS where static libs may be available.
# Use gfortran -print-file-name which queries the actual linker search paths:
# returns the full path if found, or just the bare filename if not found.
_libc_a="$(gfortran -print-file-name=libc.a 2>/dev/null)"
_libm_a="$(gfortran -print-file-name=libm.a 2>/dev/null)"
if [ "$_libc_a" = "libc.a" ] || [ ! -f "$_libc_a" ] || \
   [ "$_libm_a" = "libm.a" ] || [ ! -f "$_libm_a" ]; then
    echo "  Static libc/libm not found — removing -static for HPC compatibility"
    for cmf in CMakeLists.txt src/CMakeLists.txt; do
        if [ -f "$cmf" ] && grep -q '\-static' "$cmf" 2>/dev/null; then
            sed -i.bak 's/-static //g; s/ -static//g' "$cmf"
            echo "  Patched $cmf: removed -static flag"
        fi
    done
else
    echo "  Static libc/libm available, keeping -static flags"
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring SWAT with CMake..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_Fortran_COMPILER=gfortran
    -DCMAKE_INSTALL_PREFIX=../install
    "-DCMAKE_Fortran_FLAGS_RELEASE=-O2 $EXTRA_FFLAGS"
)

cmake "${CMAKE_ARGS[@]}" ..

# Build
echo "Building SWAT..."
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
cmake --build . --config Release -j${NCORES}

echo "Build successful!"

# Install via CMake install target
cmake --install . 2>/dev/null || true

# Find the built executable
SWAT_EXE=""
for candidate in swat swat.exe src/swat src/swat.exe; do
    if [ -f "$candidate" ]; then
        SWAT_EXE="$candidate"
        break
    fi
done

# Also check install directory
if [ -z "$SWAT_EXE" ]; then
    for candidate in ../install/bin/swat ../install/bin/swat.exe; do
        if [ -f "$candidate" ]; then
            SWAT_EXE="$candidate"
            break
        fi
    done
fi

# Search recursively as fallback
if [ -z "$SWAT_EXE" ]; then
    echo "Searching for SWAT executable..."
    SWAT_EXE=$(find . ../install -name "swat*" -type f -perm +111 2>/dev/null | head -1)
fi

if [ -z "$SWAT_EXE" ]; then
    echo "ERROR: SWAT executable not found after build"
    find . -type f -name "swat*" 2>/dev/null
    ls -la
    exit 1
fi

echo "Found executable: $SWAT_EXE"

# Create bin directory and install as swat_rel.exe
mkdir -p ../bin
cp "$SWAT_EXE" ../bin/swat_rel.exe
chmod +x ../bin/swat_rel.exe

echo "=== SWAT Build Complete ==="
echo "Installed to: bin/swat_rel.exe"

# Verify installation
if [ -f "../bin/swat_rel.exe" ]; then
    echo "Verification: swat_rel.exe exists"
    ls -la ../bin/swat_rel.exe
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'gfortran'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/swat_rel.exe'],
            'check_type': 'exists'
        },
        'order': 16  # After VIC
    }
